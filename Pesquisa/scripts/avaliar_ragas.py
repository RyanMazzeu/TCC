"""
Avaliação RAGAS sobre um log de experimentos já rodado (ex.: comparativo_6_metodos_v3.jsonl).

Métricas calculadas:
- answer_relevancy: pergunta + resposta (não depende de contexto) -- calculada para todas as linhas.
- faithfulness, context_precision, context_recall: pergunta + resposta + contextos + gabarito --
  só fazem sentido para linhas onde "contextos_recuperados" são trechos de documento de verdade
  (Stuff/Refine/Map-Reduce/Map-Rerank). Query Step-Down e Reciprocal logam ali as PERGUNTAS
  ALTERNATIVAS geradas, não trechos recuperados -- essas linhas são puladas nessas 3 métricas
  (fica None no resultado, não um valor inventado).

Juiz LLM (--juiz):
- "gpt-4o-mini" (default): o mesmo modelo que gerou as respostas -- barato, mesmo precedente do
  paper do IFES/SBC, mas com risco de viés de autopreferência (o modelo julgando a si mesmo).
- "cohere:<modelo>" (ex.: cohere:command-a-03-2025): juiz de outra família, via API compatível
  com OpenAI da Cohere. Usado numa amostra estratificada (--amostra-por-grupo) pra medir o quanto
  as notas do gpt-4o-mini concordam com as de um juiz independente.

O embedding de answer_relevancy é sempre o text-embedding-3-small da OpenAI, pros dois juízes --
assim a única coisa que muda entre eles é o LLM.

max_tokens do juiz = 4096 (o default do RAGAS, 1024, truncava a saída estruturada do Faithfulness
em respostas longas do Refine/Map-Reduce -- 5 das 6 falhas da rodada sobre o v2).
"""

import argparse
import asyncio
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

# A telemetria de uso do RAGAS faz uma chamada de rede bloqueante a cada avaliação: medido aqui,
# ~9 s por chamada ao juiz com ela ligada vs ~0,6 s desligada. Precisa vir antes do import do ragas.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from dotenv import load_dotenv
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from ragas.embeddings.base import embedding_factory
from ragas.llms.base import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR.parent / ".env")

METODOS_SEM_CONTEXTO_REAL = {"query_step_down", "reciprocal"}
COHERE_BASE_URL = "https://api.cohere.ai/compatibility/v1"
MAX_TOKENS_JUIZ = 4096


class LimitadorDeTaxa:
    """Espaça as requisições HTTP de um cliente para no máximo `por_minuto` por minuto. Sem isso,
    linhas e métricas em paralelo estouram o limite de tokens por minuto da OpenAI (200 mil na conta
    usada) e o de 20 chamadas/min da chave Trial da Cohere -- e cada 429 que esgota as tentativas
    vira métrica ausente."""

    def __init__(self, por_minuto: float):
        self.intervalo = 60.0 / por_minuto
        self.proxima = 0.0
        self.trava = asyncio.Lock()

    async def __call__(self, _request):
        async with self.trava:
            agora = time.monotonic()
            if self.proxima > agora:
                await asyncio.sleep(self.proxima - agora)
            self.proxima = max(agora, self.proxima) + self.intervalo


def _cliente(por_minuto: float, **kwargs) -> AsyncOpenAI:
    http = DefaultAsyncHttpxClient(event_hooks={"request": [LimitadorDeTaxa(por_minuto)]})
    return AsyncOpenAI(http_client=http, max_retries=8, **kwargs)


def montar_metricas(juiz: str, por_minuto: float) -> dict:
    if juiz.startswith("cohere:"):
        cliente_juiz = _cliente(por_minuto, base_url=COHERE_BASE_URL, api_key=os.environ["COHERE_API_KEY"])
        cliente_openai = _cliente(60)  # só os embeddings da Answer Relevancy
        modelo_juiz = juiz.removeprefix("cohere:")
    else:
        cliente_openai = cliente_juiz = _cliente(por_minuto)
        modelo_juiz = juiz
    llm = llm_factory(modelo_juiz, client=cliente_juiz, max_tokens=MAX_TOKENS_JUIZ)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=cliente_openai)
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


async def _pontuar_com_seguranca(nome: str, corotina, resultado: dict):
    """Roda uma chamada de métrica RAGAS sem deixar uma falha isolada (ex.: estouro de
    max_tokens em respostas muito longas do Refine) derrubar o lote inteiro. Em erro, grava
    None + o tipo/mensagem do erro num campo `<nome>_erro`, em vez de inventar um valor."""
    try:
        resultado[nome] = (await corotina).value
    except Exception as e:
        resultado[nome] = None
        resultado[f"{nome}_erro"] = f"{type(e).__name__}: {e}"


async def avaliar_linha(registro: dict, metricas: dict, somente: set[str] | None = None) -> dict:
    """`somente`: calcula só essas métricas (usado pra reparar as que falharam numa linha já gravada)."""
    pergunta = registro["pergunta"]
    resposta = registro["resposta_gerada"]
    esperada = registro.get("resposta_esperada")
    contextos = registro.get("contextos_recuperados") or []
    tem_contexto_real = registro.get("metodo_rag") not in METODOS_SEM_CONTEXTO_REAL and bool(contextos)

    resultado = {} if somente else {"faithfulness": None, "context_precision": None, "context_recall": None}
    chamadas = [
        ("answer_relevancy", metricas["answer_relevancy"].ascore(user_input=pergunta, response=resposta)),
    ]
    if tem_contexto_real:
        chamadas.append((
            "faithfulness",
            metricas["faithfulness"].ascore(user_input=pergunta, response=resposta, retrieved_contexts=contextos),
        ))
        if esperada:
            chamadas.append((
                "context_precision",
                metricas["context_precision"].ascore(user_input=pergunta, reference=esperada, retrieved_contexts=contextos),
            ))
            chamadas.append((
                "context_recall",
                metricas["context_recall"].ascore(user_input=pergunta, retrieved_contexts=contextos, reference=esperada),
            ))

    if somente:
        for nome, corotina in chamadas:
            if nome not in somente:
                corotina.close()  # descartada sem rodar (evita o aviso de corrotina nunca aguardada)
        chamadas = [(nome, corotina) for nome, corotina in chamadas if nome in somente]
    # As 4 métricas são independentes entre si -- em paralelo, a linha leva o tempo da mais lenta
    # (~45 s, Answer Relevancy) em vez da soma (~110 s).
    await asyncio.gather(*(_pontuar_com_seguranca(nome, corotina, resultado) for nome, corotina in chamadas))
    return resultado


def amostrar_por_grupo(linhas: list[dict], n: int, seed: int) -> list[dict]:
    """Amostra estratificada: n linhas por (domínio, método), sorteadas com seed fixa, mantendo a
    ordem original do log -- assim a retomada (que conta linhas já feitas) continua valendo."""
    grupos = defaultdict(list)
    for i, r in enumerate(linhas):
        grupos[(r.get("dominio"), r["metodo_rag"])].append(i)
    rng = random.Random(seed)
    escolhidos = sorted(i for idxs in grupos.values() for i in rng.sample(idxs, min(n, len(idxs))))
    return [linhas[i] for i in escolhidos]


async def reparar(saida_path: Path, metricas: dict, concorrencia: int):
    """Reavalia só as métricas que falharam (campo <métrica>_erro) nas linhas já gravadas -- ex.: um
    429 de limite de taxa -- e regrava o arquivo. As métricas que deram certo não são pagas de novo."""
    registros = [json.loads(l) for l in saida_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    pendentes = [r for r in registros if any(k.endswith("_erro") for k in r["ragas"])]
    if not pendentes:
        return
    print(f"Reparando {len(pendentes)} linhas com métricas que falharam.")

    async def _reparar(r):
        falhas = {k.removesuffix("_erro") for k in r["ragas"] if k.endswith("_erro")}
        for k in falhas:
            r["ragas"].pop(f"{k}_erro")
        r["ragas"].update(await avaliar_linha(r, metricas, somente=falhas))

    for inicio in range(0, len(pendentes), concorrencia):
        await asyncio.gather(*(_reparar(r) for r in pendentes[inicio : inicio + concorrencia]))
        temporario = saida_path.with_suffix(".tmp")
        temporario.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in registros), encoding="utf-8")
        temporario.replace(saida_path)
        print(f"  reparadas {min(inicio + concorrencia, len(pendentes))}/{len(pendentes)}")


async def main(
    log_path: Path, saida_path: Path, limite: int | None, juiz: str, amostra_por_grupo: int | None, seed: int,
    concorrencia: int, por_minuto: float,
):
    linhas = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if amostra_por_grupo:
        linhas = amostrar_por_grupo(linhas, amostra_por_grupo, seed)
    if limite:
        linhas = linhas[:limite]
    metricas = montar_metricas(juiz, por_minuto)
    if saida_path.exists():
        await reparar(saida_path, metricas, concorrencia)

    # Retomada: se a saída já existe (ex.: processo anterior foi interrompido no meio), pula as
    # linhas já avaliadas em vez de pagar de novo -- assume que a ordem das linhas não mudou.
    ja_feitas = 0
    if saida_path.exists():
        ja_feitas = sum(1 for l in saida_path.read_text(encoding="utf-8").splitlines() if l.strip())
        if ja_feitas:
            print(f"Retomando: {ja_feitas} linhas já avaliadas em {saida_path}, pulando essas.")

    async def _avaliar_protegido(registro):
        try:
            return await avaliar_linha(registro, metricas)
        except Exception as e:
            return {"erro_linha": f"{type(e).__name__}: {e}"}

    # Lotes de `concorrencia` linhas avaliadas em paralelo, mas gravadas na ordem original -- a
    # retomada continua só contando linhas. Uma linha isolada tem 4 métricas em sequência (várias
    # chamadas ao juiz cada), então rodar linha a linha levava ~1 min por linha.
    with open(saida_path, "a", encoding="utf-8") as f:
        for inicio in range(ja_feitas, len(linhas), concorrencia):
            lote = linhas[inicio : inicio + concorrencia]
            resultados = await asyncio.gather(*(_avaliar_protegido(r) for r in lote))
            for offset, (registro, scores) in enumerate(zip(lote, resultados)):
                registro["ragas"] = scores
                registro["ragas_juiz"] = juiz
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
                rotulo = f"{registro.get('dominio', registro.get('document_id'))}/{registro['metodo_rag']}"
                print(f"[{inicio + offset + 1}/{len(linhas)}] {rotulo} {registro.get('pergunta_id', '')}: {scores}")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Caminho do .jsonl de experimento já rodado")
    parser.add_argument("--saida", default=None, help="Caminho de saída (default: <log>_ragas.jsonl)")
    parser.add_argument("--limite", type=int, default=None, help="Avaliar só as N primeiras linhas (teste rápido)")
    parser.add_argument("--juiz", default="gpt-4o-mini", help='"gpt-4o-mini" ou "cohere:<modelo>"')
    parser.add_argument("--amostra-por-grupo", type=int, default=None, help="Avaliar só N linhas por (domínio, método)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concorrencia", type=int, default=8, help="Linhas avaliadas em paralelo")
    parser.add_argument("--por-minuto", type=float, default=50, help="Máximo de requisições por minuto ao juiz")
    args = parser.parse_args()

    log_path = Path(args.log)
    saida_path = Path(args.saida) if args.saida else log_path.with_name(log_path.stem + "_ragas.jsonl")
    asyncio.run(main(log_path, saida_path, args.limite, args.juiz, args.amostra_por_grupo, args.seed, args.concorrencia, args.por_minuto))
