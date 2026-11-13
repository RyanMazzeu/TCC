"""
Para cada resposta dos métodos de RAG (repetição 1, todos os LLMs, Docugami e FaQuAD), decide se o
contexto que o LLM recebeu tinha a informação que responde à pergunta, e cruza isso com o que o modelo
fez (recusou ou respondeu). Isso separa a falha da recuperação da falha da geração:

- recusa justificada: o contexto não tinha a informação (ou tinha só parte dela) e o modelo recusou;
- recusa injustificada: o contexto tinha toda a informação e o modelo ainda disse que não sabia;
- resposta sem a informação: o contexto não tinha nada do que responde à pergunta e o modelo respondeu
  assim mesmo (com o próprio conhecimento ou inventando; a métrica principal diz qual dos dois);
- rejeição negativa (Chen et al. 2024, RGB): dos casos sem a informação, a fração em que o modelo
  recusou.

O contexto é o que o método entregou ao LLM: os 4 fragmentos da pergunta original no Stuff, Refine,
Map-Reduce e Map-Rerank, e a união dos fragmentos das perguntas alternativas no Query Step-Down e no
Reciprocal. Os logs desses dois guardam só as perguntas alternativas, e a busca é refeita com elas no
mesmo índice; o script confere antes que a busca reproduz os ids gravados do Stuff em todas as perguntas.

Como saber se a informação estava no contexto: um juiz (gpt-4o-mini, temperatura 0, o mesmo do RAGAS)
recebe a pergunta, a resposta de referência e o contexto e diz se o contexto tem toda a informação
("completa"), só parte ("parcial") ou nenhuma ("ausente"). No Docugami não há critério exato: a
referência é texto livre e costuma juntar valores de vários relatórios. No FaQuAD há um, que serve de
validação do juiz: as respostas aceitas são trechos copiados do parágrafo, e basta procurar alguma
delas, normalizada, no contexto. Ele não é usado como critério porque erra para o outro lado: uma
resposta de uma palavra ("custos", "algoritmo") aparece no contexto em outra frase, sem responder à
pergunta.

O controle sem RAG fica de fora (não há contexto), e o MedQA também: o prompt manda sempre escolher uma
alternativa (houve só 3 recusas), e o gabarito não marca onde está a resposta nos livros.

Saídas (retoma de onde parou; só chama o juiz para contextos ainda não julgados):
  resultados/metricas/contextos_juiz.jsonl        veredito do juiz por (pergunta, contexto)
  resultados/metricas/contextos_classificados.jsonl uma linha por resposta
Uso: python classificar_contextos.py
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

import pandas as pd
from openai import OpenAI

import analisar_benchmarks as ab
import benchmarks
import metricas_resposta as mr
import pipeline_base as pb

DATASETS = ["docugami_10q", "faquad"]
JUIZ = "gpt-4o-mini"
SAIDA_JUIZ = pb.METRICAS_DIR / "contextos_juiz.jsonl"
SAIDA = pb.METRICAS_DIR / "contextos_classificados.jsonl"
METODOS_MULTIPLAS_BUSCAS = {"query_step_down", "reciprocal"}

PROMPT_JUIZ = """Você avalia a recuperação de um sistema de RAG. Recebe uma pergunta, a resposta de referência (correta) e os fragmentos de texto que o sistema recuperou.

Decida se os fragmentos contêm a informação necessária para dar a resposta de referência. Considere só a informação que de fato responde à pergunta (os valores, fatos ou nomes pedidos), e não comentários gerais da referência. A informação pode estar espalhada por vários fragmentos, e números de tabela podem aparecer sem o símbolo de moeda ou a escala.

- "completa": os fragmentos têm toda a informação que responde à pergunta.
- "parcial": têm parte dela (por exemplo, os valores de alguns trimestres ou empresas, mas não de todos).
- "ausente": não têm a informação pedida.

PERGUNTA:
{pergunta}

RESPOSTA DE REFERÊNCIA:
{referencia}

FRAGMENTOS RECUPERADOS:
{fragmentos}

Responda só com um JSON: {{"veredito": "completa" | "parcial" | "ausente", "justificativa": "<uma frase>"}}"""


def _hash(textos: list[str]) -> str:
    return hashlib.sha1("\x00".join(sorted(set(textos))).encode("utf-8")).hexdigest()[:16]


def carregar_respostas() -> pd.DataFrame:
    df = ab.carregar_execucoes()
    return df[(df.repeticao == 1) & df.dataset.isin(DATASETS) & df.metodo_rag.isin(ab.METODOS_RAG)].reset_index(drop=True)


def reconstruir_contextos(resp: pd.DataFrame, colecoes) -> list[list[str]]:
    """Fragmentos entregues ao LLM em cada resposta. No Step-Down e no Reciprocal, refaz a busca com as
    perguntas alternativas gravadas no log (embeddings em lote); antes, confere que a mesma busca em lote
    reproduz os ids gravados do Stuff em todas as perguntas."""
    modelo = pb.get_embedding_model()
    stuff = resp[resp.metodo_rag == "stuff"].drop_duplicates(["dataset", "pergunta_id"])
    multiplas = resp[resp.metodo_rag.isin(METODOS_MULTIPLAS_BUSCAS)]
    consultas = {(r.dataset, r.pergunta) for r in stuff.itertuples()}
    consultas |= {(r.dataset, alt) for r in multiplas.itertuples() for alt in r.contextos_recuperados}
    busca = {}
    for nome in DATASETS:
        textos = sorted(q for d, q in consultas if d == nome)
        vetores = pb._encode_documentos(modelo, textos)
        for texto, vetor in zip(textos, vetores):
            busca[nome, texto] = pb.consultar_indice(colecoes[nome], texto, k=4, vetor_query=vetor[None, :])
        print(f"{nome}: {len(textos)} buscas refeitas")
    divergentes = [r.pergunta_id for r in stuff.itertuples() if busca[r.dataset, r.pergunta]["ids"] != list(r.ids_recuperados)]
    if divergentes:
        raise RuntimeError(f"A busca não reproduz o log do Stuff em {len(divergentes)} perguntas: {divergentes[:5]}")
    print(f"busca conferida: reproduz os ids do Stuff nas {len(stuff)} perguntas")

    contextos = []
    for r in resp.itertuples():
        if r.metodo_rag in METODOS_MULTIPLAS_BUSCAS:
            textos = [t for alt in r.contextos_recuperados for t in busca[r.dataset, alt]["documentos"]]
            contextos.append(list(dict.fromkeys(textos)))
        else:
            contextos.append(list(r.contextos_recuperados))
    return contextos


def resposta_no_contexto(contexto: list[str], aceitas: list[str]) -> bool:
    texto = mr._normalizar(" ".join(contexto))
    return any(mr._normalizar(a) and f" {mr._normalizar(a)} " in f" {texto} " for a in aceitas)


def julgar(cliente, pergunta: str, referencia: str, contexto: list[str]) -> dict:
    fragmentos = "\n\n---\n\n".join(contexto)
    resposta = cliente.chat.completions.create(
        model=JUIZ, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": PROMPT_JUIZ.format(pergunta=pergunta, referencia=referencia, fragmentos=fragmentos)}],
    )
    saida = json.loads(resposta.choices[0].message.content)
    if saida.get("veredito") not in {"completa", "parcial", "ausente"}:
        raise ValueError(f"veredito inválido: {saida}")
    return {"veredito": saida["veredito"], "justificativa": saida.get("justificativa", ""),
            "tokens_prompt": resposta.usage.prompt_tokens, "tokens_gerados": resposta.usage.completion_tokens}


def main():
    resp = carregar_respostas()
    perguntas = {n: {p["id"]: p for p in benchmarks.carregar_perguntas(n)} for n in DATASETS}
    colecoes = {}
    for nome in DATASETS:
        chunks, _ = benchmarks.carregar(nome)
        colecoes[nome] = benchmarks.obter_indice(nome, chunks)
    resp["contexto"] = reconstruir_contextos(resp, colecoes)
    resp["chave"] = [f"{d}|{p}|{_hash(c)}" for d, p, c in zip(resp.dataset, resp.pergunta_id, resp.contexto)]

    julgados = {}
    if SAIDA_JUIZ.exists():
        julgados = {j["chave"]: j for j in map(json.loads, SAIDA_JUIZ.read_text(encoding="utf-8").splitlines()) if j}
    pendentes = resp.drop_duplicates("chave")
    pendentes = pendentes[~pendentes.chave.isin(julgados)]
    print(f"{len(resp)} respostas, {resp.chave.nunique()} contextos distintos, {len(pendentes)} a julgar")
    if len(pendentes):
        cliente = OpenAI()

        def _um(r):
            j = julgar(cliente, r.pergunta, perguntas[r.dataset][r.pergunta_id]["resposta_esperada"], r.contexto)
            return {"chave": r.chave, "dataset": r.dataset, "pergunta_id": r.pergunta_id, "juiz": JUIZ, **j}

        with ThreadPoolExecutor(max_workers=16) as pool, SAIDA_JUIZ.open("a", encoding="utf-8") as arq:
            for n, j in enumerate(pool.map(_um, [r for _, r in pendentes.iterrows()]), start=1):
                julgados[j["chave"]] = j
                arq.write(json.dumps(j, ensure_ascii=False) + "\n")
                arq.flush()
                if n % 200 == 0:
                    print(f"  juiz: {n}/{len(pendentes)}", flush=True)

    linhas = []
    for r in resp.itertuples():
        p = perguntas[r.dataset][r.pergunta_id]
        col = ab.METRICA_PRINCIPAL[r.dataset][0]
        metrica = getattr(r, col)
        linhas.append({"dataset": r.dataset, "llm": r.llm, "camada": r.camada, "metodo_rag": r.metodo_rag,
                       "pergunta_id": r.pergunta_id, "n_fragmentos": len(r.contexto), "chave_contexto": r.chave,
                       "veredito_juiz": julgados[r.chave]["veredito"], "recusa": bool(r.recusa),
                       "metrica_principal": None if pd.isna(metrica) else float(metrica),
                       "resposta_no_contexto": resposta_no_contexto(r.contexto, p["respostas_aceitas"]) if r.dataset == "faquad" else None,
                       "resposta_gerada": r.resposta_gerada})
    SAIDA.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in linhas), encoding="utf-8")

    out = pd.DataFrame(linhas)
    print(pd.crosstab([out.dataset, out.veredito_juiz], out.recusa, margins=True).to_string())
    fq = out[out.dataset == "faquad"]
    completa, exato = fq.veredito_juiz == "completa", fq.resposta_no_contexto.astype(bool)
    from sklearn.metrics import cohen_kappa_score
    print("FaQuAD, juiz × resposta aceita no contexto (todas as respostas):\n", pd.crosstab(fq.veredito_juiz, exato))
    print(f"concordância {(completa == exato).mean():.3f}, kappa de Cohen {cohen_kappa_score(completa, exato):.3f}")
    print(f"Escritos {SAIDA.name} e {SAIDA_JUIZ.name}")


if __name__ == "__main__":
    main()
