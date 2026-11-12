"""
Separa as recusas dos métodos de RAG em justificadas e injustificadas, conforme o contexto que o LLM
recebeu tinha ou não a informação pedida (repetição 1, todos os LLMs, Docugami e FaQuAD):

- justificada (falha da recuperação): o contexto não tinha a informação, ou tinha só parte dela;
- injustificada (falha do LLM ou do método): o contexto tinha toda a informação que responde à
  pergunta, e o modelo ainda disse que não sabia.

O contexto é o que o método entregou ao LLM: os 4 fragmentos da pergunta original no Stuff, Refine,
Map-Reduce e Map-Rerank, e a união dos fragmentos das perguntas alternativas no Query Step-Down e no
Reciprocal. Os logos desses dois guardam só as perguntas alternativas, e a busca é refeita com elas no
mesmo índice (a recuperação é determinística: o script confere isso refazendo a do Stuff).

Como saber se a informação estava no contexto:
- FaQuAD: as respostas aceitas são trechos copiados do parágrafo, então basta procurar alguma delas,
  normalizada, no texto do contexto (critério exato, sem juiz).
- Docugami: a resposta de referência é texto livre e costuma juntar valores de vários relatórios. Um
  juiz (gpt-4o-mini, temperatura 0, o mesmo do RAGAS) recebe a pergunta, a resposta de referência e o
  contexto e diz se o contexto tem toda a informação, só parte ou nenhuma. O juiz também roda no
  FaQuAD, e a concordância dele com o critério exato fica no resumo, como validação.

O MedQA fica de fora: o prompt manda sempre escolher uma alternativa, e houve só 3 recusas.

Saídas (retoma de onde parou; só chama o juiz para contextos ainda não julgados):
  resultados/metricas/recusas_juiz.jsonl         veredito do juiz por (pergunta, contexto)
  resultados/metricas/recusas_classificadas.jsonl uma linha por recusa
Uso: python classificar_recusas.py
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
os.environ.setdefault("OMP_NUM_THREADS", "3")

from openai import OpenAI

import analisar_benchmarks as ab
import benchmarks
import metricas_resposta as mr
import pipeline_base as pb

DATASETS = ["docugami_10q", "faquad"]
JUIZ = "gpt-4o-mini"
SAIDA_JUIZ = pb.METRICAS_DIR / "recusas_juiz.jsonl"
SAIDA = pb.METRICAS_DIR / "recusas_classificadas.jsonl"
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


def carregar_recusas():
    df = ab.carregar_execucoes()
    return df[(df.repeticao == 1) & df.dataset.isin(DATASETS) & df.metodo_rag.isin(ab.METODOS_RAG) & df.recusa].copy()


def reconstruir_contextos(rec, colecoes):
    """Fragmentos entregues ao LLM em cada recusa. No Step-Down e no Reciprocal, refaz a busca com as
    perguntas alternativas gravadas no log; antes, confere que a busca reproduz os ids gravados do Stuff."""
    modelo = pb.get_embedding_model()
    conferidas = rec[rec.metodo_rag == "stuff"].drop_duplicates(["dataset", "pergunta_id"]).head(20)
    for _, r in conferidas.iterrows():
        ids = pb.consultar_indice(colecoes[r.dataset], r.pergunta, k=4, vetor_query=pb._encode_query(modelo, r.pergunta))["ids"]
        if ids != list(r.ids_recuperados):
            raise RuntimeError(f"A busca não reproduz o log ({r.dataset} {r.pergunta_id}): {ids} × {r.ids_recuperados}")
    cache = {}

    def buscar(nome, pergunta):
        if (nome, pergunta) not in cache:
            vetor = pb._encode_query(modelo, pergunta)
            cache[nome, pergunta] = pb.consultar_indice(colecoes[nome], pergunta, k=4, vetor_query=vetor)["documentos"]
        return cache[nome, pergunta]

    contextos = []
    for _, r in rec.iterrows():
        if r.metodo_rag in METODOS_MULTIPLAS_BUSCAS:
            textos = [t for alt in r.contextos_recuperados for t in buscar(r.dataset, alt)]
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
    rec = carregar_recusas()
    perguntas = {n: {p["id"]: p for p in benchmarks.carregar_perguntas(n)} for n in DATASETS}
    colecoes = {}
    for nome in DATASETS:
        chunks, _ = benchmarks.carregar(nome)
        colecoes[nome] = benchmarks.obter_indice(nome, chunks)
    rec["contexto"] = reconstruir_contextos(rec, colecoes)
    rec["chave"] = [f"{d}|{p}|{_hash(c)}" for d, p, c in zip(rec.dataset, rec.pergunta_id, rec.contexto)]

    julgados = {}
    if SAIDA_JUIZ.exists():
        julgados = {j["chave"]: j for j in map(json.loads, SAIDA_JUIZ.read_text(encoding="utf-8").splitlines()) if j}
    pendentes = rec.drop_duplicates("chave")
    pendentes = pendentes[~pendentes.chave.isin(julgados)]
    print(f"{len(rec)} recusas, {rec.chave.nunique()} contextos distintos, {len(pendentes)} a julgar")
    if len(pendentes):
        cliente = OpenAI()

        def _um(r):
            j = julgar(cliente, r.pergunta, perguntas[r.dataset][r.pergunta_id]["resposta_esperada"], r.contexto)
            return {"chave": r.chave, "dataset": r.dataset, "pergunta_id": r.pergunta_id, "juiz": JUIZ, **j}

        with ThreadPoolExecutor(max_workers=16) as pool, SAIDA_JUIZ.open("a", encoding="utf-8") as arq:
            for j in pool.map(_um, [r for _, r in pendentes.iterrows()]):
                julgados[j["chave"]] = j
                arq.write(json.dumps(j, ensure_ascii=False) + "\n")
                arq.flush()

    linhas = []
    for _, r in rec.iterrows():
        p = perguntas[r.dataset][r.pergunta_id]
        veredito = julgados[r.chave]["veredito"]
        exato = resposta_no_contexto(r.contexto, p["respostas_aceitas"]) if r.dataset == "faquad" else None
        informacao_no_contexto = exato if r.dataset == "faquad" else veredito == "completa"
        linhas.append({"dataset": r.dataset, "llm": r.llm, "camada": r.camada, "metodo_rag": r.metodo_rag,
                       "pergunta_id": r.pergunta_id, "n_fragmentos": len(r.contexto), "veredito_juiz": veredito,
                       "resposta_no_contexto": exato, "informacao_no_contexto": bool(informacao_no_contexto),
                       "classificacao": "injustificada" if informacao_no_contexto else "justificada",
                       "resposta_gerada": r.resposta_gerada})
    SAIDA.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in linhas), encoding="utf-8")

    import pandas as pd
    out = pd.DataFrame(linhas)
    print(out.groupby(["dataset", "classificacao"]).size().to_string())
    print(out.groupby(["dataset", "veredito_juiz"]).size().to_string())
    fq = out[out.dataset == "faquad"]
    print("FaQuAD, juiz 'completa' × resposta no contexto:\n", pd.crosstab(fq.veredito_juiz, fq.resposta_no_contexto))
    print(f"Escritos {SAIDA.name} e {SAIDA_JUIZ.name}")


if __name__ == "__main__":
    main()
