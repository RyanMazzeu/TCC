"""
Passo 3/3 do comparativo ChromaDB × FAISS: o efeito da troca de banco na resposta do RAG.

Para cada dataset, compara o top-4 que o Stuff do gpt-4o-mini recebeu nos experimentos (índice HNSW
do ChromaDB; campo ids_recuperados do log) com o top-4 exato do FAISS (salvo pelo passo 2). Onde os
dois são iguais (mesmos fragmentos, na mesma ordem), o LLM recebe a mesma entrada, então a resposta
com o FAISS é a dos experimentos. Só as perguntas com top-4 diferente são respondidas de novo, com os
fragmentos do FAISS, e entram na comparação da métrica principal do dataset (acurácia no MedQA).

Limitação: o gpt-4o-mini roda com temperatura 0, mas a API não garante saída idêntica entre
chamadas; parte da diferença nas perguntas refeitas pode ser variação do próprio modelo.

Custo: centavos (Stuff do gpt-4o-mini só nas perguntas que mudam). Retoma de onde parou.
Saídas: stuff_faiss_<dataset>_gpt_mini.jsonl e comparativo_stuff_faiss.json (resultados/metricas).
"""

import json
import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import numpy as np
from scipy import stats

import benchmarks
import metricas_resposta
import pipeline_base as pb
from analisar_benchmarks import METRICA_PRINCIPAL
from rodar_benchmarks import _ja_feitos, _rodar_com_retentativas, log_do_experimento

LLM = "gpt_mini"
CACHE_DIR = benchmarks.VECTORSTORE_DIR / "_comparativo_bancos"
SAIDA = pb.METRICAS_DIR / "comparativo_stuff_faiss.json"


class ColecaoFixa:
    """Imita a busca do ChromaDB devolvendo um top-k já calculado (o do FAISS), pra o Stuff rodar
    pelo mesmo caminho dos experimentos (pb.rodar_experimento)."""

    def __init__(self, ids: list[str], textos: list[str], distancias: list[float]):
        self.ids, self.textos, self.distancias = ids, textos, distancias

    def query(self, query_embeddings, n_results: int) -> dict:
        return {"ids": [self.ids[:n_results]], "documents": [self.textos[:n_results]],
                "distances": [self.distancias[:n_results]]}


def _nota(registro: dict, pergunta: dict, metrica: str) -> float:
    if metrica == "similaridade_cosseno":
        return float(registro["similaridade_cosseno"])
    return float(metricas_resposta.metricas(registro, pergunta)[metrica])


def processar(nome: str) -> dict | None:
    caminho_faiss = CACHE_DIR / f"{nome}_top4_faiss.json"
    if not caminho_faiss.exists():
        print(f"[{nome}] sem o top-4 do FAISS: rode comparar_bancos_passo2_faiss.py antes.")
        return None
    top4_faiss = json.loads(caminho_faiss.read_text(encoding="utf-8"))
    chunks, todas = benchmarks.carregar(nome)
    texto_por_id = {f"{c['document_id']}#{c['chunk_index']}": c["text"] for c in chunks}
    # A mesma amostra (e ordem) das consultas do passo 1.
    perguntas = benchmarks.amostra(todas, benchmarks.PERGUNTAS_POR_CAMADA["principal"])
    assert len(perguntas) == len(top4_faiss["ids"])

    stuff_chroma = {r["pergunta_id"]: r for r in map(json.loads, log_do_experimento(nome, LLM).read_text(encoding="utf-8").splitlines())
                    if r["metodo_rag"] == "stuff" and r.get("repeticao", 1) == 1}
    # Fora do padrão benchmarks_*.jsonl de propósito: a análise principal lê todos os logs com esse nome.
    log_faiss = pb.METRICAS_DIR / f"stuff_faiss_{nome}_{LLM}.jsonl"
    feitos = {pid for _, pid, _ in _ja_feitos(log_faiss)}

    pb.CONJUNTO_PROMPTS = benchmarks.CONJUNTO_PROMPTS[nome]
    pb.configurar_llm_por_rotulo(LLM)
    llm = pb.llm_por_rotulo(LLM)
    diferentes = []
    for p, ids, distancias in zip(perguntas, top4_faiss["ids"], top4_faiss["distancias"]):
        if stuff_chroma[p["id"]]["ids_recuperados"] == ids:
            continue
        diferentes.append(p)
        if p["id"] in feitos:
            continue
        registro = _rodar_com_retentativas(
            query=p["pergunta"], colecao=ColecaoFixa(ids, [texto_por_id[i] for i in ids], distancias),
            document_id=nome, resposta_esperada=p["resposta_esperada"], metodo="stuff", log_path=log_faiss,
            extra_campos={"banco_vetorial": "faiss_indexflatl2", "dominio": nome, "dataset": nome, "llm": LLM,
                          "camada": llm["camada"], "pergunta_id": p["id"], "repeticao": 1,
                          "conjunto_prompts": pb.CONJUNTO_PROMPTS},
        )
        print(f"[{nome}/faiss] {p['id']}: sim={registro['similaridade_cosseno']:.4f}", flush=True)

    metrica, rotulo = METRICA_PRINCIPAL[nome]
    stuff_faiss = {r["pergunta_id"]: r for r in map(json.loads, log_faiss.read_text(encoding="utf-8").splitlines())} if diferentes else {}
    chroma = [_nota(stuff_chroma[p["id"]], p, metrica) for p in perguntas]
    faiss_ = [_nota(stuff_faiss[p["id"]], p, metrica) if p["id"] in stuff_faiss else c for p, c in zip(perguntas, chroma)]
    ch_dif = [_nota(stuff_chroma[p["id"]], p, metrica) for p in diferentes]
    fa_dif = [_nota(stuff_faiss[p["id"]], p, metrica) for p in diferentes]

    resultado = {
        "dataset": nome, "llm": LLM, "metrica": metrica, "metrica_rotulo": rotulo,
        "n_perguntas": len(perguntas),
        "top4_identico_ao_experimento": round(1 - len(diferentes) / len(perguntas), 4),
        "perguntas_refeitas": len(diferentes),
        "media_chroma": round(float(np.nanmean(chroma)), 4),
        "media_faiss": round(float(np.nanmean(faiss_)), 4),
        "media_chroma_nas_refeitas": round(float(np.nanmean(ch_dif)), 4) if diferentes and not np.isnan(ch_dif).all() else None,
        "media_faiss_nas_refeitas": round(float(np.nanmean(fa_dif)), 4) if diferentes and not np.isnan(fa_dif).all() else None,
    }
    if metrica == "acerto" and diferentes:
        # McNemar exato: só contam as perguntas em que um banco acerta e o outro erra.
        so_chroma = sum(1 for c, f in zip(ch_dif, fa_dif) if c > f)
        so_faiss = sum(1 for c, f in zip(ch_dif, fa_dif) if f > c)
        resultado.update(acerta_so_chroma=so_chroma, acerta_so_faiss=so_faiss,
                         p_mcnemar=round(stats.binomtest(so_chroma, so_chroma + so_faiss).pvalue, 4) if so_chroma + so_faiss else None)
    elif len(diferentes) >= 2:
        resultado["p_wilcoxon"] = round(float(stats.wilcoxon(ch_dif, fa_dif).pvalue), 4) if any(c != f for c, f in zip(ch_dif, fa_dif)) else None
    print(f"[{nome}] top-4 igual ao dos experimentos em {resultado['top4_identico_ao_experimento']:.0%}; "
          f"{rotulo}: Chroma {resultado['media_chroma']:.3f} × FAISS {resultado['media_faiss']:.3f}", flush=True)
    return resultado


if __name__ == "__main__":
    resultados = [r for r in (processar(nome) for nome in benchmarks.DATASETS) if r]
    SAIDA.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Escrito {SAIDA.name}")
