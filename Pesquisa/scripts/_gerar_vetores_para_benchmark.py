"""
Passo 1/2 do comparativo ChromaDB vs FAISS: gera embeddings (via BGE-M3/PyTorch) e mede a
indexação/consulta no ChromaDB. Salva os vetores em .npy pra um processo separado (só FAISS,
sem PyTorch) medir FAISS sem os dois disputarem o runtime OpenMP no mesmo processo -- rodar os
dois juntos travou o processo (ver nota em comparar_vectorstores.py, removido).
"""

import json
import os
import time

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import numpy as np
import psutil

import pipeline_base as pb

DADOS_DIR = pb.BASE_DIR / "dados"
CACHE_DIR = pb.BASE_DIR / "resultados" / "metricas" / "_cache_vetores_benchmark"
RESULTADO_CHROMA_PATH = pb.BASE_DIR / "resultados" / "metricas" / "_benchmark_chroma.json"
PROCESSO = psutil.Process(os.getpid())


def _rss_mb() -> float:
    return PROCESSO.memory_info().rss / (1024 * 1024)


def carregar_dominios():
    revalida = pb.carregar_e_chunkar(
        str(DADOS_DIR / "brutos" / "codigo_etica_medica_cfm.txt"), document_id="cfm_codigo_etica"
    ) + pb.carregar_e_chunkar(
        str(DADOS_DIR / "brutos" / "sbp_consulta_adolescente.txt"), document_id="sbp_consulta_adolescente"
    ) + pb.carregar_e_chunkar(
        str(DADOS_DIR / "brutos" / "revalida_ref_tev_scielo.txt"), document_id="tev_scielo"
    ) + pb.carregar_e_chunkar(
        str(DADOS_DIR / "brutos" / "revalida_ref_parto_febrasgo.txt"), document_id="parto_febrasgo"
    ) + pb.carregar_e_chunkar(
        str(DADOS_DIR / "brutos" / "revalida_ref_meningococica_ms.txt"), document_id="meningococica_ms"
    )
    grendene = pb.carregar_e_chunkar(str(DADOS_DIR / "brutos" / "grendene_itr_3t2025.txt"), document_id="grendene_itr_3t2025")
    sbc = pb.carregar_e_chunkar(str(DADOS_DIR / "brutos" / "sbc_embeddings_llms.txt"), document_id="sbc_embeddings_llms")

    dataset_revalida = json.loads((DADOS_DIR / "processados" / "revalida_2024_2_completo.json").read_text(encoding="utf-8"))
    perguntas_revalida = [f"{p['vinheta']}\n\n{p['pergunta']}" for p in dataset_revalida["perguntas"]]
    perguntas_grendene = [p["pergunta"] for p in json.loads((DADOS_DIR / "processados" / "grendene_itr_3t2025.json").read_text(encoding="utf-8"))["perguntas"]]
    perguntas_sbc = [p["pergunta"] for p in json.loads((DADOS_DIR / "processados" / "sbc_embeddings_llms.json").read_text(encoding="utf-8"))["perguntas"]]

    return [
        ("revalida", revalida, perguntas_revalida),
        ("grendene", grendene, perguntas_grendene),
        ("sbc", sbc, perguntas_sbc),
    ]


def processar_dominio(nome: str, chunks: list[dict], perguntas: list[str], k: int = 4) -> dict:
    modelo = pb.get_embedding_model()
    textos = [c["text"] for c in chunks]
    vetores = pb._encode_documentos(modelo, textos).astype("float32")

    # --- ChromaDB --- (vetores já calculados acima, passados prontos: mede só a indexação em si)
    rss_antes = _rss_mb()
    inicio = time.time()
    colecao = pb.construir_indice(chunks, colecao_nome=f"bench_chroma_{nome}", vetores=vetores)
    tempo_index_chroma = time.time() - inicio
    mem_index_chroma = _rss_mb() - rss_antes

    tempos_query_chroma = []
    resultados_chroma = []
    vetores_query = []
    for pergunta in perguntas:
        vq = pb._encode_query(modelo, pergunta).astype("float32")
        vetores_query.append(vq[0])
        inicio = time.time()
        docs = pb.recuperar(colecao, pergunta, k=k, vetor_query=vq)
        tempos_query_chroma.append(time.time() - inicio)
        resultados_chroma.append(docs)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / f"{nome}_vetores.npy", vetores)
    np.save(CACHE_DIR / f"{nome}_queries.npy", np.array(vetores_query, dtype="float32"))
    (CACHE_DIR / f"{nome}_textos.json").write_text(json.dumps(textos, ensure_ascii=False), encoding="utf-8")
    (CACHE_DIR / f"{nome}_resultados_chroma.json").write_text(json.dumps(resultados_chroma, ensure_ascii=False), encoding="utf-8")

    return {
        "dominio": nome,
        "n_chunks": len(chunks),
        "n_perguntas": len(perguntas),
        "chromadb": {
            "tempo_indexacao_s": round(tempo_index_chroma, 4),
            "memoria_indexacao_mb": round(mem_index_chroma, 2),
            "tempo_consulta_medio_ms": round(sum(tempos_query_chroma) / len(tempos_query_chroma) * 1000, 3),
        },
    }


def main():
    resultados = [processar_dominio(nome, chunks, perguntas) for nome, chunks, perguntas in carregar_dominios()]
    RESULTADO_CHROMA_PATH.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resultados, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
