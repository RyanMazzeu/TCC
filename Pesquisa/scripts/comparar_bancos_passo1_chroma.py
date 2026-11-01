"""
Passo 1/2 do comparativo ChromaDB × FAISS com os vetores dos benchmarks (FaQuAD, Docugami e MedQA:
de centenas a ~121 mil vetores). Mede, no ChromaDB (índice HNSW padrão, o mesmo dos experimentos de
RAG): tempo de indexação, memória e tempo médio de consulta, e guarda os ids do top-10 de cada
consulta. O passo 2 (comparar_bancos_passo2_faiss.py) mede o FAISS exato em outro processo -- PyTorch
e FAISS no mesmo processo disputam o runtime OpenMP e travaram o comparativo antigo.

Os vetores são os mesmos dos experimentos (cache de benchmarks.obter_indice); nada é recalculado. As
consultas são as 100 perguntas da camada principal de cada dataset. A coleção é criada numa pasta
temporária e apagada no fim, pra não mexer nos índices dos experimentos.

Custo zero (sem LLM). Rodar com a máquina sem outros processos pesados, pra os tempos valerem.
"""

import json
import os
import shutil
import time

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import numpy as np
import psutil

import benchmarks
import pipeline_base as pb

K = 10
CACHE_DIR = benchmarks.VECTORSTORE_DIR / "_comparativo_bancos"
RESULTADO_CHROMA_PATH = pb.METRICAS_DIR / "_comparativo_bancos_chroma.json"
PROCESSO = psutil.Process(os.getpid())


def _rss_mb() -> float:
    return PROCESSO.memory_info().rss / (1024 * 1024)


def processar(nome: str) -> dict:
    chunks, todas = benchmarks.carregar(nome)
    cache_vetores = benchmarks.VECTORSTORE_DIR / "_cache_embeddings" / benchmarks._nome_colecao(nome)
    vetores = benchmarks._vetores_em_blocos(chunks, cache_vetores)
    perguntas = benchmarks.amostra(todas, benchmarks.PERGUNTAS_POR_CAMADA["principal"])

    modelo = pb.get_embedding_model()
    consultas = np.array([pb._encode_query(modelo, p["pergunta"])[0] for p in perguntas], dtype="float32")

    pasta = CACHE_DIR / f"chroma_{nome}"
    shutil.rmtree(pasta, ignore_errors=True)
    rss_antes, inicio = _rss_mb(), time.time()
    colecao = pb.construir_indice(chunks, colecao_nome=f"comparativo_{nome}", vetores=vetores, diretorio=pasta)
    tempo_indexacao = time.time() - inicio
    memoria = _rss_mb() - rss_antes

    tempos, ids = [], []
    for vq in consultas:
        inicio = time.time()
        resultado = colecao.query(query_embeddings=[vq.tolist()], n_results=K)
        tempos.append(time.time() - inicio)
        ids.append(resultado["ids"][0])

    # O passo 2 lê tudo daqui, só com numpy (sem importar este módulo, que carrega o PyTorch).
    np.save(CACHE_DIR / f"{nome}_consultas.npy", consultas)
    (CACHE_DIR / f"{nome}_dados.json").write_text(json.dumps({
        "cache_vetores": str(cache_vetores),
        "ids_fragmentos": [f"{c['document_id']}#{c['chunk_index']}" for c in chunks],
        "ids_chroma": ids,
        "documentos_ouro": [p.get("documentos_ouro") for p in perguntas],
    }, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(pasta, ignore_errors=True)

    print(f"[{nome}] {len(chunks)} vetores: indexação {tempo_indexacao:.1f} s, consulta {1000 * np.mean(tempos):.2f} ms", flush=True)
    return {
        "dataset": nome,
        "n_vetores": len(chunks),
        "n_consultas": len(perguntas),
        "chromadb": {
            "indice": "HNSW (padrão do ChromaDB)",
            "tempo_indexacao_s": round(tempo_indexacao, 3),
            "memoria_indexacao_mb": round(memoria, 1),
            "tempo_consulta_medio_ms": round(1000 * float(np.mean(tempos)), 3),
        },
    }


if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resultados = [processar(nome) for nome in ["faquad", "docugami_10q", "medqa"]]
    RESULTADO_CHROMA_PATH.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
