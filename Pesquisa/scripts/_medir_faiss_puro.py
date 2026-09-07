"""
Passo 2/2 do comparativo ChromaDB vs FAISS: só importa faiss/numpy (nunca PyTorch), lendo os
vetores já calculados pelo passo 1 (_gerar_vetores_para_benchmark.py). Roda em processo separado
de propósito -- ver nota nesse outro arquivo.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
import psutil

faiss.omp_set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "resultados" / "metricas" / "_cache_vetores_benchmark"
RESULTADO_CHROMA_PATH = BASE_DIR / "resultados" / "metricas" / "_benchmark_chroma.json"
RESULTADO_FINAL_PATH = BASE_DIR / "resultados" / "metricas" / "comparativo_vectorstores.json"
PROCESSO = psutil.Process()


def _rss_mb() -> float:
    return PROCESSO.memory_info().rss / (1024 * 1024)


def medir_dominio(nome: str, k: int = 4) -> dict:
    vetores = np.load(CACHE_DIR / f"{nome}_vetores.npy")
    queries = np.load(CACHE_DIR / f"{nome}_queries.npy")
    textos = json.loads((CACHE_DIR / f"{nome}_textos.json").read_text(encoding="utf-8"))
    resultados_chroma = json.loads((CACHE_DIR / f"{nome}_resultados_chroma.json").read_text(encoding="utf-8"))

    dim = vetores.shape[1]
    rss_antes = _rss_mb()
    inicio = time.time()
    indice = faiss.IndexFlatL2(dim)
    indice.add(vetores)
    tempo_index_faiss = time.time() - inicio
    mem_index_faiss = _rss_mb() - rss_antes

    tempos_query = []
    resultados_faiss = []
    for vq in queries:
        inicio = time.time()
        _, indices = indice.search(vq.reshape(1, -1), k)
        tempos_query.append(time.time() - inicio)
        resultados_faiss.append([textos[i] for i in indices[0]])

    concordancia = sum(
        1 for a, b in zip(resultados_chroma, resultados_faiss) if set(a) == set(b)
    ) / len(resultados_chroma)

    return {
        "dominio": nome,
        "concordancia_top_k": round(concordancia, 3),
        "faiss": {
            "tempo_indexacao_s": round(tempo_index_faiss, 4),
            "memoria_indexacao_mb": round(mem_index_faiss, 2),
            "tempo_consulta_medio_ms": round(sum(tempos_query) / len(tempos_query) * 1000, 3),
        },
    }


def main():
    resultados_chroma = {r["dominio"]: r for r in json.loads(RESULTADO_CHROMA_PATH.read_text(encoding="utf-8"))}
    dominios = [p.stem.replace("_vetores", "") for p in CACHE_DIR.glob("*_vetores.npy")]

    final = []
    for nome in dominios:
        r_faiss = medir_dominio(nome)
        r = resultados_chroma[nome]
        r["concordancia_top_k"] = r_faiss["concordancia_top_k"]
        r["faiss"] = r_faiss["faiss"]
        final.append(r)

    RESULTADO_FINAL_PATH.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
