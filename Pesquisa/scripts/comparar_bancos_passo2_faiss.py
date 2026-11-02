"""
Passo 2/2 do comparativo ChromaDB × FAISS com os vetores dos benchmarks: só importa faiss/numpy (nunca
PyTorch), lendo o que o passo 1 (comparar_bancos_passo1_chroma.py) guardou. Mede o FAISS exato
(IndexFlatL2, em memória) e compara com o ChromaDB (HNSW, aproximado):

- tempo de indexação, memória e tempo médio de consulta;
- concordância do top-4: fração das consultas em que os dois bancos devolvem os mesmos 4 fragmentos
  (é o que o LLM recebe; se forem os mesmos, a resposta do RAG não muda) e sobreposição média;
- Hit@4 e Recall@4 dos dois bancos contra o documento-ouro (Docugami e FaQuAD): se a busca
  aproximada do Chroma perde o trecho certo que a exata acharia.

Saída: resultados/metricas/comparativo_vectorstores_benchmarks.json, e o top-4 do FAISS de cada consulta
(vectorstores_benchmarks/_comparativo_bancos/<dataset>_top4_faiss.json) pro passo 3.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
import psutil

faiss.omp_set_num_threads(1)

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "vectorstores_benchmarks" / "_comparativo_bancos"
METRICAS_DIR = BASE_DIR / "resultados" / "metricas"
RESULTADO_CHROMA_PATH = METRICAS_DIR / "_comparativo_bancos_chroma.json"
RESULTADO_FINAL_PATH = METRICAS_DIR / "comparativo_vectorstores_benchmarks.json"
K = 4
PROCESSO = psutil.Process()


def _rss_mb() -> float:
    return PROCESSO.memory_info().rss / (1024 * 1024)


def _vetores(cache: Path) -> np.ndarray:
    blocos = sorted(cache.glob("bloco_*.npy"))
    return np.concatenate([np.load(b).astype(np.float32) for b in blocos])


def _documento(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0]


def _hit_recall(ids_top: list[list[str]], ouros: list) -> tuple[float, float] | tuple[None, None]:
    pares = [(ids, set(o)) for ids, o in zip(ids_top, ouros) if o]
    if not pares:
        return None, None
    hits = [float(bool(o & {_documento(i) for i in ids[:K]})) for ids, o in pares]
    recalls = [len(o & {_documento(i) for i in ids[:K]}) / len(o) for ids, o in pares]
    return round(float(np.mean(hits)), 4), round(float(np.mean(recalls)), 4)


def medir(r: dict) -> dict:
    nome = r["dataset"]
    dados = json.loads((CACHE_DIR / f"{nome}_dados.json").read_text(encoding="utf-8"))
    consultas = np.load(CACHE_DIR / f"{nome}_consultas.npy")
    vetores = _vetores(Path(dados["cache_vetores"]))
    ids_fragmentos = dados["ids_fragmentos"]
    assert len(vetores) == len(ids_fragmentos) == r["n_vetores"]

    rss_antes, inicio = _rss_mb(), time.time()
    indice = faiss.IndexFlatL2(vetores.shape[1])
    indice.add(vetores)
    tempo_indexacao = time.time() - inicio
    memoria = _rss_mb() - rss_antes

    tempos, ids_faiss, distancias_faiss = [], [], []
    for vq in consultas:
        inicio = time.time()
        distancias, indices = indice.search(vq.reshape(1, -1), K)
        tempos.append(time.time() - inicio)
        ids_faiss.append([ids_fragmentos[i] for i in indices[0]])
        distancias_faiss.append([float(d) for d in distancias[0]])
    # O passo 3 (comparar_bancos_passo3_stuff_faiss.py) responde de novo, com estes fragmentos, as
    # perguntas em que o top-4 do FAISS difere do que o LLM recebeu nos experimentos.
    (CACHE_DIR / f"{nome}_top4_faiss.json").write_text(
        json.dumps({"ids": ids_faiss, "distancias": distancias_faiss}, ensure_ascii=False), encoding="utf-8")

    ids_chroma = [ids[:K] for ids in dados["ids_chroma"]]
    iguais = [set(a) == set(b) for a, b in zip(ids_chroma, ids_faiss)]
    sobreposicao = [len(set(a) & set(b)) / K for a, b in zip(ids_chroma, ids_faiss)]
    hit_c, rec_c = _hit_recall(ids_chroma, dados["documentos_ouro"])
    hit_f, rec_f = _hit_recall(ids_faiss, dados["documentos_ouro"])

    r["faiss"] = {
        "indice": "IndexFlatL2 (busca exata)",
        "tempo_indexacao_s": round(tempo_indexacao, 3),
        "memoria_indexacao_mb": round(memoria, 1),
        "tempo_consulta_medio_ms": round(1000 * float(np.mean(tempos)), 3),
    }
    r["concordancia_top4"] = round(float(np.mean(iguais)), 4)
    r["sobreposicao_media_top4"] = round(float(np.mean(sobreposicao)), 4)
    r["hit@4"] = {"chromadb": hit_c, "faiss": hit_f}
    r["recall@4"] = {"chromadb": rec_c, "faiss": rec_f}
    print(f"[{nome}] concordância top-4 {r['concordancia_top4']:.1%}, sobreposição {r['sobreposicao_media_top4']:.1%}, "
          f"Hit@4 Chroma {hit_c} × FAISS {hit_f}", flush=True)
    return r


if __name__ == "__main__":
    resultados = [medir(r) for r in json.loads(RESULTADO_CHROMA_PATH.read_text(encoding="utf-8"))]
    RESULTADO_FINAL_PATH.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Escrito {RESULTADO_FINAL_PATH.relative_to(BASE_DIR)}")
