"""
Exporta os fragmentos de texto dos datasets para colab/fragmentos_<dataset>.jsonl.gz, exatamente
como benchmarks.carregar os fragmenta aqui, pra os embeddings serem calculados no Google Colab (GPU)
por colab/colab_embeddings.py. Os vetores voltam com um manifesto (número de fragmentos + hash dos
textos), que benchmarks.obter_indice confere antes de usar.

Uso: python exportar_fragmentos.py [dataset ...]   (sem argumentos: docugami_10q e medqa)
"""

import gzip
import json
import os
import sys

# Sem isso o nome da coleção sai com o embedding padrão do pipeline (bge-small), e os vetores do
# Colab (que são BGE-M3) iam pra uma pasta que o construir_indices.py não procura.
os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import benchmarks

COLAB_DIR = benchmarks.pb.BASE_DIR / "colab"

if __name__ == "__main__":
    COLAB_DIR.mkdir(exist_ok=True)
    for nome in sys.argv[1:] or ["docugami_10q", "medqa"]:
        chunks, _ = benchmarks.carregar(nome)
        destino = COLAB_DIR / f"fragmentos_{nome}.jsonl.gz"
        with gzip.open(destino, "wt", encoding="utf-8") as f:
            f.write(json.dumps({"dataset": nome, "colecao": benchmarks._nome_colecao(nome),
                                "n_fragmentos": len(chunks)}, ensure_ascii=False) + "\n")
            for c in chunks:
                f.write(json.dumps(c["text"], ensure_ascii=False) + "\n")
        print(f"{destino.relative_to(benchmarks.pb.BASE_DIR)}: {len(chunks)} fragmentos ({destino.stat().st_size / 1e6:.0f} MB)")
