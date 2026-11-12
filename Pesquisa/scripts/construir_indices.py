"""
Cria os índices vetoriais dos benchmarks (BGE-M3, ChromaDB em vectorstores_benchmarks/) e roda as
métricas de recuperação. Não chama LLM nenhum: custo zero.

Os embeddings do Docugami (~4,5 mil fragmentos) e do MedQA (~121 mil) levam ~1,7 h e ~27 h em CPU
(i5-12450H, medido), então são calculados no Google Colab com GPU, em minutos:
  1. python exportar_fragmentos.py         -> colab/fragmentos_*.jsonl.gz
  2. no Colab, colab/colab_embeddings.py   -> embeddings_colab.zip (instruções no próprio arquivo)
  3. descompactar o zip em vectorstores_benchmarks/_cache_embeddings/
  4. python construir_indices.py           -> lê os vetores, confere o manifesto e cria os índices
O FaQuAD (318 fragmentos, ~7 min) é calculado aqui mesmo. Com --local, calcula todos aqui (lento;
retoma de onde parou se for interrompido).
"""

import argparse
import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import avaliar_recuperacao
import benchmarks

CALCULADOS_NO_COLAB = {"docugami_10q", "medqa"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Calcula aqui também os embeddings do Docugami e do MedQA")
    args = parser.parse_args()

    prontos = []
    for nome in ["faquad", "docugami_10q", "medqa"]:
        cache = benchmarks.VECTORSTORE_DIR / "_cache_embeddings" / benchmarks._nome_colecao(nome)
        if nome in CALCULADOS_NO_COLAB and not args.local and not (cache / "manifesto.json").exists():
            print(f"[{nome}] sem os vetores do Colab em {cache} -- veja os passos no topo deste arquivo "
                  "(ou rode com --local).", flush=True)
            continue
        chunks, _ = benchmarks.carregar(nome)
        benchmarks.obter_indice(nome, chunks)
        prontos.append(nome)
        print(f"[{nome}] índice pronto: {len(chunks)} fragmentos", flush=True)
    if all(nome in prontos for nome in benchmarks.DATASETS_COM_TRECHO_CERTO):
        avaliar_recuperacao.main()
