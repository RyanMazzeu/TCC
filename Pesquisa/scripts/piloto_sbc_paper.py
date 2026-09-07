"""
Piloto PT-BR: paper de Medeiros & Oliveira (IFES/SBC) sobre embeddings e LLMs
para RAG em português — paralelo brasileiro ao "Llama2 Paper" do paper-âncora.

Documento único como corpus, igual ao piloto do Grendene. Perguntas escritas
à mão, cada uma ancorada em trecho literal do paper.

Requer EMBEDDING_MODEL_NAME=BAAI/bge-m3.
"""

import json
import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import pipeline_base as pb

DADOS_DIR = pb.BASE_DIR / "dados"


def main():
    dataset = json.loads((DADOS_DIR / "processados" / "sbc_embeddings_llms.json").read_text(encoding="utf-8"))

    chunks = pb.carregar_e_chunkar(
        str(DADOS_DIR / "brutos" / "sbc_embeddings_llms.txt"), document_id="sbc_embeddings_llms"
    )
    print(f"{len(chunks)} chunks gerados.")

    colecao = pb.construir_indice(chunks, colecao_nome="piloto_sbc_paper")

    log_path = pb.METRICAS_DIR / "piloto_sbc_stuff.jsonl"
    for item in dataset["perguntas"]:
        registro = pb.rodar_experimento(
            query=item["pergunta"],
            colecao=colecao,
            document_id="sbc_embeddings_llms",
            resposta_esperada=item["resposta_esperada"],
            metodo="stuff",
            log_path=log_path,
        )
        print(f"\n=== {item['id']} ===")
        print("Resposta gerada:", registro["resposta_gerada"])
        print("Similaridade:", round(registro["similaridade_cosseno"], 4))


if __name__ == "__main__":
    main()
