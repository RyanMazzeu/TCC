"""
Piloto PT-BR: Grendene S.A., ITR 30/09/2025 — paralelo brasileiro ao SEC 10-Q.

Documento único como corpus (igual ao piloto do Llama2 Paper), sem gabarito
pronto — as 3 perguntas em dados/processados/grendene_itr_3t2025.json foram
escritas à mão, cada uma ancorada em um trecho literal do documento.

Requer EMBEDDING_MODEL_NAME=BAAI/bge-m3 (mesmo motivo do piloto do Revalida).
"""

import json
import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import pipeline_base as pb

DADOS_DIR = pb.BASE_DIR / "dados"


def main():
    dataset = json.loads((DADOS_DIR / "processados" / "grendene_itr_3t2025.json").read_text(encoding="utf-8"))

    chunks = pb.carregar_e_chunkar(
        str(DADOS_DIR / "brutos" / "grendene_itr_3t2025.txt"), document_id="grendene_itr_3t2025"
    )
    print(f"{len(chunks)} chunks gerados.")

    colecao = pb.construir_indice(chunks, colecao_nome="piloto_grendene_itr")

    log_path = pb.METRICAS_DIR / "piloto_grendene_stuff.jsonl"
    for item in dataset["perguntas"]:
        registro = pb.rodar_experimento(
            query=item["pergunta"],
            colecao=colecao,
            document_id="grendene_itr_3t2025",
            resposta_esperada=item["resposta_esperada"],
            metodo="stuff",
            log_path=log_path,
        )
        print(f"\n=== {item['id']} ===")
        print("Resposta gerada:", registro["resposta_gerada"])
        print("Similaridade:", round(registro["similaridade_cosseno"], 4))


if __name__ == "__main__":
    main()
