"""
Piloto PT-BR: Revalida INEP 2024/2, Questão 3 (sigilo médico com adolescente).

Corpus de recuperação = as duas fontes citadas no próprio gabarito oficial
(Código de Ética Médica do CFM + guia da SBP sobre consulta ao adolescente),
não uma escolha nossa - reduz o risco de montar um corpus que não bate com
o que a banca do Revalida considerou correto.

Requer EMBEDDING_MODEL_NAME=BAAI/bge-m3 no ambiente (bge-small-en-v1.5, o
default do pipeline_base, só entende inglês).
"""

import json
import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import pipeline_base as pb

DADOS_DIR = pb.BASE_DIR / "dados"


def main():
    dataset = json.loads((DADOS_DIR / "processados" / "revalida_2024_2_questao3.json").read_text(encoding="utf-8"))

    chunks = pb.carregar_e_chunkar(
        str(DADOS_DIR / "brutos" / "codigo_etica_medica_cfm.txt"), document_id="cfm_codigo_etica"
    ) + pb.carregar_e_chunkar(
        str(DADOS_DIR / "brutos" / "sbp_consulta_adolescente.txt"), document_id="sbp_consulta_adolescente"
    )
    print(f"{len(chunks)} chunks gerados a partir de {len(dataset['corpus_documentos'])} documentos-fonte.")

    colecao = pb.construir_indice(chunks, colecao_nome="piloto_revalida_q3")

    log_path = pb.METRICAS_DIR / "piloto_revalida_stuff.jsonl"
    resultados = []
    for item in dataset["perguntas"]:
        query = f"{dataset['vinheta']}\n\n{item['pergunta']}"
        registro = pb.rodar_experimento(
            query=query,
            colecao=colecao,
            document_id="revalida_2024_2_q3",
            resposta_esperada=item["resposta_esperada"],
            metodo="stuff",
            log_path=log_path,
        )
        registro["pergunta_id"] = item["id"]
        resultados.append(registro)
        print(f"\n=== {item['id']} ===")
        print("Resposta gerada:", registro["resposta_gerada"])
        print("Similaridade:", round(registro["similaridade_cosseno"], 4))

    return resultados


if __name__ == "__main__":
    main()
