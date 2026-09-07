"""
Filtro de compressão contextual: testa 4 limiares (sem filtro, baixo, médio, alto) de
similaridade de cosseno sobre os chunks recuperados, usando o método Stuff -- o mais barato,
suficiente pra medir o trade-off tokens vs qualidade em si, sem precisar repetir para os outros
5 métodos de RAG.

Limiares escolhidos (0.0 / 0.3 / 0.5 / 0.7) não são calibrados por domínio -- o objetivo do
próprio teste é ver empiricamente quanto cada um poda por domínio (registrado em
"n_chunks_mantidos" no log), não presumir de antemão qual valor é "razoável" em cada corpus.
"""

import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import pipeline_base as pb
from comparar_metodos import carregar_grendene, carregar_revalida, carregar_sbc

LOG_PATH = pb.METRICAS_DIR / "comparativo_compressao_contextual.jsonl"
LIMIARES = {"sem_filtro": 0.0, "baixo": 0.3, "medio": 0.5, "alto": 0.7}


def main():
    dominios = [carregar_revalida(), carregar_grendene(), carregar_sbc()]

    for nome, chunks, perguntas in dominios:
        colecao = pb.construir_indice(chunks, colecao_nome=f"compressao_{nome}")
        for rotulo_limiar, limiar in LIMIARES.items():
            for p in perguntas:
                registro = pb.rodar_experimento(
                    query=p["query"],
                    colecao=colecao,
                    document_id=nome,
                    resposta_esperada=p["esperada"],
                    metodo="stuff",
                    log_path=LOG_PATH,
                    limiar_compressao=limiar,
                    extra_campos={"dominio": nome, "pergunta_id": p["id"], "limiar_rotulo": rotulo_limiar},
                )
                n_chunks = registro["compressao_contextual"]["n_chunks_mantidos"]
                print(
                    f"[{nome}/{rotulo_limiar}] {p['id']}: sim={registro['similaridade_cosseno']:.4f} "
                    f"tokens={registro['tokens_totais']} chunks_mantidos={n_chunks}"
                )


if __name__ == "__main__":
    main()
