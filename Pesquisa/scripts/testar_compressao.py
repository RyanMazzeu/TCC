"""
Filtro de compressão contextual: testa 4 limiares (sem filtro, baixo, médio, alto) de
similaridade de cosseno sobre os chunks recuperados, usando o método Stuff -- o mais barato,
suficiente pra medir o trade-off tokens vs qualidade em si, sem precisar repetir para os outros
5 métodos de RAG.

Limiares escolhidos (0.0 / 0.3 / 0.5 / 0.7) não são calibrados por domínio -- o objetivo do
próprio teste é ver empiricamente quanto cada um poda por domínio (registrado em
"n_chunks_mantidos" no log), não presumir de antemão qual valor é "razoável" em cada corpus.
"""

import json
import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import pipeline_base as pb
from comparar_metodos import carregar_grendene, carregar_revalida, carregar_sbc

LOG_PATH = pb.METRICAS_DIR / "comparativo_compressao_contextual_v2.jsonl"
LIMIARES = {"sem_filtro": 0.0, "baixo": 0.3, "medio": 0.5, "alto": 0.7}


def combinacoes_ja_feitas(log_path) -> set[tuple[str, str, str]]:
    """Retomada: o log é só de append, então rodar de novo depois de uma interrupção duplicaria as
    linhas já gravadas. Devolve (dominio, limiar_rotulo, pergunta_id) de tudo que já está no log,
    pra essas combinações serem puladas em vez de pagas de novo."""
    if not log_path.exists():
        return set()
    feitas = set()
    for linha in log_path.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            r = json.loads(linha)
            feitas.add((r["dominio"], r["limiar_rotulo"], r["pergunta_id"]))
    return feitas


def main():
    dominios = [carregar_revalida(), carregar_grendene(), carregar_sbc()]
    ja_feitas = combinacoes_ja_feitas(LOG_PATH)
    if ja_feitas:
        print(f"Retomando: {len(ja_feitas)} combinações já em {LOG_PATH.name}, pulando essas.")

    for nome, chunks, perguntas in dominios:
        pendentes = [
            (rotulo_limiar, limiar, p)
            for rotulo_limiar, limiar in LIMIARES.items()
            for p in perguntas
            if (nome, rotulo_limiar, p["id"]) not in ja_feitas
        ]
        if not pendentes:
            print(f"[{nome}] completo, pulando (nem reconstrói o índice).")
            continue

        colecao = pb.construir_indice(chunks, colecao_nome=f"compressao_{nome}")
        for rotulo_limiar, limiar, p in pendentes:
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
