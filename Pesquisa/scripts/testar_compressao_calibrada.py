"""
Compressão contextual com limiares calibrados por domínio.

A rodada com limiares fixos (0,3 / 0,5 / 0,7 -- testar_compressao.py) mostrou que 0,3 e 0,5 não
cortam nada: com BGE-M3, os chunks do top-4 já vêm com similaridade acima disso. Aqui o limiar de
cada domínio sai da distribuição observada: os quantis 25 / 50 / 75 da similaridade dos chunks do
top-4 de todas as perguntas daquele domínio -- limiares que descartam ~25%, ~50% e ~75% dos chunks
em média, em vez de valores copiados de outro contexto. "sem_filtro" é rodado de novo junto, pra
comparação pareada dentro do mesmo arquivo.

Limitação assumida: a calibração usa as mesmas perguntas que depois são avaliadas (calibração
dentro da amostra) -- com 7 a 12 perguntas por domínio não dá pra separar um conjunto só de
calibração. Os limiares calculados ficam em limiares_calibrados.json.
"""

import json
import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import numpy as np

import pipeline_base as pb
from comparar_metodos import carregar_grendene, carregar_revalida, carregar_sbc
from testar_compressao import combinacoes_ja_feitas

LOG_PATH = pb.METRICAS_DIR / "comparativo_compressao_calibrada.jsonl"
LIMIARES_PATH = pb.METRICAS_DIR / "limiares_calibrados.json"
QUANTIS = {"p25": 25, "p50": 50, "p75": 75}
K = 4


def calibrar(colecao, perguntas) -> tuple[dict, dict]:
    sims = [s for p in perguntas for _, s in pb.recuperar_com_similaridade(colecao, p["query"], k=K)]
    limiares = {"sem_filtro": 0.0}
    limiares.update({rotulo: round(float(np.percentile(sims, q)), 4) for rotulo, q in QUANTIS.items()})
    return limiares, {"n_chunks": len(sims), "min": round(min(sims), 4), "max": round(max(sims), 4)}


def main():
    dominios = [carregar_revalida(), carregar_grendene(), carregar_sbc()]
    ja_feitas = combinacoes_ja_feitas(LOG_PATH)
    if ja_feitas:
        print(f"Retomando: {len(ja_feitas)} combinações já em {LOG_PATH.name}, pulando essas.")
    calibracao = json.loads(LIMIARES_PATH.read_text(encoding="utf-8")) if LIMIARES_PATH.exists() else {}

    for nome, chunks, perguntas in dominios:
        colecao = pb.construir_indice(chunks, colecao_nome=f"compressao_{nome}")
        # Recalibrar a cada retomada daria o mesmo resultado (recuperação determinística), mas
        # reaproveitar o que já foi gravado garante que os limiares não mudam no meio da rodada.
        if nome not in calibracao:
            limiares, distribuicao = calibrar(colecao, perguntas)
            calibracao[nome] = {"limiares": limiares, "distribuicao_sim_top4": distribuicao}
            LIMIARES_PATH.write_text(json.dumps(calibracao, ensure_ascii=False, indent=2), encoding="utf-8")
        limiares = calibracao[nome]["limiares"]
        print(f"[{nome}] limiares calibrados: {limiares}")

        for rotulo_limiar, limiar in limiares.items():
            for p in perguntas:
                if (nome, rotulo_limiar, p["id"]) in ja_feitas:
                    continue
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
                    f"[{nome}/{rotulo_limiar}={limiar}] {p['id']}: sim={registro['similaridade_cosseno']:.4f} "
                    f"tokens={registro['tokens_totais']} chunks_mantidos={n_chunks}"
                )


if __name__ == "__main__":
    main()
