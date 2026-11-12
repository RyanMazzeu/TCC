"""
Similaridade de cosseno com o bge-small-en-v1.5, o embedding do artigo-base (Şakar & Emekci 2025), nos
dois datasets em inglês (Docugami e MedQA): deixa a comparação com a Tabela 1 deles mais justa que a
feita com o BGE-M3 dos experimentos, cuja escala de similaridade é outra.

Mesmo cálculo de pipeline_base.calcular_similaridade (cosseno entre os vetores da resposta gerada e
da resposta de referência, sem instrução de consulta), só trocando o modelo. Local, sem custo.
Saída: resultados/metricas/similaridade_bge_small.jsonl (uma linha por resposta da repetição 1).
"""

import json

import numpy as np
from FlagEmbedding import FlagModel

import benchmarks
import pipeline_base as pb
from rodar_benchmarks import log_do_experimento

MODELO = "BAAI/bge-small-en-v1.5"
DATASETS_EN = ["docugami_10q", "medqa"]
SAIDA = pb.METRICAS_DIR / "similaridade_bge_small.jsonl"


def main():
    modelo = FlagModel(MODELO, use_fp16=False)
    linhas = []
    for nome in DATASETS_EN:
        for llm in [l["rotulo"] for l in pb.LLMS_DO_COMPARATIVO]:
            caminho = log_do_experimento(nome, llm)
            if not caminho.exists():
                continue
            for texto in caminho.read_text(encoding="utf-8").splitlines():
                r = json.loads(texto)
                if r.get("repeticao", 1) == 1:
                    linhas.append(r)
    textos = sorted({t for r in linhas for t in (r["resposta_gerada"] or "", r["resposta_esperada"] or "")})
    print(f"{len(linhas)} respostas, {len(textos)} textos distintos", flush=True)
    vetores = modelo.encode(textos, batch_size=64)
    vetores = vetores / np.linalg.norm(vetores, axis=1, keepdims=True)
    indice = {t: i for i, t in enumerate(textos)}
    with SAIDA.open("w", encoding="utf-8") as f:
        for r in linhas:
            a, b = vetores[indice[r["resposta_gerada"] or ""]], vetores[indice[r["resposta_esperada"] or ""]]
            f.write(json.dumps({"dataset": r["dataset"], "llm": r["llm"], "metodo_rag": r["metodo_rag"],
                                "pergunta_id": r["pergunta_id"], "similaridade_bge_small": round(float(a @ b), 6)},
                               ensure_ascii=False) + "\n")
    print(f"Escrito {SAIDA.name}")


if __name__ == "__main__":
    main()
