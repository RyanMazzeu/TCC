"""
Valida o juiz RAGAS do TCC contra o RAGBench (Friel et al. 2024): roda a Faithfulness, com a mesma
configuração de avaliar_ragas.py, nas respostas da amostra dados/processados/ragbench_amostra_juiz.jsonl
e mede a AUROC contra o rótulo `adherence_score` do RAGBench (a resposta é ou não sustentada pelos
documentos). Os rótulos foram gerados pelo GPT-4 e conferidos com anotação humana no DelucionQA.

Pra cada subconjunto, compara duas AUROCs sobre as mesmas respostas:
- a do nosso juiz;
- a da Faithfulness que os próprios autores do RAGBench calcularam (campo ragas_faithfulness_ragbench).

Isso substitui a pergunta "o gpt-4o-mini concorda com o Cohere?" por "o juiz concorda com um
gabarito anotado?" -- e vale pros números de RAGAS de todos os LLMs geradores, já que o juiz é o mesmo.

Uso: python validar_juiz_ragbench.py [--juiz gpt-4o-mini]
"""

import argparse
import asyncio
import json
from collections import defaultdict

from sklearn.metrics import roc_auc_score

from avaliar_ragas import BASE_DIR, _pontuar_com_seguranca, montar_metricas

AMOSTRA = BASE_DIR / "dados" / "processados" / "ragbench_amostra_juiz.jsonl"
METRICAS_DIR = BASE_DIR / "resultados" / "metricas"


def _auroc(linhas: list[dict], campo: str) -> float | None:
    pares = [(l["adherence_score"], l[campo]) for l in linhas if l.get(campo) is not None]
    rotulos = {r for r, _ in pares}
    if len(rotulos) < 2:
        return None  # AUROC precisa das duas classes
    return round(roc_auc_score([r for r, _ in pares], [s for _, s in pares]), 3)


def resumir(linhas: list[dict]) -> dict:
    grupos = defaultdict(list)
    for l in linhas:
        grupos[l["subconjunto"]].append(l)
    grupos["geral"] = linhas
    return {
        nome: {
            "n": len(g),
            "falhas_juiz": sum(1 for l in g if l.get("faithfulness_juiz") is None),
            "auroc_nosso_juiz": _auroc(g, "faithfulness_juiz"),
            "auroc_ragas_ragbench": _auroc(g, "ragas_faithfulness_ragbench"),
        }
        for nome, g in grupos.items()
    }


async def main(juiz: str, concorrencia: int, por_minuto: float):
    saida = METRICAS_DIR / f"ragbench_validacao_juiz_{juiz.replace(':', '_')}.jsonl"
    linhas = [json.loads(l) for l in AMOSTRA.read_text(encoding="utf-8").splitlines() if l.strip()]
    feitas = []
    if saida.exists():
        feitas = [json.loads(l) for l in saida.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"Retomando: {len(feitas)} respostas já avaliadas em {saida.name}.")
    faithfulness = montar_metricas(juiz, por_minuto)["faithfulness"]

    async def _avaliar(linha):
        resultado = {}
        await _pontuar_com_seguranca(
            "faithfulness",
            faithfulness.ascore(user_input=linha["pergunta"], response=linha["resposta"], retrieved_contexts=linha["documentos"]),
            resultado,
        )
        return {**linha, "faithfulness_juiz": resultado["faithfulness"], "erro_juiz": resultado.get("faithfulness_erro"), "juiz": juiz}

    with open(saida, "a", encoding="utf-8") as f:
        for inicio in range(len(feitas), len(linhas), concorrencia):
            lote = await asyncio.gather(*(_avaliar(l) for l in linhas[inicio : inicio + concorrencia]))
            for linha in lote:
                f.write(json.dumps(linha, ensure_ascii=False) + "\n")
            f.flush()
            feitas += lote
            print(f"  {len(feitas)}/{len(linhas)}")

    resumo = resumir(feitas)
    (METRICAS_DIR / f"ragbench_validacao_juiz_{juiz.replace(':', '_')}_resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{'subconjunto':12s} {'n':>4s} {'falhas':>6s} {'AUROC juiz':>11s} {'AUROC RAGBench':>15s}")
    for nome, r in resumo.items():
        fmt = lambda v: f"{v:.3f}" if v is not None else "-"  # noqa: E731
        print(f"{nome:12s} {r['n']:4d} {r['falhas_juiz']:6d} {fmt(r['auroc_nosso_juiz']):>11s} {fmt(r['auroc_ragas_ragbench']):>15s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--juiz", default="gpt-4o-mini", help='"gpt-4o-mini" ou "cohere:<modelo>"')
    parser.add_argument("--concorrencia", type=int, default=8)
    parser.add_argument("--por-minuto", type=float, default=50)
    args = parser.parse_args()
    asyncio.run(main(args.juiz, args.concorrencia, args.por_minuto))
