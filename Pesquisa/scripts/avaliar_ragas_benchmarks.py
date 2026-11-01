"""
RAGAS sobre os logs da rodada dos benchmarks (rodar_benchmarks.py), com o mesmo juiz e a mesma
configuração de avaliar_ragas.py, em dois modos:

- respostas: Faithfulness e Answer Relevancy por resposta (repetição 1). A Faithfulness só existe nos
  métodos que entregam um conjunto único de trechos ao LLM (Stuff, Refine, Map-Reduce, Map-Rerank); a
  Answer Relevancy vale pra todos, inclusive o controle sem RAG. Saída: ragas_<dataset>_<llm>.jsonl.
- contexto: Context Precision e Context Recall uma vez por pergunta. Essas duas métricas avaliam os
  trechos recuperados contra a resposta de referência, e os trechos são os mesmos em todos os métodos
  e LLMs (a busca usa a pergunta original) -- calcular de novo pra cada linha só mediria o ruído do
  juiz. Usa os trechos do Stuff de um LLM qualquer da camada principal. No MedQA, é a única avaliação
  da busca. Saída: ragas_contexto_<dataset>.jsonl.

Retoma de onde parou. Uso:
  python avaliar_ragas_benchmarks.py respostas [--datasets ...] [--llms ...]
  python avaliar_ragas_benchmarks.py contexto  [--datasets ...]
"""

import argparse
import asyncio
import json

import benchmarks
from avaliar_ragas import avaliar_linha, montar_metricas
from rodar_benchmarks import log_do_experimento

METRICAS_DIR = benchmarks.pb.METRICAS_DIR
METRICAS_RESPOSTA = {"faithfulness", "answer_relevancy"}
METRICAS_CONTEXTO = {"context_precision", "context_recall"}


def _ler(caminho) -> list[dict]:
    if not caminho.exists():
        return []
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def _com_erro(notas: dict) -> bool:
    return any(k == "erro_linha" or k.endswith("_erro") for k in notas)


def _sem_credito(notas: dict) -> bool:
    return any(codigo in json.dumps(notas) for codigo in ("insufficient_quota", "credit_balance_exhausted"))


async def _avaliar_em_lotes(linhas: list[dict], saida, metricas: dict, somente: set[str], chave, concorrencia: int, juiz: str):
    # Linhas com erro saem do arquivo e voltam pra fila: numa retomada (ex.: depois de faltar crédito),
    # são avaliadas de novo em vez de ficarem gravadas como falha.
    gravadas = _ler(saida)
    boas = [r for r in gravadas if not _com_erro(r["ragas"])]
    if len(boas) < len(gravadas):
        saida.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in boas), encoding="utf-8")
    feitas = {chave(r) for r in boas}
    pendentes = [r for r in linhas if chave(r) not in feitas]
    print(f"{saida.name}: {len(feitas)} já avaliadas, {len(pendentes)} pendentes"
          + (f" ({len(gravadas) - len(boas)} com erro, refeitas)" if len(boas) < len(gravadas) else ""))

    async def _uma(registro):
        try:
            return await avaliar_linha(registro, metricas, somente=somente)
        except Exception as e:
            return {"erro_linha": f"{type(e).__name__}: {e}"}

    with open(saida, "a", encoding="utf-8") as f:
        for inicio in range(0, len(pendentes), concorrencia):
            lote = pendentes[inicio : inicio + concorrencia]
            resultados = await asyncio.gather(*(_uma(r) for r in lote))
            if any(_sem_credito(n) for n in resultados):
                raise SystemExit("Sem créditos na API do juiz: adicione créditos e rode o mesmo comando de novo pra retomar")
            for registro, notas in zip(lote, resultados):
                f.write(json.dumps({
                    "dataset": registro["dataset"], "llm": registro["llm"], "metodo_rag": registro["metodo_rag"],
                    "pergunta_id": registro["pergunta_id"], "ragas": notas, "ragas_juiz": juiz,
                }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {min(inicio + concorrencia, len(pendentes))}/{len(pendentes)}", flush=True)


async def respostas(datasets, llms, metricas, concorrencia, juiz):
    for nome in datasets:
        for rotulo in llms:
            linhas = [r for r in _ler(log_do_experimento(nome, rotulo)) if r.get("repeticao", 1) == 1]
            if not linhas:
                continue
            await _avaliar_em_lotes(linhas, METRICAS_DIR / f"ragas_{nome}_{rotulo}.jsonl", metricas,
                                    METRICAS_RESPOSTA, lambda r: (r["metodo_rag"], r["pergunta_id"]), concorrencia, juiz)


async def contexto(datasets, metricas, concorrencia, juiz):
    principais = [l["rotulo"] for l in benchmarks.pb.LLMS_DO_COMPARATIVO if l["camada"] == "principal"]
    for nome in datasets:
        # Um registro de Stuff por pergunta, do primeiro LLM da camada principal que já tiver rodado.
        por_pergunta = {}
        for rotulo in principais:
            for r in _ler(log_do_experimento(nome, rotulo)):
                if r["metodo_rag"] == "stuff" and r.get("repeticao", 1) == 1:
                    por_pergunta.setdefault(r["pergunta_id"], r)
        if not por_pergunta:
            print(f"[{nome}] sem log de Stuff ainda -- rode rodar_benchmarks.py antes.")
            continue
        await _avaliar_em_lotes(list(por_pergunta.values()), METRICAS_DIR / f"ragas_contexto_{nome}.jsonl", metricas,
                                METRICAS_CONTEXTO, lambda r: r["pergunta_id"], concorrencia, juiz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modo", choices=["respostas", "contexto"])
    parser.add_argument("--datasets", nargs="+", default=benchmarks.DATASETS, choices=benchmarks.DATASETS)
    parser.add_argument("--llms", nargs="+", default=[l["rotulo"] for l in benchmarks.pb.LLMS_DO_COMPARATIVO])
    parser.add_argument("--juiz", default="gpt-4o-mini")
    parser.add_argument("--concorrencia", type=int, default=8)
    parser.add_argument("--por-minuto", type=float, default=50)
    args = parser.parse_args()
    metricas = montar_metricas(args.juiz, args.por_minuto)
    if args.modo == "respostas":
        asyncio.run(respostas(args.datasets, args.llms, metricas, args.concorrencia, args.juiz))
    else:
        asyncio.run(contexto(args.datasets, metricas, args.concorrencia, args.juiz))
