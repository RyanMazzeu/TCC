"""
Rodada com os benchmarks (ver plano_execucao_benchmarks.md): os 6 métodos de RAG + o controle sem
RAG × os LLMs de pb.LLMS_DO_COMPARATIVO × os datasets de benchmarks.DATASETS (Docugami SEC 10-Q,
MedQA, FaQuAD), em duas camadas:
- principal: os 5 modelos econômicos, 100 perguntas por dataset;
- generalizacao: os 2 intermediários, só nas 25 primeiras dessas perguntas (todos os 7 modelos
  respondem às mesmas 25).

Repetições: os modelos que não aceitam temperatura 0 rodam o Stuff mais 2 vezes (repetições 2 e 3)
nas 25 perguntas da camada de generalização, pra medir quanto as respostas variam de uma execução pra
outra. Na repetição 1 entram todos os métodos.

Cada dataset tem um índice só, compartilhado por todos os métodos e LLMs. Retoma de onde parou: pula
as combinações (método, pergunta, repetição) que já estão no log -- uma queda de API no meio da rodada
se resolve rodando o mesmo comando de novo. LLMs sem chave no .env são pulados. Cada par (dataset,
LLM) grava no próprio log (benchmarks_<dataset>_<llm>.jsonl), pra dar pra rodar vários processos em
paralelo.

Uso:
  python rodar_benchmarks.py --llms gpt_mini --perguntas 2           # piloto
  python rodar_benchmarks.py --camada principal                      # os 5 econômicos, 100 perguntas
  python rodar_benchmarks.py --camada generalizacao                  # os 2 intermediários, 25 perguntas
  python rodar_benchmarks.py --repeticoes                            # Stuff 2x a mais, modelos sem temperatura
"""

import argparse
import json
import os
import time
from pathlib import Path

import openai

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import benchmarks
import pipeline_base as pb
from testar_llms import motivo_para_pular

METODOS = ["sem_rag", "stuff", "refine", "map_reduce", "map_rerank", "query_step_down", "reciprocal"]
REPETICOES_EXTRAS = [2, 3]

# Quedas de rede e limites de taxa não derrubam a rodada: a mesma pergunta é tentada de novo a cada
# minuto, por até meia hora (a rodada de 25/10 parou numa queda de conexão). Como a linha só vai pro
# log quando o método termina, uma tentativa interrompida não deixa registro pela metade.
_ERROS_TRANSITORIOS = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)
# A OpenAI devolve "sem crédito" como RateLimitError (429), mas não passa sozinho: tentar de novo só
# atrasa a parada da cadeia em continuar_openai.sh.
_SEM_CREDITO = {"insufficient_quota", "credit_balance_exhausted"}
_TENTATIVAS = 30
_ESPERA_S = 60


def _rodar_com_retentativas(**kwargs) -> dict:
    for tentativa in range(1, _TENTATIVAS + 1):
        try:
            return pb.rodar_experimento(**kwargs)
        except _ERROS_TRANSITORIOS as e:
            if getattr(e, "code", None) in _SEM_CREDITO:
                raise SystemExit(f"Sem créditos na API ({e.code}): adicione créditos e rode o mesmo comando de novo pra retomar")
            print(f"  erro transitório ({type(e).__name__}), tentativa {tentativa}/{_TENTATIVAS}; "
                  f"nova tentativa em {_ESPERA_S} s", flush=True)
            time.sleep(_ESPERA_S)
    raise RuntimeError(f"{_TENTATIVAS} tentativas sem sucesso; rode o mesmo comando de novo pra retomar")


def log_do_experimento(dataset: str, llm: str) -> Path:
    return pb.METRICAS_DIR / f"benchmarks_{dataset}_{llm}.jsonl"


def _ja_feitos(log_path: Path) -> set:
    feitos = set()
    if log_path.exists():
        for linha in log_path.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                r = json.loads(linha)
                feitos.add((r["metodo_rag"], r["pergunta_id"], r.get("repeticao", 1)))
    return feitos


def main(datasets: list[str], llms: list[str], metodos: list[str], n_perguntas: int | None, repeticoes: list[int]):
    for nome in datasets:
        chunks, todas = benchmarks.carregar(nome)
        colecao = benchmarks.obter_indice(nome, chunks)
        pb.CONJUNTO_PROMPTS = benchmarks.CONJUNTO_PROMPTS[nome]
        for rotulo in llms:
            llm = pb.llm_por_rotulo(rotulo)
            motivo = motivo_para_pular(llm["backend"])
            if motivo:
                print(f"[{nome}/{rotulo}] pulado: {motivo}")
                continue
            n = n_perguntas if n_perguntas is not None else benchmarks.PERGUNTAS_POR_CAMADA[llm["camada"]]
            perguntas = benchmarks.amostra(todas, n)
            log_path = log_do_experimento(nome, rotulo)
            feitos = _ja_feitos(log_path)
            pb.configurar_llm_por_rotulo(rotulo)
            for repeticao in repeticoes:
                for metodo in metodos:
                    for p in perguntas:
                        if (metodo, p["id"], repeticao) in feitos:
                            continue
                        registro = _rodar_com_retentativas(
                            query=p["pergunta"],
                            colecao=colecao,
                            document_id=nome,
                            resposta_esperada=p["resposta_esperada"],
                            metodo=metodo,
                            log_path=log_path,
                            extra_campos={"dominio": nome, "dataset": nome, "llm": rotulo, "camada": llm["camada"],
                                          "pergunta_id": p["id"], "repeticao": repeticao,
                                          "conjunto_prompts": pb.CONJUNTO_PROMPTS},
                        )
                        print(
                            f"[{nome}/{rotulo}/{metodo}/r{repeticao}] {p['id']}: sim={registro['similaridade_cosseno']:.4f} "
                            f"tokens={registro['tokens_totais']} t_ger={registro['tempo_geracao_s']:.2f}s",
                            flush=True,
                        )


if __name__ == "__main__":
    rotulos = [llm["rotulo"] for llm in pb.LLMS_DO_COMPARATIVO]
    parser = argparse.ArgumentParser()
    parser.add_argument("--camada", choices=list(benchmarks.PERGUNTAS_POR_CAMADA),
                        help="Default: todos os LLMs (cada um com o número de perguntas da sua camada)")
    parser.add_argument("--datasets", nargs="+", default=benchmarks.DATASETS, choices=benchmarks.DATASETS)
    parser.add_argument("--llms", nargs="+", choices=rotulos, help="Default: os LLMs da camada")
    parser.add_argument("--metodos", nargs="+", choices=METODOS, help="Default: os 6 métodos + o controle sem RAG")
    parser.add_argument("--perguntas", type=int, default=None,
                        help="Perguntas por dataset (sobrepõe o número da camada; útil pro piloto)")
    parser.add_argument("--repeticoes", action="store_true",
                        help="Roda só as repetições extras do Stuff, nos modelos sem temperatura 0")
    args = parser.parse_args()

    llms = args.llms or [l["rotulo"] for l in pb.LLMS_DO_COMPARATIVO if args.camada in (None, l["camada"])]
    if args.repeticoes:
        llms = [r for r in llms if not pb.llm_por_rotulo(r)["aceita_temperatura"]]
        n = args.perguntas if args.perguntas is not None else benchmarks.PERGUNTAS_POR_CAMADA["generalizacao"]
        main(args.datasets, llms, ["stuff"], n, REPETICOES_EXTRAS)
    else:
        main(args.datasets, llms, args.metodos or METODOS, args.perguntas, [1])
