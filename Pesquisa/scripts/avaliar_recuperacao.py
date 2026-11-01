"""
Métricas de recuperação dos benchmarks, sem LLM nenhum (custo zero): pra cada pergunta, busca os
10 fragmentos mais próximos da pergunta original e compara com o gabarito do próprio dataset.

- Docugami: relevância por relatório (o 10-Q citado como fonte na resposta de referência).
- FaQuAD: relevância por parágrafo (o parágrafo de onde a pergunta foi escrita).
Hit@k = algum documento-ouro no top-k; Recall@k = fração dos documentos-ouro no top-k (no Docugami,
50 perguntas pedem 4 relatórios); MRR = 1/posição do primeiro fragmento relevante.

O MedQA fica de fora: o gabarito não marca de onde, nos livros, sai a resposta (a busca dele é
avaliada pela Context Precision/Recall do RAGAS, em avaliar_ragas_benchmarks.py).

A recuperação com a pergunta original é a mesma pra Stuff, Refine, Map-Reduce e Map-Rerank (e é o
contexto semente do Step-Down e do Reciprocal), então isto mede o recuperador, não o método.
"""

import json
import os
import statistics

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import benchmarks
import pipeline_base as pb

K_MAX = 10
KS = [1, 2, 4, 8, 10]
SAIDA_POR_PERGUNTA = pb.METRICAS_DIR / "recuperacao_benchmarks.jsonl"
SAIDA_RESUMO = pb.METRICAS_DIR / "recuperacao_benchmarks_resumo.json"


def avaliar_dataset(nome: str) -> list[dict]:
    chunks, perguntas = benchmarks.carregar(nome)
    colecao = benchmarks.obter_indice(nome, chunks)
    linhas = []
    for p in perguntas:
        relevantes = set(p["documentos_ouro"])
        if not relevantes:
            continue  # 2 perguntas do Docugami não citam o relatório de origem
        ids = pb.consultar_indice(colecao, p["pergunta"], k=K_MAX)["ids"]
        unidades = [benchmarks.documento_do_chunk(i) for i in ids]
        primeira = next((pos for pos, u in enumerate(unidades, start=1) if u in relevantes), None)
        linha = {"dataset": nome, "pergunta_id": p["id"], "n_relevantes": len(relevantes), "ids_top10": ids,
                 "mrr": 1 / primeira if primeira else 0.0}
        for k in KS:
            achados = relevantes & set(unidades[:k])
            linha[f"hit@{k}"] = float(bool(achados))
            linha[f"recall@{k}"] = len(achados) / len(relevantes)
        linhas.append(linha)
    print(f"{nome}: {len(linhas)}/{len(perguntas)} perguntas com gabarito de recuperação, {len(chunks)} fragmentos")
    return linhas


def main():
    todas, resumo = [], {}
    for nome in benchmarks.DATASETS_COM_TRECHO_CERTO:
        linhas = avaliar_dataset(nome)
        todas += linhas
        metricas = ["mrr"] + [f"{m}@{k}" for m in ("hit", "recall") for k in KS]
        resumo[nome] = {"n": len(linhas), **{m: round(statistics.mean(l[m] for l in linhas), 4) for m in metricas}}
    SAIDA_POR_PERGUNTA.write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in todas), encoding="utf-8")
    SAIDA_RESUMO.write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'dataset':14s} {'n':>4s} {'MRR':>6s} " + " ".join(f"{'Hit@' + str(k):>7s}" for k in KS)
          + " " + " ".join(f"{'Rec@' + str(k):>7s}" for k in KS))
    for nome, r in resumo.items():
        print(f"{nome:14s} {r['n']:4d} {r['mrr']:6.3f} " + " ".join(f"{r[f'hit@{k}']:7.3f}" for k in KS)
              + " " + " ".join(f"{r[f'recall@{k}']:7.3f}" for k in KS))


if __name__ == "__main__":
    main()
