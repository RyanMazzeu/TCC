"""
Mesma figura da ordem dos métodos do artigo (Fig. 2: 7 LLMs × 6 métodos nas 25 perguntas comuns, cor =
posição do método dentro de cada LLM, W de Kendall entre os 7), mas com a similaridade de cosseno entre
a resposta gerada e a de referência (embeddings BGE-M3, o campo similaridade_cosseno dos logs) no lugar
da métrica principal de cada dataset. No Docugami entram as mesmas 14 perguntas da Fig. 2 (as que têm
valor no gabarito), para que a única diferença entre as duas figuras seja a métrica.

Não faz parte do artigo: grava em resultados/graficos_apresentacao/, fora da pasta que o
gerar_artigo_benchmarks.py copia para o Artigo_TCC02.
Uso: python figura_ordem_metodos_cosseno.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analisar_benchmarks as ab
import benchmarks
import gerar_artigo_benchmarks as gab

SAIDA = gab.BASE_DIR / "resultados" / "graficos_apresentacao"


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "pdf.fonttype": 42})
    df = ab.carregar_execucoes()
    resumo = {}
    for nome in benchmarks.DATASETS:
        g = df[(df.dataset == nome) & (df.repeticao == 1) & (df.metodo_rag.isin(ab.METODOS_RAG))
               & (df.pergunta_id.isin(gab.ids_generalizacao(nome)))]
        g = g[g[ab.METRICA_PRINCIPAL[nome][0]].notna()]  # mesmas perguntas da Fig. 2 (14 no Docugami)
        t7 = g.pivot_table(index="llm", columns="metodo_rag", values="similaridade_cosseno", aggfunc="mean")[ab.METODOS_RAG]
        resumo[nome] = {"tabela7": t7, "w7": ab.kendall_w(t7), "p7": ab.kendall_w_p(t7)}
        print(f"{nome}: {g.pergunta_id.nunique()} perguntas, W = {resumo[nome]['w7']:.3f} (p = {resumo[nome]['p7']:.4f})")
    gab.figura_ordem_metodos(resumo, [SAIDA / "ordem_metodos_cosseno.pdf", SAIDA / "ordem_metodos_cosseno.png"], casas=3)
    print(f"Escritos {SAIDA.relative_to(gab.BASE_DIR)}/ordem_metodos_cosseno.pdf e .png")


if __name__ == "__main__":
    main()
