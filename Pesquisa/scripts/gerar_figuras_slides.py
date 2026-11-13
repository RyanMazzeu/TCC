"""
Figuras dos slides de resultados (pasta "Slide de Apresentação/figuras"), em PNG de 300 dpi, a partir dos mesmos dados
e das mesmas funções de desenho do artigo (gerar_artigo_benchmarks.py). As figuras de vários painéis do artigo saem
aqui um painel por arquivo, num formato mais largo e com a fonte proporcionalmente maior, para caber na tela.

Uso: python gerar_figuras_slides.py   (rodar depois de gerar_artigo_benchmarks.py)
"""

import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analisar_benchmarks as ab
import gerar_artigo_benchmarks as g

SAIDA = g.BASE_DIR.parent / "Slide de Apresentação" / "figuras"
DPI = 300
NOME_CURTO = {"docugami_10q": "docugami", "medqa": "medqa", "faquad": "faquad"}


def _rasterizar(pdf_artigo: str, png: str):
    """Figura do artigo que já cabe bem num slide (larga e baixa): só vira PNG."""
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile", str(g.GRAFICOS / pdf_artigo),
                    str(SAIDA / png.removesuffix(".png"))], check=True)


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "pdf.fonttype": 42})
    df = ab.carregar_execucoes()
    g.TABELAS.mkdir(parents=True, exist_ok=True)
    medias, _ = g.tabela_principal(df)
    tokens = g.tabela_tokens(df)
    custo = g.tabela_custo_usd(df)

    _rasterizar("efeito_rag.pdf", "efeito_rag.png")
    _rasterizar("ordem_metodos.pdf", "ordem_metodos.png")

    # Custo × métrica principal, um conjunto por imagem.
    for nome in ("docugami_10q", "faquad", "medqa"):
        g._figura_contra_custo(medias, custo, g._eixo_y_metrica, None, SAIDA / f"custo_desempenho_{NOME_CURTO[nome]}.png",
                               conjuntos=[nome], figsize=(9.0, 3.55),
                               ajuste=dict(left=0.07, right=0.99, top=0.78, bottom=0.15),
                               legenda_ret=[0.07, 0.845, 0.92, 0.15], dpi=DPI)
    # Custo × tokens: o padrão (uma diagonal por modelo) é o mesmo nos três conjuntos; o slide mostra o Docugami.
    g._figura_contra_custo(tokens, custo, g._eixo_y_tokens, "Tokens por pergunta", SAIDA / "custo_tokens_docugami.png",
                           conjuntos=["docugami_10q"], figsize=(9.0, 3.55),
                           ajuste=dict(left=0.095, right=0.99, top=0.78, bottom=0.15),
                           legenda_ret=[0.095, 0.845, 0.895, 0.15], dpi=DPI)

    ctx = g.carregar_contextos()
    recusas = ctx[ctx.recusa].groupby(["dataset", "metodo_rag", "llm", "veredito_juiz"]).size()
    sem = ctx[ctx.veredito_juiz == "ausente"].assign(conduta=lambda d: np.where(d.recusa, "recusou", "respondeu"))
    sem = sem.groupby(["dataset", "metodo_rag", "llm", "conduta"]).size()
    for nome in g.DATASETS_CONTEXTO:
        g._barras_por_metodo_llm(recusas, g.VEREDITOS_RECUSA, "Recusas", SAIDA / f"recusas_{NOME_CURTO[nome]}.png",
                                 conjuntos=[nome], figsize=(7.6, 2.9),
                                 ajuste=dict(left=0.075, right=0.995, top=0.79, bottom=0.21),
                                 legenda_ancora=(0.065, 1.0), dpi=DPI)
    g._barras_por_metodo_llm(sem, g.CONDUTAS_SEM_INFORMACAO, "Respostas sem a\ninformação no contexto",
                             SAIDA / "sem_informacao_docugami.png", conjuntos=["docugami_10q"], figsize=(7.6, 2.9),
                             ajuste=dict(left=0.085, right=0.995, top=0.83, bottom=0.21),
                             legenda_ancora=(0.075, 1.0), dpi=DPI)
    print(f"Escritas as figuras dos slides em {SAIDA}")


if __name__ == "__main__":
    main()
