"""
Fluxogramas do artigo, no estilo das Figs. 2, 4-7, 10 e 11 de Şakar e Emekci (2025, CC BY 4.0), em português e
com os parâmetros deste trabalho: o pipeline de RAG (indexação e consulta) e os seis métodos de injeção de
contexto lado a lado. As caixas em laranja são chamadas ao LLM, de modo que contá-las dá o número de chamadas de
cada método. Chamado por gerar_artigo_benchmarks.py, que grava os PDFs em resultados/graficos_benchmarks/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

BEGE = "#fbe3c4"      # caixas comuns (dados, prompts, respostas)
LARANJA_LLM = "#f2b36b"  # chamadas ao LLM
BORDA = "#3b3a38"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
FONTE = "DejaVu Sans"
# Mesma cor de cada método na figura de custo × desempenho.
COR_METODO = {"stuff": "#2a78d6", "refine": "#eda100", "map_reduce": "#e87ba4", "map_rerank": "#008300",
              "query_step_down": "#6250d6", "reciprocal": "#1baf7a"}


def caixa(ax, x, y, w, h, texto, fs=5.6, llm=False, cor=None, lw=0.7, negrito=True):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                fc=cor or (LARANJA_LLM if llm else BEGE), ec=BORDA, lw=lw, zorder=3))
    ax.text(x, y, texto, ha="center", va="center", fontsize=fs, fontfamily=FONTE, color=TEXTO, zorder=4,
            fontweight="bold" if negrito else "normal", linespacing=1.15)
    return (x, y, w, h)


def pilha(ax, x, y, w, h, texto, n=3, d=0.8, fs=5.6, inclinada=False):
    """Pilha de documentos (folhas deslocadas); inclinada = paralelogramos, como os documentos de entrada."""
    for i in range(n):  # de trás para a frente: a última folha, embaixo à direita, fica na frente
        cx = x - (n - 1) * d / 2 + i * d
        cy = y + (n - 1) * d / 2 - i * d
        if inclinada:
            s = h * 0.25
            pts = [(cx - w / 2 + s, cy + h / 2), (cx + w / 2 + s, cy + h / 2), (cx + w / 2 - s, cy - h / 2),
                   (cx - w / 2 - s, cy - h / 2)]
            ax.add_patch(Polygon(pts, closed=True, fc=BEGE, ec=BORDA, lw=0.7, zorder=3))
        else:
            ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.3",
                                        fc=BEGE, ec=BORDA, lw=0.7, zorder=3))
    fx = x + (n - 1) * d / 2
    fy = y - (n - 1) * d / 2
    ax.text(fx, fy, texto, ha="center", va="center", fontsize=fs, fontfamily=FONTE, color=TEXTO, zorder=4,
            fontweight="bold", linespacing=1.15)
    return (fx, fy, w, h)


def cilindro(ax, x, y, w, h, texto, fs=5.6):
    e = h * 0.22
    ax.add_patch(Ellipse((x, y - h / 2 + e / 2), w, e, fc=BEGE, ec=BORDA, lw=0.7, zorder=3))
    ax.add_patch(Rectangle((x - w / 2, y - h / 2 + e / 2), w, h - e, fc=BEGE, ec="none", zorder=3))
    ax.plot([x - w / 2, x - w / 2], [y - h / 2 + e / 2, y + h / 2 - e / 2], color=BORDA, lw=0.7, zorder=3)
    ax.plot([x + w / 2, x + w / 2], [y - h / 2 + e / 2, y + h / 2 - e / 2], color=BORDA, lw=0.7, zorder=3)
    ax.add_patch(Ellipse((x, y + h / 2 - e / 2), w, e, fc=BEGE, ec=BORDA, lw=0.7, zorder=3))
    ax.text(x, y - e / 3, texto, ha="center", va="center", fontsize=fs, fontfamily=FONTE, color=TEXTO, zorder=4,
            fontweight="bold", linespacing=1.15)
    return (x, y, w, h)


def seta(ax, a, b, rad=0.0, cor=BORDA):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6.5, lw=0.7, color=cor, shrinkA=0,
                                 shrinkB=0, connectionstyle=f"arc3,rad={rad}", zorder=2))


def borda(c, lado):
    """Ponto no meio de um lado da caixa c = (x, y, w, h): 'l', 'r', 't' ou 'b'."""
    x, y, w, h = c
    return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}[lado]


def _eixo(ax, largura, altura):
    ax.set_xlim(0, largura)
    ax.set_ylim(0, altura)
    ax.set_aspect("equal")
    ax.axis("off")


def figura_pipeline(destino: Path):
    fig = plt.figure(figsize=(7.1, 2.45))
    ax = fig.add_axes([0, 0, 1, 1])
    _eixo(ax, 71, 24.5)
    # Indexação, uma vez por conjunto
    docs = pilha(ax, 6, 11, 8, 4.6, "DOCUMENTOS", inclinada=True)
    frag = caixa(ax, 17.5, 11, 8.6, 6, "DIVISÃO EM\nFRAGMENTOS\n1.000 caracteres")
    emb1 = caixa(ax, 28.5, 11, 8.2, 6, "MODELO DE\nEMBEDDING\nBGE-M3")
    vet = pilha(ax, 39, 11, 8, 5, "VETORES DOS\nFRAGMENTOS")
    bd = cilindro(ax, 50, 11, 8.6, 9, "BANCO\nVETORIAL\nChromaDB")
    seta(ax, (docs[0] + 4.6, 11), borda(frag, "l"))
    seta(ax, borda(frag, "r"), borda(emb1, "l"))
    seta(ax, borda(emb1, "r"), (39 - 0.8 - 4, 11.8))
    seta(ax, (vet[0] + 4, vet[1] + 1.5), borda(bd, "l"), rad=-0.25)
    # Consulta, a cada pergunta
    ax.text(64.5, 23.2, "PERGUNTA DO USUÁRIO", ha="center", va="center", fontsize=6.2, fontfamily=FONTE,
            fontweight="bold", color=TEXTO)
    emb2 = caixa(ax, 60, 17.5, 8, 4.6, "MODELO DE\nEMBEDDING")
    busca = caixa(ax, 60, 10, 8, 4.6, "BUSCA\nSEMÂNTICA")
    recup = caixa(ax, 65.6, 2.8, 10, 4.6, "FRAGMENTOS\nRECUPERADOS (k = 4)", fs=5)
    llm = caixa(ax, 68.5, 10, 4.6, 4.6, "LLM", llm=True)
    resp = caixa(ax, 68.5, 17.5, 4.6, 4.6, "RESPOSTA", fs=4.8)
    seta(ax, (62.5, 22.4), borda(emb2, "t"), rad=0.3)
    seta(ax, borda(emb2, "b"), borda(busca, "t"))
    seta(ax, borda(busca, "l"), (bd[0] + bd[2] / 2, 10.5))
    seta(ax, (bd[0] + 1, bd[1] - bd[3] / 2), borda(recup, "l"), rad=0.3)
    seta(ax, (llm[0], recup[1] + recup[3] / 2), borda(llm, "b"))
    seta(ax, borda(llm, "t"), borda(resp, "b"))
    seta(ax, borda(resp, "t"), (66.5, 22.4), rad=0.3)
    ax.text(28, 1.2, "INDEXAÇÃO (uma vez por conjunto)", ha="center", fontsize=5.6, fontfamily=FONTE,
            style="italic", color=TEXTO_2)
    ax.text(60, 6.4, "CONSULTA\n(a cada pergunta)", ha="center", va="center", fontsize=5.2, fontfamily=FONTE,
            style="italic", color=TEXTO_2)
    fig.savefig(destino / "pipeline_rag.pdf")
    plt.close(fig)


def _titulo(ax, metodo, texto, largura):
    ax.add_patch(Rectangle((0.3, 19.0), 1.3, 1.3, fc=COR_METODO[metodo], ec="none"))
    ax.text(2.2, 19.65, texto, ha="left", va="center", fontsize=6.6, fontfamily=FONTE, fontweight="bold", color=TEXTO)


def figura_metodos(destino: Path):
    fig, eixos = plt.subplots(3, 2, figsize=(7.1, 6.3))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.06, hspace=0.08)
    L, A = 34.5, 20.5
    for ax in eixos.flat:
        _eixo(ax, L, A)
    ys4 = [15.5, 11.2, 6.9, 2.6]  # quatro linhas paralelas

    # Stuff
    ax = eixos[0, 0]
    _titulo(ax, "stuff", "Stuff  (1 chamada)", L)
    p = caixa(ax, 4.2, 13.5, 7, 3.4, "PERGUNTA")
    f = pilha(ax, 4.6, 6.5, 7.6, 3.6, "4 FRAGMENTOS", fs=4.8)
    pr = caixa(ax, 15, 10, 7.6, 7, "PROMPT\npergunta +\n4 fragmentos", negrito=False)
    l = caixa(ax, 24.2, 10, 5, 3.4, "LLM", llm=True)
    r = caixa(ax, 31, 10, 6, 3.4, "RESPOSTA", fs=5)
    seta(ax, borda(p, "r"), (pr[0] - 3.8, 11.8))
    seta(ax, (f[0] + 3.8, f[1]), (pr[0] - 3.8, 8.2))
    seta(ax, borda(pr, "r"), borda(l, "l"))
    seta(ax, borda(l, "r"), borda(r, "l"))

    # Refine
    ax = eixos[0, 1]
    _titulo(ax, "refine", "Refine  (4 chamadas, em sequência)", L)
    xs = [5, 12.5, 20, 27.5]
    ant = None
    for i, x in enumerate(xs):
        fi = pilha(ax, x, 14.2, 4, 3, f"F{i + 1}", n=1)
        li = caixa(ax, x, 8.2, 5, 3.4, "LLM", llm=True)
        seta(ax, borda(fi, "b"), borda(li, "t"))
        if ant:
            seta(ax, borda(ant, "r"), borda(li, "l"))
        ant = li
    ax.text(8.75, 9.5, "parcial", ha="center", fontsize=4.6, fontfamily=FONTE, style="italic", color=TEXTO_2)
    r = caixa(ax, 27.5, 2.6, 6, 3, "RESPOSTA", fs=5)
    seta(ax, borda(ant, "b"), borda(r, "t"))
    ax.text(12, 3.3, "a resposta é reescrita só se\no novo fragmento acrescentar", ha="center", va="center",
            fontsize=4.8, fontfamily=FONTE, style="italic", color=TEXTO_2)

    # Map-Reduce
    ax = eixos[1, 0]
    _titulo(ax, "map_reduce", "Map-Reduce  (5 chamadas)", L)
    comb = caixa(ax, 24.2, 9, 5.6, 5, "LLM\ncombina", llm=True, fs=5.2)
    r = caixa(ax, 31.2, 9, 5.6, 3.4, "RESPOSTA", fs=4.8)
    for i, y in enumerate(ys4):
        y = y - 0.6
        fi = pilha(ax, 2.8, y, 3.8, 2.8, f"F{i + 1}", n=1)
        li = caixa(ax, 9, y, 4.6, 2.8, "LLM", llm=True)
        pi = caixa(ax, 16, y, 6, 2.8, "parcial", negrito=False, fs=5.2)
        seta(ax, borda(fi, "r"), borda(li, "l"))
        seta(ax, borda(li, "r"), borda(pi, "l"))
        seta(ax, borda(pi, "r"), (comb[0] - comb[2] / 2, 9 + (y - 9) * 0.25), rad=0.0)
    seta(ax, borda(comb, "r"), borda(r, "l"))

    # Map-Rerank
    ax = eixos[1, 1]
    _titulo(ax, "map_rerank", "Map-Rerank  (4 chamadas)", L)
    sel = caixa(ax, 24.2, 9, 5.6, 5, "MAIOR\nNOTA", fs=5.2)
    r = caixa(ax, 31.2, 9, 5.6, 3.4, "RESPOSTA", fs=4.8)
    for i, (y, nota) in enumerate(zip(ys4, [72, 35, 90, 41])):
        y = y - 0.6
        fi = pilha(ax, 2.8, y, 3.8, 2.8, f"F{i + 1}", n=1)
        li = caixa(ax, 9, y, 4.6, 2.8, "LLM", llm=True)
        pi = caixa(ax, 16, y, 6, 2.8, f"resposta\nnota {nota}", negrito=False, fs=4.8, lw=1.4 if nota == 90 else 0.7)
        seta(ax, borda(fi, "r"), borda(li, "l"))
        seta(ax, borda(li, "r"), borda(pi, "l"))
        seta(ax, borda(pi, "r"), (sel[0] - sel[2] / 2, 9 + (y - 9) * 0.25))
    seta(ax, borda(sel, "r"), borda(r, "l"))

    # Query Step-Down e Reciprocal: mesma primeira etapa
    for ax, metodo, titulo, final, final_llm in [
            (eixos[2, 0], "query_step_down", "Query Step-Down  (6 chamadas)", "LLM\ncombina", True),
            (eixos[2, 1], "reciprocal", "Reciprocal  (5 chamadas)", "MAIOR\nNOTA", False)]:
        _titulo(ax, metodo, titulo, L)
        p = caixa(ax, 2.9, 9, 5, 3.4, "PERGUNTA", fs=4.4)
        g = caixa(ax, 9.0, 9, 5.0, 6, "LLM\ngera 4\nperguntas", llm=True, fs=4.8)
        seta(ax, borda(p, "r"), borda(g, "l"))
        fim = caixa(ax, 26.0, 9, 4.8, 5, final, llm=final_llm, fs=4.8)
        r = caixa(ax, 31.6, 9, 5.2, 3.4, "RESPOSTA", fs=4.4)
        for i, y in enumerate(ys4):
            y = y - 0.6
            pi = caixa(ax, 14.6, y, 3.6, 2.8, f"P{i + 1}", negrito=False, fs=5.2)
            si = caixa(ax, 20.2, y, 5.2, 3.0, "busca +\nLLM" if final_llm else "busca + LLM\n+ nota", llm=True,
                       fs=4.3, negrito=False)
            seta(ax, borda(g, "r"), borda(pi, "l"))
            seta(ax, borda(pi, "r"), borda(si, "l"))
            seta(ax, borda(si, "r"), (fim[0] - fim[2] / 2, 9 + (y - 9) * 0.25))
        seta(ax, borda(fim, "r"), borda(r, "l"))
    fig.savefig(destino / "metodos_rag.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    destino = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "resultados" / "graficos_benchmarks"
    figura_pipeline(destino)
    figura_metodos(destino)
