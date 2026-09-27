"""
Análise estatística e figuras dos resultados finais -- o que entra no artigo do TCC02.

Lê os JSONL de resultados/metricas e gera:
- resultados/tabelas/*.tex   tabelas booktabs prontas pra \\input{} no LaTeX
- resultados/graficos/*.pdf  figuras vetoriais (e .png, pra conferir rápido)
- resultados/analise_estatistica.md  todos os números e testes num lugar só, pra redação

Testes: Friedman (métodos pareados por pergunta) + Wilcoxon pareado par a par com correção de
Holm; IC 95% por bootstrap percentil (10.000 reamostragens, seed fixa); Spearman pra correlação
entre métricas e entre juízes. Com 27 perguntas os testes têm pouco poder -- p > 0,05 aqui quer
dizer "não dá pra afirmar diferença", não "não há diferença".

Roda com o que existir: se um arquivo (ex.: RAGAS do v3) ainda estiver incompleto, a análise
correspondente usa as linhas disponíveis e o resumo registra quantas eram.
"""

import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
METRICAS_DIR = BASE_DIR / "resultados" / "metricas"
TABELAS_DIR = BASE_DIR / "resultados" / "tabelas"
GRAFICOS_DIR = BASE_DIR / "resultados" / "graficos"
RESUMO_PATH = BASE_DIR / "resultados" / "analise_estatistica.md"

METODOS = ["stuff", "refine", "map_reduce", "map_rerank", "query_step_down", "reciprocal"]
METODOS_COM_CONTEXTO = METODOS[:4]
NOME_METODO = {
    "stuff": "Stuff", "refine": "Refine", "map_reduce": "Map-Reduce", "map_rerank": "Map-Rerank",
    "query_step_down": "Query Step-Down", "reciprocal": "Reciprocal",
}
DOMINIOS = ["revalida", "grendene", "sbc"]
NOME_DOMINIO = {"revalida": "Revalida", "grendene": "Grendene", "sbc": "SBC"}
NOME_DOMINIO_LONGO = {"revalida": "Revalida (médico)", "grendene": "Grendene (financeiro)", "sbc": "SBC (técnico-científico)"}
METRICAS_RAGAS = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
NOME_METRICA = {
    "answer_relevancy": "Answer Relevancy", "faithfulness": "Faithfulness",
    "context_precision": "Context Precision", "context_recall": "Context Recall",
    "similaridade_cosseno": "Similaridade cosseno",
}
# Recusa = o modelo declarou não ter a resposta. Padrões tirados das respostas reais do v3
# ("Não sei.", "Não há informação relevante", "O contexto não fornece/menciona...").
RECUSA = re.compile(
    r"^\s*n[ãa]o sei\b|n[ãa]o h[áa] informa[çc][ãa]o relevante|o contexto n[ãa]o (?:fornece|menciona|cont[ée]m|apresenta)",
    re.IGNORECASE,
)
SEED = 42
N_BOOTSTRAP = 10_000

# Paleta da skill de dataviz, validada (3 slots, all-pairs, modo claro). O aqua fica abaixo de 3:1
# de contraste com o fundo -> nas figuras com cor por domínio, toda linha tem rótulo direto e
# marcador próprio (identidade nunca só pela cor).
AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COR_DOMINIO = {"revalida": AZUL, "grendene": LARANJA, "sbc": AQUA}
MARCADOR_DOMINIO = {"revalida": "o", "grendene": "s", "sbc": "^"}
TINTA, TINTA_2, MUDO, GRADE, EIXO, NEUTRO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
RAMPA_AZUL = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": EIXO,
    "axes.labelcolor": TINTA_2,
    "axes.titlesize": 10,
    "axes.titlecolor": TINTA,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUDO,
    "ytick.color": MUDO,
    "xtick.labelcolor": TINTA_2,
    "ytick.labelcolor": TINTA_2,
    "grid.color": GRADE,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

resumo: list[str] = []


# ---------------------------------------------------------------- utilidades


def ler_jsonl(nome: str) -> pd.DataFrame:
    caminho = METRICAS_DIR / nome
    if not caminho.exists():
        return pd.DataFrame()
    linhas = [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(linhas)
    if "ragas" in df.columns:
        ragas = pd.json_normalize(df["ragas"]).reindex(columns=METRICAS_RAGAS)
        df = pd.concat([df.drop(columns=["ragas"]), ragas], axis=1)
    if "pergunta_id" in df.columns:
        df["uid"] = df["dominio"] + "/" + df["pergunta_id"]
    return df


def br(x, casas: int = 3) -> str:
    """Número no formato brasileiro (vírgula decimal, ponto de milhar)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    texto = f"{x:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def br_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "--"
    return "< 0,001" if p < 0.001 else br(p, 3)


def ic_bootstrap(valores) -> tuple[float, float]:
    valores = np.asarray([v for v in valores if v is not None and not np.isnan(v)])
    if len(valores) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(SEED)
    medias = rng.choice(valores, size=(N_BOOTSTRAP, len(valores)), replace=True).mean(axis=1)
    return (float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5)))


def holm(pvalores: list[float]) -> list[float]:
    ordem = np.argsort(pvalores)
    m = len(pvalores)
    ajustados = np.empty(m)
    acumulado = 0.0
    for posicao, i in enumerate(ordem):
        acumulado = max(acumulado, min(1.0, (m - posicao) * pvalores[i]))
        ajustados[i] = acumulado
    return ajustados.tolist()


def friedman_com_posthoc(df: pd.DataFrame, coluna: str, metodos: list[str]) -> dict:
    """Friedman com as perguntas como blocos (só as perguntas com valor em todos os métodos) +
    Wilcoxon pareado par a par, p ajustado por Holm."""
    tabela = df.pivot_table(index="uid", columns="metodo_rag", values=coluna, aggfunc="first")
    tabela = tabela.reindex(columns=metodos).dropna()
    saida = {"n_blocos": len(tabela), "metodos": metodos}
    if len(tabela) < 3:
        return saida
    estatistica, p = stats.friedmanchisquare(*[tabela[m] for m in metodos])
    saida.update({"chi2": float(estatistica), "p": float(p)})
    pares, pvals, difs = [], [], []
    for a, b in combinations(metodos, 2):
        diferenca = tabela[a] - tabela[b]
        try:
            p_par = stats.wilcoxon(tabela[a], tabela[b]).pvalue if (diferenca != 0).any() else 1.0
        except ValueError:
            p_par = 1.0
        pares.append((a, b))
        pvals.append(float(p_par))
        difs.append(float(diferenca.mean()))
    saida["pares"] = [
        {"a": a, "b": b, "dif_media": d, "p": p_bruto, "p_holm": p_ajust}
        for (a, b), d, p_bruto, p_ajust in zip(pares, difs, pvals, holm(pvals))
    ]
    return saida


def escrever_tabela(nome: str, cabecalho: list[str], linhas: list[list[str]], legenda: str, alinhamento: str,
                    nota: str | None = None, cabecalho_superior: str | None = None):
    """`cabecalho_superior`: linha LaTeX pronta (com \\multicolumn/\\cmidrule) acima do cabeçalho,
    pra agrupar colunas quando a tabela é larga demais pra cabeçalhos longos numa linha só."""
    corpo = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\caption{{{legenda}}}",
        f"\\label{{tab:{nome}}}",
        f"\\begin{{tabular}}{{{alinhamento}}}",
        "\\toprule",
    ]
    if cabecalho_superior:
        corpo.append(cabecalho_superior)
    corpo += [
        " & ".join(cabecalho) + " \\\\",
        "\\midrule",
    ]
    for linha in linhas:
        if linha == ["\\midrule"]:
            corpo.append("\\midrule")
        else:
            corpo.append(" & ".join(linha) + " \\\\")
    corpo += ["\\bottomrule", "\\end{tabular}"]
    if nota:
        corpo.append(f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\footnotesize {nota}}}")
    corpo.append("\\end{table}")
    (TABELAS_DIR / f"{nome}.tex").write_text("\n".join(corpo) + "\n", encoding="utf-8")


def salvar_figura(fig, nome: str):
    fig.savefig(GRAFICOS_DIR / f"{nome}.pdf")
    fig.savefig(GRAFICOS_DIR / f"{nome}.png")
    plt.close(fig)


def formatar_eixo_br(eixo, casas: int = 2):
    eixo.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, casas)))


def md_tabela(cabecalho: list[str], linhas: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
        + ["| " + " | ".join(l) + " |" for l in linhas]
    )


# ---------------------------------------------------------------- 1. comparativo dos 6 métodos


def analisar_metodos(v3: pd.DataFrame):
    v3 = v3.copy()
    v3["recusa"] = v3["resposta_gerada"].str.contains(RECUSA)

    resumo.append("## 1. Comparativo dos 6 métodos (v3: 27 perguntas × 6 métodos = "
                  f"{len(v3)} execuções)\n")

    # 1a. similaridade por domínio × método + geral (média, DP, IC95)
    linhas_tex, linhas_md = [], []
    for d in DOMINIOS:
        n = v3.loc[v3.dominio == d, "pergunta_id"].nunique()
        medias = [v3[(v3.dominio == d) & (v3.metodo_rag == m)]["similaridade_cosseno"].mean() for m in METODOS]
        melhor = int(np.nanargmax(medias))
        celulas = [f"\\textbf{{{br(x)}}}" if i == melhor else br(x) for i, x in enumerate(medias)]
        linhas_tex.append([f"{NOME_DOMINIO[d]} ({n})"] + celulas)
        linhas_md.append([f"{NOME_DOMINIO[d]} ({n})"] + [br(x) for x in medias])
    geral = v3.groupby("metodo_rag")["similaridade_cosseno"]
    medias_gerais = [geral.mean()[m] for m in METODOS]
    melhor = int(np.nanargmax(medias_gerais))
    linhas_tex.append(["\\midrule"])
    linhas_tex.append(["Geral (27)"] + [f"\\textbf{{{br(x)}}}" if i == melhor else br(x) for i, x in enumerate(medias_gerais)])
    linhas_tex.append(["Desvio padrão"] + [br(geral.std()[m]) for m in METODOS])
    ics = {m: ic_bootstrap(v3[v3.metodo_rag == m]["similaridade_cosseno"]) for m in METODOS}
    linhas_tex.append(["IC 95\\%"] + [f"[{br(ics[m][0], 2)}; {br(ics[m][1], 2)}]" for m in METODOS])
    linhas_md.append(["**Geral (27)**"] + [br(x) for x in medias_gerais])
    linhas_md.append(["DP"] + [br(geral.std()[m]) for m in METODOS])
    linhas_md.append(["IC 95%"] + [f"[{br(ics[m][0], 2)}; {br(ics[m][1], 2)}]" for m in METODOS])
    escrever_tabela(
        "similaridade_metodos",
        ["Domínio ($n$)"] + [NOME_METODO[m] for m in METODOS],
        linhas_tex,
        "Similaridade de cosseno média entre resposta gerada e resposta de referência, por domínio e método de RAG (BGE-M3, ChromaDB, gpt-4o-mini, $k=4$). Em negrito, o melhor método de cada linha.",
        "l" + "c" * len(METODOS),
        nota="IC 95\\%: intervalo de confiança da média por bootstrap percentil (10.000 reamostragens).",
    )
    resumo.append("### 1.1 Similaridade de cosseno média\n")
    resumo.append(md_tabela(["Domínio (n)"] + [NOME_METODO[m] for m in METODOS], linhas_md) + "\n")

    # 1b. teste de Friedman + post-hoc
    teste = friedman_com_posthoc(v3, "similaridade_cosseno", METODOS)
    resumo.append(f"**Friedman** (27 perguntas como blocos, 6 métodos): χ² = {br(teste['chi2'], 2)}, "
                  f"p = {br_p(teste['p'])}.\n")
    significativos = [pr for pr in teste["pares"] if pr["p_holm"] < 0.05]
    resumo.append("Pares com diferença significativa após Holm (α = 0,05): "
                  + (", ".join(f"{NOME_METODO[pr['a']]} vs {NOME_METODO[pr['b']]} (Δ = {br(pr['dif_media'])}, "
                               f"p_Holm = {br_p(pr['p_holm'])})" for pr in significativos) or "nenhum")
                  + ".\n")
    menores = sorted(teste["pares"], key=lambda pr: pr["p"])[:5]
    resumo.append("Os 5 menores p brutos: " + "; ".join(
        f"{NOME_METODO[pr['a']]} vs {NOME_METODO[pr['b']]}: Δ = {br(pr['dif_media'])}, p = {br_p(pr['p'])}, "
        f"p_Holm = {br_p(pr['p_holm'])}" for pr in menores) + ".\n")
    escrever_tabela(
        "posthoc_similaridade",
        ["Par de métodos", "$\\Delta$ médio", "$p$", "$p$ (Holm)"],
        [[f"{NOME_METODO[pr['a']]} vs {NOME_METODO[pr['b']]}", br(pr["dif_media"]), br_p(pr["p"]), br_p(pr["p_holm"])]
         for pr in sorted(teste["pares"], key=lambda pr: pr["p"])],
        f"Comparações par a par da similaridade de cosseno (Wilcoxon pareado por pergunta, $n = {teste['n_blocos']}$), "
        f"após teste de Friedman ($\\chi^2 = {br(teste['chi2'], 2)}$, $p = {br_p(teste['p'])}$).",
        "lccc",
    )

    # 1c. custo
    custo = v3.groupby("metodo_rag").agg(
        tokens=("tokens_totais", "mean"), tokens_dp=("tokens_totais", "std"),
        tempo=("tempo_total_s", "mean"), tempo_dp=("tempo_total_s", "std"),
        sim=("similaridade_cosseno", "mean"), recusas=("recusa", "sum"),
    ).reindex(METODOS)
    base_tokens = custo.loc["stuff", "tokens"]
    escrever_tabela(
        "custo_metodos",
        ["Método", "Chamadas ao LLM", "Tokens (média $\\pm$ DP)", "Relativo ao Stuff", "Tempo total (s)"],
        [[NOME_METODO[m], chamadas, f"{br(custo.loc[m, 'tokens'], 0)} $\\pm$ {br(custo.loc[m, 'tokens_dp'], 0)}",
          f"{br(custo.loc[m, 'tokens'] / base_tokens, 1)}$\\times$",
          f"{br(custo.loc[m, 'tempo'], 2)} $\\pm$ {br(custo.loc[m, 'tempo_dp'], 2)}"]
         for m, chamadas in zip(METODOS, ["1", "$k$", "$k+1$", "$k$", "6", "5"])],
        "Custo médio por pergunta de cada método de RAG ($k = 4$; 27 perguntas).",
        "lcccc",
    )
    resumo.append("### 1.2 Custo por pergunta\n")
    resumo.append(md_tabela(
        ["Método", "Tokens (média ± DP)", "× Stuff", "Tempo total (s)"],
        [[NOME_METODO[m], f"{br(custo.loc[m, 'tokens'], 0)} ± {br(custo.loc[m, 'tokens_dp'], 0)}",
          br(custo.loc[m, "tokens"] / base_tokens, 1), f"{br(custo.loc[m, 'tempo'], 2)} ± {br(custo.loc[m, 'tempo_dp'], 2)}"]
         for m in METODOS]) + "\n")

    # 1d. recusas
    recusas = v3.pivot_table(index="metodo_rag", columns="dominio", values="recusa", aggfunc="sum").reindex(index=METODOS, columns=DOMINIOS)
    totais = v3.groupby("metodo_rag")["recusa"].sum().reindex(METODOS)
    n_por_dominio = {d: v3.loc[v3.dominio == d, "pergunta_id"].nunique() for d in DOMINIOS}
    escrever_tabela(
        "recusas",
        ["Método"] + [f"{NOME_DOMINIO[d]} ({n_por_dominio[d]})" for d in DOMINIOS] + ["Total (27)"],
        [[NOME_METODO[m]] + [str(int(recusas.loc[m, d])) for d in DOMINIOS] + [f"{int(totais[m])} ({br(100 * totais[m] / 27, 0)}\\%)"]
         for m in METODOS],
        "Número de respostas em que o modelo declarou não ter a informação (\\emph{Não sei}, \\emph{Não há informação relevante}, \\emph{O contexto não fornece...}), por método e domínio.",
        "lcccc",
    )
    sim_sem_recusa = v3[~v3.recusa].groupby("metodo_rag")["similaridade_cosseno"].mean().reindex(METODOS)
    resumo.append("### 1.3 Recusas (o modelo declarou não ter a resposta)\n")
    resumo.append(md_tabela(
        ["Método"] + [NOME_DOMINIO[d] for d in DOMINIOS] + ["Total", "Sim. média sem recusas"],
        [[NOME_METODO[m]] + [str(int(recusas.loc[m, d])) for d in DOMINIOS] + [str(int(totais[m])), br(sim_sem_recusa[m])]
         for m in METODOS]) + "\n")
    sim_recusa = v3[v3.recusa]["similaridade_cosseno"]
    resumo.append(f"Similaridade média das respostas-recusa: {br(sim_recusa.mean())} (n = {len(sim_recusa)}); "
                  f"das demais: {br(v3[~v3.recusa]['similaridade_cosseno'].mean())}.\n")

    # figura 1: similaridade geral com IC
    ordem = sorted(METODOS, key=lambda m: geral.mean()[m])
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    for i, m in enumerate(ordem):
        media = geral.mean()[m]
        lo, hi = ics[m]
        ax.plot([lo, hi], [i, i], color=AZUL, linewidth=2, solid_capstyle="round")
        ax.plot(media, i, "o", color=AZUL, markersize=8, markeredgecolor="white", markeredgewidth=2)
        ax.text(hi + 0.008, i, br(media), va="center", fontsize=8, color=TINTA_2)
    ax.set_yticks(range(len(ordem)), [NOME_METODO[m] for m in ordem])
    ax.set_xlabel("Similaridade de cosseno média (IC 95%)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    formatar_eixo_br(ax.xaxis)
    ax.set_xlim(0.5, 0.88)
    salvar_figura(fig, "similaridade_metodos")

    # figura 2: mapa de calor domínio × método
    matriz = np.array([[v3[(v3.dominio == d) & (v3.metodo_rag == m)]["similaridade_cosseno"].mean() for m in METODOS] for d in DOMINIOS])
    fig, ax = plt.subplots(figsize=(6.4, 2.1))
    im = ax.imshow(matriz, cmap=RAMPA_AZUL, vmin=0.5, vmax=0.85, aspect="auto")
    for i in range(len(DOMINIOS)):
        for j in range(len(METODOS)):
            v = matriz[i, j]
            ax.text(j, i, br(v), ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.70 else TINTA, fontweight="bold" if j == np.nanargmax(matriz[i]) else "normal")
    ax.set_xticks(range(len(METODOS)), [NOME_METODO[m].replace("-", "-\n", 1) if len(NOME_METODO[m]) > 8 else NOME_METODO[m] for m in METODOS], fontsize=8)
    ax.set_yticks(range(len(DOMINIOS)), [NOME_DOMINIO_LONGO[d] for d in DOMINIOS], fontsize=8)
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(METODOS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMINIOS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    barra = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    barra.outline.set_visible(False)
    barra.ax.tick_params(labelsize=7, color=MUDO)
    formatar_eixo_br(barra.ax.yaxis)
    salvar_figura(fig, "similaridade_heatmap")

    # figura 3: custo × qualidade
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    deslocamentos = {"stuff": (8, -3), "refine": (8, -3), "map_reduce": (8, -9), "map_rerank": (8, 2),
                     "query_step_down": (-8, 6), "reciprocal": (-8, -12)}
    for m in METODOS:
        x, y = custo.loc[m, "tokens"], custo.loc[m, "sim"]
        ax.plot(x, y, "o", color=AZUL, markersize=8, markeredgecolor="white", markeredgewidth=2)
        dx, dy = deslocamentos[m]
        ax.annotate(NOME_METODO[m], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    color=TINTA_2, ha="left" if dx > 0 else "right")
    ax.set_xlabel("Tokens médios por pergunta")
    ax.set_ylabel("Similaridade de cosseno média")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 0)))
    formatar_eixo_br(ax.yaxis)
    ax.set_xlim(0, 7800)
    ax.set_ylim(0.62, 0.77)
    salvar_figura(fig, "custo_qualidade")

    return v3, custo, teste


# ---------------------------------------------------------------- 2. RAGAS


def analisar_ragas(v3: pd.DataFrame, ragas: pd.DataFrame):
    resumo.append(f"## 2. RAGAS sobre o v3 (juiz gpt-4o-mini; {len(ragas)} de {len(v3)} linhas avaliadas)\n")
    if ragas.empty:
        return None
    ragas = ragas.merge(v3[["experimento_id", "recusa"]], on="experimento_id", how="left")
    colunas = METRICAS_RAGAS + ["similaridade_cosseno"]
    medias = ragas.groupby("metodo_rag")[colunas].mean().reindex(METODOS)
    validos = ragas.groupby("metodo_rag")[METRICAS_RAGAS].count().reindex(METODOS).fillna(0)
    # contagem de erros vem do JSON bruto (campo <metrica>_erro, que o json_normalize descarta)
    brutos =[json.loads(l) for l in (METRICAS_DIR / "comparativo_6_metodos_v3_ragas.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    erros = {c: sum(1 for r in brutos if f"{c}_erro" in r.get("ragas", {})) for c in METRICAS_RAGAS}

    def celula(m, c, fmt=br):
        if c in METRICAS_RAGAS[1:] and m not in METODOS_COM_CONTEXTO:
            return "n/a"
        return fmt(medias.loc[m, c])

    # Numa recusa ("Não sei.") não há afirmação a verificar e a Faithfulness fica mal definida
    # (o 4o-mini tende a dar 0, o Command A deu 1 na amostra) -- a média sem recusas vai ao lado.
    faith_sr = ragas[ragas.recusa == False].groupby("metodo_rag")["faithfulness"].mean().reindex(METODOS)  # noqa: E712
    melhores = {c: medias.loc[METODOS if c in ("answer_relevancy", "similaridade_cosseno") else METODOS_COM_CONTEXTO, c].idxmax() for c in colunas}
    melhor_sr = faith_sr.loc[METODOS_COM_CONTEXTO].idxmax()

    def celula_sr(m):
        if m not in METODOS_COM_CONTEXTO:
            return "n/a"
        return f"\\textbf{{{br(faith_sr[m])}}}" if m == melhor_sr else br(faith_sr[m])

    linhas_tex = [[NOME_METODO[m]] + [
        (f"\\textbf{{{celula(m, c)}}}" if melhores[c] == m else celula(m, c)) for c in colunas[:2]] + [celula_sr(m)] + [
        (f"\\textbf{{{celula(m, c)}}}" if melhores[c] == m else celula(m, c)) for c in colunas[2:]] for m in METODOS]
    escrever_tabela(
        "ragas_metodos",
        ["Método", "Answer Rel.", "Faithf.", "Faithf. s/ recusas", "Context Prec.", "Context Rec.", "Cosseno"],
        linhas_tex,
        "Métricas RAGAS médias por método (juiz gpt-4o-mini), com a similaridade de cosseno como referência. Em negrito, o melhor valor de cada coluna.",
        "lcccccc",
        nota="Faithf. s/ recusas: média excluindo as respostas em que o modelo declarou não ter a informação, para as quais a métrica não tem afirmações a verificar. n/a: Query Step-Down e Reciprocal fazem recuperações próprias por pergunta alternativa, sem um conjunto único de contextos para avaliar; só Answer Relevancy se aplica a eles. "
             f"Falhas de avaliação registradas como ausentes (não imputadas): Faithfulness {erros['faithfulness']}, Answer Relevancy {erros['answer_relevancy']}, "
             f"Context Precision {erros['context_precision']}, Context Recall {erros['context_recall']}.",
    )
    resumo.append(md_tabela(
        ["Método", "Answer Rel.", "Faithfulness", "Context Prec.", "Context Rec.", "Cosseno", "n válidos (AR/F/CP/CR)"],
        [[NOME_METODO[m]] + [celula(m, c) for c in colunas]
         + ["/".join(str(int(validos.loc[m, c])) for c in METRICAS_RAGAS)] for m in METODOS]) + "\n")
    resumo.append(f"Falhas de avaliação (None, não imputado): {erros}.\n")
    resumo.append("Faithfulness sem recusas: " + ", ".join(f"{NOME_METODO[m]} {br(faith_sr[m])}" for m in METODOS_COM_CONTEXTO) + ".\n")

    # testes
    testes = {}
    for c, metodos in [("faithfulness", METODOS_COM_CONTEXTO), ("answer_relevancy", METODOS)]:
        t = friedman_com_posthoc(ragas, c, metodos)
        testes[c] = t
        if "chi2" in t:
            sig = [pr for pr in t["pares"] if pr["p_holm"] < 0.05]
            resumo.append(f"**Friedman – {NOME_METRICA[c]}** ({t['n_blocos']} perguntas completas, {len(metodos)} métodos): "
                          f"χ² = {br(t['chi2'], 2)}, p = {br_p(t['p'])}. Pares significativos (Holm): "
                          + (", ".join(f"{NOME_METODO[pr['a']]} vs {NOME_METODO[pr['b']]} (Δ = {br(pr['dif_media'])}, p_Holm = {br_p(pr['p_holm'])})" for pr in sig) or "nenhum") + ".\n")

    # correlação cosseno × RAGAS (linha a linha)
    resumo.append("**Correlação de Spearman, linha a linha, entre similaridade de cosseno e cada métrica RAGAS:**\n")
    for c in METRICAS_RAGAS:
        par = ragas[["similaridade_cosseno", c]].dropna()
        if len(par) > 3:
            rho, p = stats.spearmanr(par["similaridade_cosseno"], par[c])
            resumo.append(f"- {NOME_METRICA[c]}: ρ = {br(rho, 2)} (p = {br_p(p)}, n = {len(par)})")
    resumo.append("")

    # Os 4 métodos com contexto recebem os MESMOS k fragmentos pra cada pergunta, então Context
    # Precision/Recall deveriam dar o mesmo valor nos 4 -- a diferença entre eles é ruído do juiz
    # (um reteste embutido nos dados, com 27 perguntas × 4 avaliações).
    com_ctx = ragas[ragas.metodo_rag.isin(METODOS_COM_CONTEXTO)].copy()
    com_ctx["ctx"] = com_ctx["contextos_recuperados"].apply(tuple)
    identicos = int((com_ctx.groupby("uid")["ctx"].nunique() == 1).sum())
    resumo.append(f"**Ruído do juiz em entradas idênticas:** em {identicos} de {com_ctx.uid.nunique()} perguntas os 4 métodos com contexto "
                  "receberam exatamente os mesmos fragmentos, então Context Precision e Context Recall deveriam coincidir entre eles.")
    for c in ["context_precision", "context_recall"]:
        amplitude = com_ctx.groupby("uid")[c].agg(lambda v: v.max() - v.min())
        resumo.append(f"- {NOME_METRICA[c]}: mesmo valor nos 4 métodos em {int((amplitude < 1e-6).sum())} de {len(amplitude)} perguntas; "
                      f"amplitude média entre os 4 = {br(amplitude.mean())}; amplitude máxima = {br(amplitude.max())}.")
    resumo.append("")

    # recusas × context recall: o contexto tinha a resposta e o modelo recusou?
    recusas_ctx = ragas[(ragas.recusa == True) & ragas.metodo_rag.isin(METODOS_COM_CONTEXTO)]  # noqa: E712
    if len(recusas_ctx):
        com_recall_alto = recusas_ctx[recusas_ctx.context_recall >= 0.5]
        resumo.append(f"**Recusas × Context Recall** (métodos com contexto): {len(recusas_ctx)} recusas; em "
                      f"{len(com_recall_alto)} delas o Context Recall era ≥ 0,5 (o juiz considerou que o contexto recuperado "
                      f"cobria ao menos metade do gabarito). Context Recall médio nas recusas: {br(recusas_ctx.context_recall.mean())}.\n")
        por_metodo = recusas_ctx.groupby("metodo_rag").agg(n=("recusa", "size"), recall=("context_recall", "mean")).reindex(METODOS_COM_CONTEXTO)
        resumo.append(md_tabela(["Método", "Recusas", "Context Recall médio nelas"],
                                [[NOME_METODO[m], str(int(por_metodo.loc[m, 'n'])) if not np.isnan(por_metodo.loc[m, 'n']) else "0",
                                  br(por_metodo.loc[m, "recall"])] for m in METODOS_COM_CONTEXTO]) + "\n")

    # figura: mapa de calor RAGAS
    matriz = np.array([[medias.loc[m, c] if not (c in METRICAS_RAGAS[1:] and m not in METODOS_COM_CONTEXTO) else np.nan
                        for c in colunas] for m in METODOS])
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    mascarada = np.ma.masked_invalid(matriz)
    cmap = RAMPA_AZUL.copy()
    cmap.set_bad(NEUTRO)
    ax.imshow(mascarada, cmap=cmap, vmin=0.3, vmax=0.95, aspect="auto")
    for i in range(len(METODOS)):
        for j in range(len(colunas)):
            v = matriz[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color=MUDO)
            else:
                ax.text(j, i, br(v), ha="center", va="center", fontsize=8.5, color="white" if v > 0.72 else TINTA)
    ax.set_xticks(range(len(colunas)), ["Answer\nRelevancy", "Faithfulness", "Context\nPrecision", "Context\nRecall", "Similaridade\ncosseno"], fontsize=8)
    ax.set_yticks(range(len(METODOS)), [NOME_METODO[m] for m in METODOS], fontsize=8)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(colunas)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METODOS)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.axvline(len(colunas) - 1.5, color="white", linewidth=6)
    salvar_figura(fig, "ragas_heatmap")
    return medias, testes


# ---------------------------------------------------------------- 3. concordância entre juízes


def _concordancia(pares: pd.DataFrame, c: str, sufixo_a: str, sufixo_b: str) -> dict:
    par = pares[[f"{c}{sufixo_a}", f"{c}{sufixo_b}"]].dropna()
    a, b = par[f"{c}{sufixo_a}"], par[f"{c}{sufixo_b}"]
    if len(par) <= 3:
        return {"n": len(par), "rho": np.nan, "mad": np.nan, "vies": np.nan, "p_vies": np.nan, "media_a": np.nan, "media_b": np.nan}
    rho, _ = stats.spearmanr(a, b)
    p_vies = stats.wilcoxon(a, b).pvalue if (a != b).any() else 1.0
    return {"n": len(par), "rho": float(rho), "mad": float((a - b).abs().mean()), "vies": float((a - b).mean()),
            "p_vies": float(p_vies), "media_a": float(a.mean()), "media_b": float(b.mean())}


def analisar_juizes(ragas_mini: pd.DataFrame, ragas_cohere: pd.DataFrame, ragas_reteste: pd.DataFrame, v3: pd.DataFrame):
    """Compara a discordância entre juízes (4o-mini × Command A) com a variação do próprio 4o-mini
    avaliando as mesmas respostas duas vezes: só faz sentido chamar de viés a parte da discordância
    que passa do ruído de reteste."""
    resumo.append("## 3. Validação do juiz: gpt-4o-mini × Command A (Cohere) e reteste do gpt-4o-mini\n")
    if ragas_mini.empty or ragas_cohere.empty:
        resumo.append("Sem dados suficientes ainda.\n")
        return None
    colunas = ["experimento_id"] + METRICAS_RAGAS
    pares = (ragas_cohere[colunas + ["metodo_rag", "dominio"]].rename(columns={c: f"{c}_cohere" for c in METRICAS_RAGAS})
             .merge(ragas_mini[colunas].rename(columns={c: f"{c}_mini" for c in METRICAS_RAGAS}), on="experimento_id"))
    if not ragas_reteste.empty:
        pares = pares.merge(ragas_reteste[colunas].rename(columns={c: f"{c}_reteste" for c in METRICAS_RAGAS}),
                            on="experimento_id", how="left")
    pares = pares.merge(v3[["experimento_id", "recusa"]], on="experimento_id", how="left")
    tem_reteste = "answer_relevancy_reteste" in pares.columns
    resumo.append(f"Amostra estratificada: {len(ragas_cohere)} linhas avaliadas pelo Command A (3 por domínio × método, seed 42); "
                  f"{len(pares)} também já avaliadas pelo 4o-mini na rodada principal"
                  + (f"; {int(pares['answer_relevancy_reteste'].notna().sum())} com reteste do 4o-mini" if tem_reteste else "") + ".\n")

    linhas_tex, linhas_md = [], []
    for c in METRICAS_RAGAS:
        inter = _concordancia(pares, c, "_mini", "_cohere")
        intra = _concordancia(pares, c, "_mini", "_reteste") if tem_reteste else None
        sem_recusa = _concordancia(pares[pares.recusa == False], c, "_mini", "_cohere")  # noqa: E712
        celulas = [NOME_METRICA[c]]
        if intra:
            celulas += [str(intra["n"]), br(intra["rho"], 2), br(intra["mad"])]
        celulas += [str(inter["n"]), br(inter["rho"], 2), br(inter["mad"]), br(inter["vies"]), br_p(inter["p_vies"]),
                    br(sem_recusa["rho"], 2)]
        linhas_tex.append(celulas)
        linhas_md.append(celulas)
        resumo.append(f"- {NOME_METRICA[c]}: média 4o-mini = {br(inter['media_a'])}, média Command A = {br(inter['media_b'])} (n = {inter['n']})")
    resumo.append("")
    cab_intra = ["$n$", "$\\rho$", "Dif. abs."] if tem_reteste else []
    escrever_tabela(
        "concordancia_juizes",
        ["Métrica"] + cab_intra + ["$n$", "$\\rho$", "Dif. abs.", "Viés", "$p$", "$\\rho$ s/ recusas"],
        linhas_tex,
        "Concordância do juiz das métricas RAGAS numa amostra estratificada de respostas. Reteste: o gpt-4o-mini avaliando as mesmas respostas uma segunda vez. Inter-juiz: gpt-4o-mini contra o Cohere Command A, de outra família de modelos.",
        "l" + ("ccc|" if tem_reteste else "") + "cccccc",
        nota="$\\rho$: correlação de Spearman; Dif. abs.: diferença absoluta média; Viés: média de (4o-mini $-$ Command A), com $p$ do teste de Wilcoxon pareado; "
             "$\\rho$ s/ recusas: correlação inter-juiz excluindo as respostas em que o modelo declarou não ter a informação.",
        cabecalho_superior=("& \\multicolumn{3}{c|}{Reteste (4o-mini $\\times$ 4o-mini)} & "
                            "\\multicolumn{6}{c}{Inter-juiz (4o-mini $\\times$ Command A)} \\\\ "
                            "\\cmidrule(lr){2-4} \\cmidrule(lr){5-10}") if tem_reteste else None,
    )
    resumo.append(md_tabela(["Métrica"] + (["n reteste", "ρ reteste", "Dif. abs. reteste"] if tem_reteste else [])
                            + ["n inter", "ρ inter", "Dif. abs. inter", "Viés (mini − A)", "p viés", "ρ inter sem recusas"], linhas_md) + "\n")

    # figura: dispersão reteste (linha de cima) e inter-juiz (linha de baixo)
    linhas_fig = [("_reteste", "gpt-4o-mini (2ª avaliação)")] if tem_reteste else []
    linhas_fig.append(("_cohere", "Command A"))
    fig, eixos = plt.subplots(len(linhas_fig), 4, figsize=(7.2, 2.0 * len(linhas_fig) + 0.3), sharex=True, sharey=True, squeeze=False)
    for i, (sufixo, rotulo_y) in enumerate(linhas_fig):
        for j, c in enumerate(METRICAS_RAGAS):
            ax = eixos[i][j]
            par = pares[[f"{c}_mini", f"{c}{sufixo}"]].dropna()
            rho = _concordancia(pares, c, "_mini", sufixo)["rho"]
            ax.plot([0, 1], [0, 1], color=EIXO, linewidth=1, zorder=1)
            ax.scatter(par[f"{c}_mini"], par[f"{c}{sufixo}"], s=20, color=AZUL, alpha=0.7, edgecolor="white", linewidth=0.8, zorder=2)
            ax.set_title((f"{NOME_METRICA[c]}\n" if i == 0 else "") + f"ρ = {br(rho, 2)}", fontsize=8.5)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect("equal")
            ax.grid(True)
            ax.set_axisbelow(True)
            formatar_eixo_br(ax.xaxis, 1)
            formatar_eixo_br(ax.yaxis, 1)
        eixos[i][0].set_ylabel(rotulo_y, fontsize=8)
    fig.supxlabel("gpt-4o-mini (avaliação principal)", fontsize=9, color=TINTA_2, y=-0.02)
    salvar_figura(fig, "concordancia_juizes")

    # ranking de métodos segundo cada juiz, na mesma amostra
    for c, metodos in [("answer_relevancy", METODOS), ("faithfulness", METODOS_COM_CONTEXTO)]:
        m_mini = pares.groupby("metodo_rag")[f"{c}_mini"].mean().reindex(metodos)
        m_cohere = pares.groupby("metodo_rag")[f"{c}_cohere"].mean().reindex(metodos)
        if m_mini.isna().any() or m_cohere.isna().any():
            continue
        tau, p_tau = stats.kendalltau(m_mini, m_cohere)
        ordem_mini = " > ".join(NOME_METODO[m] for m in m_mini.sort_values(ascending=False).index)
        ordem_cohere = " > ".join(NOME_METODO[m] for m in m_cohere.sort_values(ascending=False).index)
        resumo.append(f"**Ranking de métodos por {NOME_METRICA[c]} na amostra** — 4o-mini: {ordem_mini}; "
                      f"Command A: {ordem_cohere}; τ de Kendall = {br(tau, 2)} (p = {br_p(p_tau)}).\n")

    # faithfulness das recusas: sem afirmações a verificar, a métrica fica mal definida
    recusas = pares[pares.recusa == True]  # noqa: E712
    if len(recusas):
        resumo.append(f"**Faithfulness nas recusas da amostra** (n = {len(recusas)}): 4o-mini média {br(recusas['faithfulness_mini'].mean())}, "
                      f"Command A média {br(recusas['faithfulness_cohere'].mean())}.\n")
    return pares


# ---------------------------------------------------------------- 4. compressão contextual


def analisar_compressao(fixa: pd.DataFrame, calibrada: pd.DataFrame):
    resumo.append("## 4. Compressão contextual (método Stuff, 27 perguntas)\n")
    niveis_fixos = ["sem_filtro", "baixo", "medio", "alto"]
    rotulo_fixo = {"sem_filtro": "sem filtro", "baixo": "0,3", "medio": "0,5", "alto": "0,7"}
    niveis_cal = ["sem_filtro", "p25", "p50", "p75"]
    rotulo_cal = {"sem_filtro": "sem filtro", "p25": "P25", "p50": "P50", "p75": "P75"}
    limiares = json.loads((METRICAS_DIR / "limiares_calibrados.json").read_text(encoding="utf-8")) if (METRICAS_DIR / "limiares_calibrados.json").exists() else {}

    def bloco(df, niveis, rotulos, titulo, com_limiar):
        if df.empty:
            return []
        linhas = []
        for d in DOMINIOS:
            base = df[(df.dominio == d) & (df.limiar_rotulo == "sem_filtro")].set_index("pergunta_id")
            for nv in niveis:
                sub = df[(df.dominio == d) & (df.limiar_rotulo == nv)].set_index("pergunta_id")
                if sub.empty:
                    continue
                chunks = sub["compressao_contextual"].apply(lambda c: c["n_chunks_mantidos"]).mean()
                red_tok = 100 * (sub["tokens_totais"].mean() / base["tokens_totais"].mean() - 1)
                var_sim = 100 * (sub["similaridade_cosseno"].mean() / base["similaridade_cosseno"].mean() - 1)
                limiar = (f"{rotulos[nv]} ({br(limiares[d]['limiares'][nv], 3)})" if com_limiar and nv != "sem_filtro" and d in limiares
                          else rotulos[nv])
                linhas.append([NOME_DOMINIO[d], limiar, br(chunks, 2), br(sub["tokens_totais"].mean(), 0),
                               "--" if nv == "sem_filtro" else f"{br(red_tok, 1)}\\%",
                               br(sub["similaridade_cosseno"].mean()),
                               "--" if nv == "sem_filtro" else f"{br(var_sim, 1)}\\%"])
        # linha geral + Wilcoxon pareado (27 perguntas) contra sem filtro
        base = df[df.limiar_rotulo == "sem_filtro"].set_index("uid")
        testes = []
        for nv in niveis[1:]:
            sub = df[df.limiar_rotulo == nv].set_index("uid")
            if sub.empty:
                continue
            comuns = base.index.intersection(sub.index)
            a, b = base.loc[comuns, "similaridade_cosseno"], sub.loc[comuns, "similaridade_cosseno"]
            p = stats.wilcoxon(a, b).pvalue if (a != b).any() else 1.0
            red_tok = 100 * (sub.loc[comuns, "tokens_totais"].mean() / base.loc[comuns, "tokens_totais"].mean() - 1)
            var_sim = 100 * (b.mean() / a.mean() - 1)
            testes.append((rotulos[nv], len(comuns), red_tok, var_sim, p))
        resumo.append(f"### {titulo}\n")
        resumo.append(md_tabela(["Domínio", "Limiar", "Chunks mantidos", "Tokens", "Δ tokens", "Similaridade", "Δ similaridade"],
                                [[c.replace("\\%", "%") for c in l] for l in linhas]) + "\n")
        resumo.append("Geral, pareado por pergunta contra sem filtro (Wilcoxon): " + "; ".join(
            f"{nv}: n = {n}, Δ tokens = {br(rt, 1)}%, Δ similaridade = {br(vs, 1)}%, p = {br_p(p)}" for nv, n, rt, vs, p in testes) + ".\n")
        return linhas, testes

    cab = ["Domínio", "Limiar", "Chunks", "Tokens", "$\\Delta$ tokens", "Similaridade", "$\\Delta$ sim."]
    fixo = bloco(fixa, niveis_fixos, rotulo_fixo, "4.1 Limiares fixos (0,3 / 0,5 / 0,7)", False)
    if fixo:
        escrever_tabela("compressao_fixa", cab, fixo[0],
                        "Compressão contextual com limiares fixos de similaridade (método Stuff, $k = 4$). $\\Delta$: variação relativa à mesma pergunta sem filtro.",
                        "llccccc")
    cal = bloco(calibrada, niveis_cal, rotulo_cal, "4.2 Limiares calibrados por domínio (quantis 25/50/75 da similaridade do top-4)", True)
    if cal:
        escrever_tabela("compressao_calibrada", cab, cal[0],
                        "Compressão contextual com limiares calibrados por domínio: quantis 25, 50 e 75 da similaridade dos chunks do top-4 de todas as perguntas do domínio (valor do limiar entre parênteses).",
                        "llccccc")
        resumo.append("Limiares calibrados: " + "; ".join(
            f"{NOME_DOMINIO[d]}: " + ", ".join(f"{k} = {br(v, 3)}" for k, v in limiares[d]["limiares"].items() if k != "sem_filtro")
            + f" (sim. do top-4 entre {br(limiares[d]['distribuicao_sim_top4']['min'], 3)} e {br(limiares[d]['distribuicao_sim_top4']['max'], 3)})"
            for d in DOMINIOS if d in limiares) + ".\n")

        # figura: curva tokens × similaridade por domínio
        fig, ax = plt.subplots(figsize=(5.8, 3.2))
        for d in DOMINIOS:
            pontos = []
            for nv in niveis_cal:
                sub = calibrada[(calibrada.dominio == d) & (calibrada.limiar_rotulo == nv)]
                if not sub.empty:
                    pontos.append((sub["tokens_totais"].mean(), sub["similaridade_cosseno"].mean(), rotulo_cal[nv]))
            if not pontos:
                continue
            xs, ys, rot = zip(*pontos)
            ax.plot(xs, ys, color=COR_DOMINIO[d], linewidth=2, marker=MARCADOR_DOMINIO[d], markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5)
            # O Revalida passa logo abaixo da linha da Grendene: seus rótulos vão embaixo dos pontos
            abaixo = d == "revalida"
            # "sem filtro" é sempre o ponto mais à direita de cada linha (dito na legenda da figura)
            for x, y, r in zip(xs[1:], ys[1:], rot[1:]):
                ax.annotate(r, (x, y), xytext=(0, -13 if abaixo else 7), textcoords="offset points", fontsize=7,
                            color=MUDO, ha="center")
            ax.annotate(NOME_DOMINIO_LONGO[d], (xs[0], ys[0]), xytext=(8, -16 if abaixo else -3), textcoords="offset points",
                        fontsize=8, color=TINTA_2, ha="left")
        ax.set_xlabel("Tokens médios por pergunta")
        ax.set_ylabel("Similaridade de cosseno média")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: br(v, 0)))
        formatar_eixo_br(ax.yaxis)
        base, topo = ax.get_ylim()
        ax.set_ylim(base - 0.03, topo + 0.01)
        salvar_figura(fig, "compressao_calibrada")
    return fixo, cal


# ---------------------------------------------------------------- 5. bancos vetoriais


def analisar_vectorstores():
    dados = json.loads((METRICAS_DIR / "comparativo_vectorstores.json").read_text(encoding="utf-8"))
    linhas = []
    for r in dados:
        c, f = r["chromadb"], r["faiss"]
        linhas.append([f"{NOME_DOMINIO[r['dominio']]} ({r['n_chunks']})",
                       br(c["tempo_indexacao_s"] * 1000, 1), br(f["tempo_indexacao_s"] * 1000, 1),
                       br(c["tempo_consulta_medio_ms"], 3), br(f["tempo_consulta_medio_ms"], 3),
                       br(c["memoria_indexacao_mb"], 2), br(f["memoria_indexacao_mb"], 2),
                       f"{br(100 * r['concordancia_top_k'], 0)}\\%"])
    escrever_tabela(
        "vectorstores",
        ["Domínio (fragmentos)", "Chroma", "FAISS", "Chroma", "FAISS", "Chroma", "FAISS", "top-$k$"],
        linhas,
        "Comparação de infraestrutura entre ChromaDB (persistente) e FAISS (\\texttt{IndexFlatL2}, em memória) sobre os mesmos vetores BGE-M3.",
        "lccccccc",
        cabecalho_superior=("& \\multicolumn{2}{c}{Indexação (ms)} & \\multicolumn{2}{c}{Consulta (ms)} & "
                            "\\multicolumn{2}{c}{Memória (MB)} & Concordância \\\\ "
                            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}"),
    )
    resumo.append("## 5. Bancos vetoriais\n")
    resumo.append(md_tabela(["Domínio (chunks)", "Index. Chroma (ms)", "Index. FAISS (ms)", "Consulta Chroma (ms)",
                             "Consulta FAISS (ms)", "Mem. Chroma (MB)", "Mem. FAISS (MB)", "Concord. top-k"],
                            [[c.replace("\\%", "%") for c in l] for l in linhas]) + "\n")


# ---------------------------------------------------------------- main


def main():
    TABELAS_DIR.mkdir(parents=True, exist_ok=True)
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    resumo.append("# Análise estatística — resultados finais\n")
    resumo.append("Gerado por `scripts/analisar_resultados.py` (não editar à mão; rodar de novo o script).\n")

    v3 = ler_jsonl("comparativo_6_metodos_v3.jsonl")
    v3, _, _ = analisar_metodos(v3)
    ragas = ler_jsonl("comparativo_6_metodos_v3_ragas.jsonl")
    analisar_ragas(v3, ragas)
    analisar_juizes(ragas, ler_jsonl("comparativo_6_metodos_v3_ragas_juiz_cohere.jsonl"),
                    ler_jsonl("comparativo_6_metodos_v3_ragas_reteste_mini.jsonl"), v3)
    analisar_compressao(ler_jsonl("comparativo_compressao_contextual_v2.jsonl"), ler_jsonl("comparativo_compressao_calibrada.jsonl"))
    analisar_vectorstores()

    RESUMO_PATH.write_text("\n".join(resumo) + "\n", encoding="utf-8")
    print(f"Resumo: {RESUMO_PATH}\nTabelas: {TABELAS_DIR}\nFiguras: {GRAFICOS_DIR}")


if __name__ == "__main__":
    main()
