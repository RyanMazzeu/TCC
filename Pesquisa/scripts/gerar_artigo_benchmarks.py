"""
Tabelas (LaTeX) e figuras (PDF) do artigo do TCC02 a partir da rodada dos benchmarks, com os mesmos
dados e cálculos de analisar_benchmarks.py. O artigo (Artigo_TCC02/main.tex) lê direto destas pastas:
rodar este script de novo atualiza o artigo.

Saídas: resultados/tabelas_benchmarks/*.tex e resultados/graficos_benchmarks/*.pdf.
Uso: python gerar_artigo_benchmarks.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy import stats

import analisar_benchmarks as ab
import benchmarks
import pipeline_base as pb
from analisar_resultados import BASE_DIR, METRICAS_DIR, friedman_com_posthoc, holm

TABELAS = BASE_DIR / "resultados" / "tabelas_benchmarks"
GRAFICOS = BASE_DIR / "resultados" / "graficos_benchmarks"

PRINCIPAIS = ["qwen", "deepseek", "claude_haiku", "gpt_luna", "gpt_mini"]
TODOS = PRINCIPAIS + ["claude_sonnet", "gpt_sol"]
NOME_LLM = {"qwen": "Qwen", "deepseek": "DeepSeek", "claude_haiku": "Haiku", "gpt_luna": "Luna",
            "gpt_mini": "4o-mini", "claude_sonnet": "Sonnet", "gpt_sol": "Sol"}
NOME_LLM_LONGO = {"qwen": "Qwen3.5-9B", "deepseek": "DeepSeek V4 Pro", "claude_haiku": "Claude Haiku 5.5",
                  "gpt_luna": "GPT-6 Luna", "gpt_mini": "gpt-4o-mini", "claude_sonnet": "Claude Sonnet 5",
                  "gpt_sol": "GPT-6 Sol"}
NOME_DATASET = {"docugami_10q": "Docugami SEC 10-Q", "medqa": "MedQA", "faquad": "FaQuAD"}
METRICA_CURTA = {"docugami_10q": "valores do gabarito", "medqa": "acurácia", "faquad": "cobertura de tokens"}
CHAMADAS = {"sem_rag": "1", "stuff": "1", "refine": "4", "map_reduce": "5", "map_rerank": "4",
            "query_step_down": "6", "reciprocal": "5"}

# Paleta (skill dataviz, instância de referência): categóricas na ordem fixa, texto e superfície.
AZUL, LARANJA, CINZA = "#2a78d6", "#eb6834", "#8a8984"
TEXTO, TEXTO_2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def num(x, casas: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def p_tex(p: float) -> str:
    if p is None or np.isnan(p):
        return "--"
    return "$<$0,001" if p < 0.001 else num(p, 3)


def escrever(nome: str, linhas: list[str]):
    (TABELAS / f"{nome}.tex").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def ids_generalizacao(nome: str) -> set:
    return {p["id"] for p in benchmarks.amostra(benchmarks.carregar_perguntas(nome), benchmarks.PERGUNTAS_POR_CAMADA["generalizacao"])}


def efeito_rag(g: pd.DataFrame, col: str) -> dict:
    """{método: (diferença média contra o sem RAG, p_Holm)} de um LLM num dataset (Wilcoxon pareado)."""
    tabela = g.pivot_table(index="uid", columns="metodo_rag", values=col, aggfunc="first")
    difs, pvals = {}, []
    for m in ab.METODOS_RAG:
        par = tabela[["sem_rag", m]].dropna()
        d = par[m] - par["sem_rag"]
        difs[m] = d.mean()
        pvals.append(stats.wilcoxon(par[m], par["sem_rag"]).pvalue if (d != 0).any() else 1.0)
    return {m: (difs[m], p) for m, p in zip(ab.METODOS_RAG, holm(pvals))}


def tabela_principal(df: pd.DataFrame):
    medias, sig, friedman = {}, {}, {}
    for nome in benchmarks.DATASETS:
        col = ab.METRICA_PRINCIPAL[nome][0]
        for llm in PRINCIPAIS:
            g = df[(df.dataset == nome) & (df.llm == llm) & (df.repeticao == 1)]
            medias[nome, llm] = g.groupby("metodo_rag")[col].mean()
            sig[nome, llm] = efeito_rag(g, col)
            friedman[nome, llm] = friedman_com_posthoc(g, col, ab.METODOS_RAG).get("p", np.nan)
    cab = " & ".join(f"\\multicolumn{{5}}{{c}}{{{NOME_DATASET[n]} ({METRICA_CURTA[n]})}}" for n in benchmarks.DATASETS)
    sub = " & ".join(" & ".join(NOME_LLM[l] for l in PRINCIPAIS) for _ in benchmarks.DATASETS)
    linhas = [
        "\\begin{table*}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.6pt}",
        "\\caption{Métrica principal de cada dataset por método e LLM na camada principal (100 perguntas por dataset; no "
        "Docugami, as 66 com valor numérico no gabarito). Em negrito, o melhor valor de cada coluna. * diferença em relação ao "
        "controle sem RAG com $p_{\\text{Holm}} < 0{,}05$ (Wilcoxon pareado). Na última linha, o valor-$p$ do teste de Friedman "
        "entre os seis métodos de RAG.}",
        "\\label{tab:principal}",
        "\\begin{tabular}{l" + "ccccc".join(["", "@{\\hspace{7pt}}", "@{\\hspace{7pt}}", ""]) + "}",
        "\\toprule", f" & {cab} \\\\",
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-16}", f"Método & {sub} \\\\", "\\midrule",
    ]
    for m in ab.METODOS:
        celulas = []
        for nome in benchmarks.DATASETS:
            for llm in PRINCIPAIS:
                serie = medias[nome, llm]
                v = serie[m]
                texto = num(v)
                if m != "sem_rag" and sig[nome, llm][m][1] < 0.05:
                    texto += "*"
                if np.isclose(v, serie.max()):
                    texto = f"\\textbf{{{texto}}}"
                celulas.append(texto)
        linhas.append(f"{ab.NOME_METODO[m]} & " + " & ".join(celulas) + " \\\\")
        if m == "sem_rag":
            linhas.append("\\midrule")
    linhas += ["\\midrule", "Friedman $p$ & " + " & ".join(p_tex(friedman[n, l]) for n in benchmarks.DATASETS for l in PRINCIPAIS) + " \\\\",
               "\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    escrever("tab_principal", linhas)
    return medias, sig


def ranking(tabela: pd.DataFrame) -> pd.Series:
    """Posto médio de cada método (1 = melhor) entre os LLMs (linhas)."""
    return tabela.rank(axis=1, ascending=False).mean().sort_values()


def tabela_concordancia(df: pd.DataFrame) -> dict:
    resumo = {}
    for nome in benchmarks.DATASETS:
        col = ab.METRICA_PRINCIPAL[nome][0]
        g = df[(df.dataset == nome) & (df.repeticao == 1) & (df.metodo_rag.isin(ab.METODOS_RAG))]
        t5 = g[g.llm.isin(PRINCIPAIS)].pivot_table(index="llm", columns="metodo_rag", values=col, aggfunc="mean")[ab.METODOS_RAG]
        g25 = g[g.pergunta_id.isin(ids_generalizacao(nome))]
        t7 = g25.pivot_table(index="llm", columns="metodo_rag", values=col, aggfunc="mean")[ab.METODOS_RAG]
        resumo[nome] = {"w5": ab.kendall_w(t5), "p5": ab.kendall_w_p(t5), "w7": ab.kendall_w(t7), "p7": ab.kendall_w_p(t7),
                        "ranking": ranking(t5).to_dict(), "tabela7": t7}
    linhas = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
              "\\caption{Posto médio de cada método entre os cinco LLMs da camada principal (1 = melhor) e concordância da ordem "
              "dos seis métodos entre os LLMs (W de Kendall: 1 = mesma ordem em todos; 0 = nenhuma concordância; entre parênteses, "
              "o valor-$p$).}", "\\label{tab:concordancia}",
              "\\begin{tabular}{lccc}", "\\toprule",
              "Método & Docugami & MedQA & FaQuAD \\\\", "\\midrule"]
    for m in ab.METODOS_RAG:
        linhas.append(f"{ab.NOME_METODO[m]} & " + " & ".join(num(resumo[n]["ranking"][m], 1) for n in benchmarks.DATASETS) + " \\\\")
    linhas.append("\\midrule")
    for chave, rotulo in (("5", "W, 5 LLMs (100 perg.)"), ("7", "W, 7 LLMs (25 perg.)")):
        linhas.append(f"{rotulo} & " + " & ".join(f"{num(resumo[n]['w' + chave])} ({p_tex(resumo[n]['p' + chave])})"
                                                   for n in benchmarks.DATASETS) + " \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    escrever("tab_concordancia", linhas)
    return resumo


def tabela_custo(df: pd.DataFrame):
    g = df[(df.llm.isin(PRINCIPAIS)) & (df.repeticao == 1)]
    custo = g.groupby(["dataset", "llm", "metodo_rag"])["custo_usd"].mean().unstack()
    relativo = custo.div(custo["stuff"], axis=0)
    tokens = g.groupby("metodo_rag")["tokens_totais"].mean()
    tempo = g.groupby("metodo_rag")["tempo_total_s"].median()
    linhas = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
              "\\caption{Custo por resposta na camada principal (5 LLMs, 3 datasets, 1.500 respostas por método). Custo relativo: "
              "custo em US\\$ do método dividido pelo do Stuff, por LLM e dataset (mediana e faixa dos 15 pares). Tempo: da "
              "recuperação ao fim da geração.}", "\\label{tab:custo}",
              "\\begin{tabular}{lcrcc}", "\\toprule",
              "Método & Chamadas & Tokens & \\shortstack{Custo relativo\\\\ao Stuff} & \\shortstack{Tempo (s,\\\\mediana)} \\\\", "\\midrule"]
    for m in ab.METODOS:
        r = relativo[m]
        faixa = f"{num(r.median(), 1)} ({num(r.min(), 1)}--{num(r.max(), 1)})" if m != "stuff" else "1"
        linhas.append(f"{ab.NOME_METODO[m]} & {CHAMADAS[m]} & {num(tokens[m], 0)} & {faixa} & {num(tempo[m], 1)} \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    escrever("tab_custo", linhas)
    return relativo, tokens, tempo


def tabela_llms(df: pd.DataFrame):
    info = {l["rotulo"]: l for l in pb.LLMS_DO_COMPARATIVO}
    meta = {"qwen": ("Alibaba", "abertos", "9,7 bi"), "deepseek": ("DeepSeek", "abertos", "1,6 tri"),
            "claude_haiku": ("Anthropic", "fechados", "n. d."), "gpt_luna": ("OpenAI", "fechados", "n. d."),
            "gpt_mini": ("OpenAI", "fechados", "n. d."), "claude_sonnet": ("Anthropic", "fechados", "n. d."),
            "gpt_sol": ("OpenAI", "fechados", "n. d.")}
    linhas = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{1.8pt}",
              "\\caption{LLMs avaliados. Preço em US\\$ por milhão de tokens (entrada/saída), consultado em outubro e novembro de "
              "2026. Temp. 0: o modelo aceita temperatura 0; os demais rodaram com a temperatura padrão da API e tiveram o Stuff "
              "repetido três vezes. n. d.: não divulgado.}", "\\label{tab:llms}",
              "\\begin{tabular}{llllcc}", "\\toprule",
              "Modelo & Empresa & Pesos & Parâm. & US\\$/1M & Temp. 0 \\\\", "\\midrule",
              "\\multicolumn{6}{l}{\\emph{Camada principal (econômicos): 100 perguntas por dataset}} \\\\"]
    for llm in TODOS:
        if llm == "claude_sonnet":
            linhas.append("\\multicolumn{6}{l}{\\emph{Camada de generalização (intermediários): 25 perguntas por dataset}} \\\\")
        e, s = info[llm]["preco"]
        emp, pesos, tam = meta[llm]
        linhas.append(f"{NOME_LLM_LONGO[llm]} & {emp} & {pesos} & {tam} & {num(e, 2)}/{num(s, 2)} & "
                      f"{'sim' if info[llm]['aceita_temperatura'] else 'não'} \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    escrever("tab_llms", linhas)


def tabela_porte(df: pd.DataFrame) -> pd.DataFrame:
    porte = ab.comparar_porte(df)
    linhas = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.6pt}",
              "\\caption{Modelo econômico com o seu melhor método (escolhido na camada principal) contra o intermediário da mesma "
              "família com o Stuff (Luna e Sol, da OpenAI; Haiku e Sonnet, da Anthropic), nas perguntas da camada de generalização (25; no Docugami, as 14 com valor no gabarito). "
              "$p$: McNemar exato (acurácia) ou Wilcoxon pareado. Custo: razão entre o custo médio por pergunta do intermediário "
              "e o do econômico.}", "\\label{tab:porte}",
              "\\begin{tabular}{llccc c}", "\\toprule",
              "Dataset & Econômico + método & Econ. & Interm. + Stuff & $p$ & Custo \\\\", "\\midrule"]
    for r in porte.itertuples():
        linhas.append(f"{NOME_DATASET[r.dataset].replace(' SEC 10-Q', '')} & {NOME_LLM[r.economico]} + {ab.NOME_METODO[r.melhor_metodo_economico]} "
                      f"& {num(r.media_economico_melhor)} & {num(r.media_intermediario_stuff)} & "
                      f"{p_tex(r.p)} & {num(r.custo_intermediario_stuff / r.custo_economico_melhor, 1)}× \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    escrever("tab_porte", linhas)
    return porte


def tabela_ancora(df: pd.DataFrame) -> pd.DataFrame:
    bge = pd.DataFrame([json.loads(t) for t in (METRICAS_DIR / "similaridade_bge_small.jsonl").read_text(encoding="utf-8").splitlines()])
    med = bge.groupby(["dataset", "metodo_rag"])["similaridade_bge_small"].median().unstack()
    linhas = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
              "\\caption{Mediana da similaridade de cosseno com a resposta de referência: Tabela 1 de Şakar e Emekci "
              "\\cite{sakar2025rag} e este trabalho, com o mesmo embedding (bge-small-en-v1.5) e todas as respostas dos 7 LLMs. "
              "Entre parênteses, a posição do método.}", "\\label{tab:ancora}",
              "\\begin{tabular}{lcccc}", "\\toprule",
              " & \\multicolumn{2}{c}{Docugami SEC 10-Q} & \\multicolumn{2}{c}{MedQA} \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Método & Artigo-base & Este trabalho & Artigo-base & Este trabalho \\\\", "\\midrule"]
    posicoes = {}
    for nome in ("docugami_10q", "medqa"):
        ref = pd.Series(ab.SIMILARIDADE_PAPER_ANCORA[nome])
        posicoes[nome, "ref"] = ref.rank(ascending=False).astype(int)
        posicoes[nome, "nos"] = med.loc[nome, ab.METODOS_RAG].rank(ascending=False, method="min").astype(int)
    for m in ab.METODOS_RAG:
        celulas = []
        for nome in ("docugami_10q", "medqa"):
            celulas += [f"{num(ab.SIMILARIDADE_PAPER_ANCORA[nome][m], 3)} ({posicoes[nome, 'ref'][m]}º)",
                        f"{num(med.loc[nome, m], 3)} ({posicoes[nome, 'nos'][m]}º)"]
        linhas.append(f"{ab.NOME_METODO[m]} & " + " & ".join(celulas) + " \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    escrever("tab_ancora", linhas)
    return med


def tabela_recuperacao() -> dict:
    resumo = json.loads((METRICAS_DIR / "recuperacao_benchmarks_resumo.json").read_text(encoding="utf-8"))
    bancos = {r["dataset"]: r for r in json.loads((METRICAS_DIR / "comparativo_vectorstores_benchmarks.json").read_text(encoding="utf-8"))}
    ctx = {}
    for nome in benchmarks.DATASETS:
        linhas_ctx = [json.loads(t)["ragas"] for t in (METRICAS_DIR / f"ragas_contexto_{nome}.jsonl").read_text(encoding="utf-8").splitlines()]
        ctx[nome] = (np.nanmean([l.get("context_precision", np.nan) for l in linhas_ctx]),
                     np.nanmean([l.get("context_recall", np.nan) for l in linhas_ctx]))
    linhas = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
              "\\caption{Recuperação ($k = 4$). Hit@4, Recall@4 e MRR@10 contra o trecho certo marcado no gabarito, em todas as "
              "perguntas; o MedQA não marca o trecho. Context Precision (CP) e Context Recall (CR) do RAGAS, uma vez por pergunta "
              "da amostra de 100.}", "\\label{tab:recuperacao}",
              "\\begin{tabular}{lccccc}", "\\toprule",
              "Dataset & Hit@4 & Recall@4 & MRR@10 & CP & CR \\\\", "\\midrule"]
    for nome in benchmarks.DATASETS:
        r = resumo.get(nome)
        rec = [num(r["hit@4"]), num(r["recall@4"]), num(r["mrr"])] if r else ["--"] * 3
        linhas.append(f"{NOME_DATASET[nome].replace(' SEC 10-Q', '')} & " + " & ".join(rec) +
                      f" & {num(ctx[nome][0])} & {num(ctx[nome][1])} \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    escrever("tab_recuperacao", linhas)
    return {"resumo": resumo, "ctx": ctx}


def tabela_ragas() -> pd.DataFrame:
    respostas = []
    for caminho in sorted(METRICAS_DIR.glob("ragas_*.jsonl")):
        if caminho.name.startswith("ragas_contexto_"):
            continue
        for texto in caminho.read_text(encoding="utf-8").splitlines():
            r = json.loads(texto)
            respostas.append({**{k: r[k] for k in ("dataset", "llm", "metodo_rag", "pergunta_id")}, **r["ragas"]})
    rag = pd.DataFrame(respostas)
    fiel = rag.pivot_table(index="metodo_rag", columns="dataset", values="faithfulness", aggfunc="mean")
    relev = rag.pivot_table(index="metodo_rag", columns="dataset", values="answer_relevancy", aggfunc="mean")
    linhas = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.4pt}",
              "\\caption{Métricas RAGAS por método (média das respostas dos 7 LLMs; juiz gpt-4o-mini). A \\emph{Faithfulness} só se "
              "aplica aos métodos com um conjunto único de trechos.}", "\\label{tab:ragas}",
              "\\begin{tabular}{lcccccc}", "\\toprule",
              " & \\multicolumn{3}{c}{\\emph{Faithfulness}} & \\multicolumn{3}{c}{\\emph{Answer Relevancy}} \\\\",
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
              "Método & Docug. & MedQA & FaQuAD & Docug. & MedQA & FaQuAD \\\\", "\\midrule"]
    for m in ab.METODOS:
        f = [num(fiel.loc[m, d]) if m in fiel.index and not np.isnan(fiel.loc[m, d]) else "--" for d in benchmarks.DATASETS]
        a = [num(relev.loc[m, d]) for d in benchmarks.DATASETS]
        linhas.append(f"{ab.NOME_METODO[m]} & " + " & ".join(f + a) + " \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    escrever("tab_ragas", linhas)
    return rag


def tabela_bancos():
    bancos = json.loads((METRICAS_DIR / "comparativo_vectorstores_benchmarks.json").read_text(encoding="utf-8"))
    stuff = {r["dataset"]: r for r in json.loads((METRICAS_DIR / "comparativo_stuff_faiss.json").read_text(encoding="utf-8"))}
    linhas = ["\\begin{table*}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
              "\\caption{ChromaDB (HNSW, busca aproximada) e FAISS (\\texttt{IndexFlatL2}, busca exata) sobre os mesmos vetores "
              "BGE-M3. Top-4 idêntico: consultas em que os dois bancos entregam os mesmos quatro fragmentos. Nas duas últimas "
              "colunas, a métrica principal do Stuff do gpt-4o-mini nas 100 perguntas, refazendo com o top-4 do FAISS as perguntas "
              "em que ele difere do índice dos experimentos.}", "\\label{tab:bancos}",
              "\\begin{tabular}{lrcccccccc}", "\\toprule",
              " & & \\multicolumn{2}{c}{Indexação (s)} & \\multicolumn{2}{c}{Consulta (ms)} & \\multicolumn{2}{c}{Memória (MB)} & "
              "& \\\\", "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
              "Dataset & Vetores & Chroma & FAISS & Chroma & FAISS & Chroma & FAISS & Top-4 idêntico & Stuff: Chroma × FAISS \\\\",
              "\\midrule"]
    for r in bancos:
        c, f, s = r["chromadb"], r["faiss"], stuff[r["dataset"]]
        linhas.append(f"{NOME_DATASET[r['dataset']]} & {num(r['n_vetores'], 0)} & {num(c['tempo_indexacao_s'], 2)} & "
                      f"{num(f['tempo_indexacao_s'], 3)} & {num(c['tempo_consulta_medio_ms'], 2)} & {num(f['tempo_consulta_medio_ms'], 2)} & "
                      f"{num(c['memoria_indexacao_mb'], 1)} & {num(f['memoria_indexacao_mb'], 1)} & "
                      f"{num(100 * r['concordancia_top4'], 0)}\\% & {num(s['media_chroma'])} × {num(s['media_faiss'])} \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    escrever("tab_bancos", linhas)
    return bancos, stuff


def _estilo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TEXTO_2)
        ax.spines[lado].set_linewidth(0.6)
    ax.tick_params(colors=TEXTO_2, labelsize=7.5, width=0.6, length=3)
    ax.xaxis.grid(True, color=GRADE, linewidth=0.6)
    ax.set_axisbelow(True)


def figura_efeito_rag(medias: dict):
    """Por dataset e LLM: sem RAG, Stuff e o melhor método de RAG daquele LLM (camada principal)."""
    fig, eixos = plt.subplots(1, 3, figsize=(7.1, 2.45), sharey=True)
    for ax, nome in zip(eixos, benchmarks.DATASETS):
        _estilo(ax)
        for y, llm in enumerate(PRINCIPAIS):
            s = medias[nome, llm]
            rag = s[ab.METODOS_RAG]
            melhor = rag.idxmax()
            ax.plot([s["sem_rag"], rag.max()], [y, y], color=GRADE, linewidth=2.2, zorder=1, solid_capstyle="round")
            ax.scatter(s["sem_rag"], y, s=34, facecolor="white", edgecolor=CINZA, linewidth=1.4, zorder=3)
            ax.scatter(s["stuff"], y, s=34, color=AZUL, edgecolor="white", linewidth=1.0, zorder=4)
            ax.scatter(rag.max(), y, s=34, color=LARANJA, edgecolor="white", linewidth=1.0, zorder=5)
            ax.annotate(ab.NOME_METODO[melhor], (rag.max(), y), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=TEXTO_2)
        ax.set_title(f"{NOME_DATASET[nome]}\n({METRICA_CURTA[nome]})", fontsize=8, color=TEXTO, pad=4)
        ax.set_xlim(0, 1.15 if nome != "docugami_10q" else 0.45)
        ax.set_ylim(-0.6, len(PRINCIPAIS) - 0.4)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    eixos[0].set_yticks(range(len(PRINCIPAIS)), [NOME_LLM[l] for l in PRINCIPAIS])
    eixos[0].invert_yaxis()
    marcadores = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=CINZA, markersize=5.5, label="Sem RAG"),
                  plt.Line2D([], [], marker="o", linestyle="", color=AZUL, markersize=5.5, label="Stuff"),
                  plt.Line2D([], [], marker="o", linestyle="", color=LARANJA, markersize=5.5, label="Melhor método de RAG do LLM")]
    fig.legend(handles=marcadores, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.2)
    fig.savefig(GRAFICOS / "efeito_rag.pdf", bbox_inches="tight")
    plt.close(fig)


def figura_ordem_metodos(resumo: dict):
    """Generalização: métrica principal nas 25 perguntas, 7 LLMs × 6 métodos; cor = posição do método
    dentro do LLM (mais escuro = melhor)."""
    cmap = LinearSegmentedColormap.from_list("seq_azul", SEQ_AZUL[::-1])
    fig, eixos = plt.subplots(1, 3, figsize=(7.1, 2.75), sharey=True)
    for ax, nome in zip(eixos, benchmarks.DATASETS):
        t = resumo[nome]["tabela7"].reindex(TODOS)
        postos = t.rank(axis=1, ascending=False, method="average")
        ax.imshow(postos.values, cmap=cmap, vmin=1, vmax=6, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                escuro = postos.values[i, j] <= 3
                ax.text(j, i, num(t.values[i, j]), ha="center", va="center", fontsize=6.3,
                        color="white" if escuro else TEXTO)
        ax.set_xticks(range(len(ab.METODOS_RAG)), [ab.NOME_METODO[m] for m in ab.METODOS_RAG], rotation=40, ha="right", fontsize=7)
        ax.set_xticks(np.arange(-0.5, len(ab.METODOS_RAG)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(TODOS)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="both", length=0, colors=TEXTO_2)
        ax.axhline(4.5, color=TEXTO, linewidth=0.9)
        for lado in ax.spines.values():
            lado.set_visible(False)
        ax.set_title(f"{NOME_DATASET[nome]}\nW = {num(resumo[nome]['w7'])}", fontsize=8, color=TEXTO, pad=4)
    eixos[0].set_yticks(range(len(TODOS)), [NOME_LLM[l] for l in TODOS], fontsize=7.5)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(GRAFICOS / "ordem_metodos.pdf", bbox_inches="tight")
    plt.close(fig)


def resumo_numeros(df, medias, sig, resumo, relativo, tokens, tempo, porte, med_bge, rag, bancos, stuff):
    """Números citados no texto do artigo, num JSON só (pra conferência)."""
    saida = {"execucoes": len(df), "custo_nominal_total_usd": round(float(df["custo_usd"].sum()), 2),
             "custo_nominal_por_llm_usd": df.groupby("llm")["custo_usd"].sum().round(2).to_dict()}
    saida["principal"] = {f"{n}|{l}": {m: round(float(v), 3) for m, v in medias[n, l].items()} for n, l in medias}
    saida["efeito_rag"] = {f"{n}|{l}": {m: [round(float(d), 3), round(float(p), 4)] for m, (d, p) in sig[n, l].items()} for n, l in sig}
    saida["concordancia"] = {n: {k: (round(v, 3) if isinstance(v, float) else {m: round(x, 2) for m, x in v.items()})
                                 for k, v in r.items() if k != "tabela7"} for n, r in resumo.items()}
    saida["generalizacao"] = {n: r["tabela7"].round(3).to_dict(orient="index") for n, r in resumo.items()}
    saida["custo_relativo_mediana"] = relativo.median().round(2).to_dict()
    saida["tokens_media"] = tokens.round(0).to_dict()
    saida["tempo_mediana_s"] = tempo.round(1).to_dict()
    saida["porte"] = json.loads(porte.to_json(orient="records"))
    saida["bge_small_mediana"] = med_bge.round(3).to_dict(orient="index")
    recusa = df[(df.repeticao == 1)].groupby(["dataset", "llm", "metodo_rag"])["recusa"].mean().round(3)
    saida["recusa"] = {f"{d}|{l}|{m}": float(v) for (d, l, m), v in recusa.items()}
    rep = df[df.metodo_rag == "stuff"]
    dps = {}
    for (nome, llm), h in rep.groupby(["dataset", "llm"]):
        col = ab.METRICA_PRINCIPAL[nome][0]
        dp = h.groupby("pergunta_id")[col].std().dropna()
        if h["repeticao"].nunique() > 1 and len(dp):
            dps[f"{nome}|{llm}"] = round(float(dp.mean()), 3)
    saida["desvio_repeticoes"] = dps
    saida["ragas_faithfulness"] = rag.pivot_table(index=["dataset", "llm"], columns="metodo_rag", values="faithfulness").round(3).to_json()
    (TABELAS / "numeros_artigo.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    TABELAS.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "pdf.fonttype": 42})
    df = ab.carregar_execucoes()
    medias, sig = tabela_principal(df)
    resumo = tabela_concordancia(df)
    relativo, tokens, tempo = tabela_custo(df)
    tabela_llms(df)
    porte = tabela_porte(df)
    med_bge = tabela_ancora(df)
    tabela_recuperacao()
    rag = tabela_ragas()
    bancos, stuff = tabela_bancos()
    figura_efeito_rag(medias)
    figura_ordem_metodos(resumo)
    resumo_numeros(df, medias, sig, resumo, relativo, tokens, tempo, porte, med_bge, rag, bancos, stuff)
    print(f"Escritos {TABELAS.relative_to(BASE_DIR)} e {GRAFICOS.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
