"""
Tabelas (LaTeX) e figuras (PDF) do artigo do TCC02 a partir da rodada dos benchmarks, com os mesmos
dados e cálculos de analisar_benchmarks.py. No fim, copia as tabelas e as figuras para dentro do
artigo (Artigo_TCC02/tabelas e Artigo_TCC02/figuras), para que a pasta do artigo compile sozinha,
inclusive no Overleaf: rodar este script de novo atualiza o artigo.

Saídas: resultados/tabelas_benchmarks/*.tex e resultados/graficos_benchmarks/*.pdf, e as cópias em
Artigo_TCC02/tabelas/*.tex e Artigo_TCC02/figuras/*.pdf.
Uso: python gerar_artigo_benchmarks.py
"""

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy import stats
from scipy.optimize import nnls

import analisar_benchmarks as ab
import diagramas_rag
import benchmarks
import pipeline_base as pb
from analisar_resultados import BASE_DIR, METRICAS_DIR, friedman_com_posthoc, holm

TABELAS = BASE_DIR / "resultados" / "tabelas_benchmarks"
GRAFICOS = BASE_DIR / "resultados" / "graficos_benchmarks"
ARTIGO = BASE_DIR.parent / "Artigo_TCC02"

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


def empatados_no_maior(serie: pd.Series, casas: int = 2) -> list:
    """Os itens cujo valor, escrito como no artigo (`casas` decimais), é igual ao maior. O negrito das tabelas e os
    rótulos das figuras usam este critério, para nunca distinguir dois valores que o leitor vê iguais."""
    maior = num(serie.max(), casas)
    return [k for k, v in serie.items() if num(v, casas) == maior]


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


INTERMEDIARIOS = ["claude_sonnet", "gpt_sol"]
NOTA_INTERMEDIARIOS = "$^\\dagger$ Intermediários: nas 25 perguntas da camada de generalização."


def cabecalho_7llms(rotulo_dataset) -> list[str]:
    """Abertura do tabular com os sete LLMs por dataset: os cinco econômicos e, separados, os dois intermediários.
    Com 21 colunas, o DeepSeek vai em duas linhas no cabeçalho para a tabela caber na largura da página."""
    bloco = "ccccc@{\\hspace{3pt}}cc"
    cab = " & ".join(f"\\multicolumn{{7}}{{c}}{{{rotulo_dataset(n)}}}" for n in benchmarks.DATASETS)
    nomes = {**NOME_LLM, "deepseek": "\\shortstack{Deep\\\\Seek}"}
    sub = " & ".join(" & ".join(nomes[l] + ("$^\\dagger$" if l in INTERMEDIARIOS else "") for l in TODOS)
                     for _ in benchmarks.DATASETS)
    return ["\\begin{tabular}{l" + bloco.join(["", "@{\\hspace{4pt}}", "@{\\hspace{4pt}}", ""]) + "}",
            "\\toprule", f" & {cab} \\\\",
            "\\cmidrule(lr){2-8}\\cmidrule(lr){9-15}\\cmidrule(lr){16-22}", f"Método & {sub} \\\\", "\\midrule"]


def tabela_principal(df: pd.DataFrame):
    medias, sig, friedman = {}, {}, {}
    for nome in benchmarks.DATASETS:
        col = ab.METRICA_PRINCIPAL[nome][0]
        for llm in TODOS:
            g = df[(df.dataset == nome) & (df.llm == llm) & (df.repeticao == 1)]
            medias[nome, llm] = g.groupby("metodo_rag")[col].mean()
            sig[nome, llm] = efeito_rag(g, col)
            friedman[nome, llm] = friedman_com_posthoc(g, col, ab.METODOS_RAG).get("p", np.nan)
    linhas = [
        "\\begin{table*}[!t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{1.5pt}",
        "\\caption{Métrica principal de cada dataset por método e LLM. Econômicos: camada principal (100 perguntas por dataset). "
        + NOTA_INTERMEDIARIOS + " Em negrito, o melhor valor de cada "
        "coluna (mais de um quando empatam na segunda casa decimal). * diferença em relação ao controle sem RAG com $p_{\\text{Holm}} < 0{,}05$ (Wilcoxon pareado). Na última "
        "linha, o valor-$p$ do teste de Friedman entre os seis métodos de RAG (--: o mesmo resultado com todos os métodos em "
        "todas as perguntas, sem teste possível).}",
        "\\label{tab:principal}",
    ] + cabecalho_7llms(lambda n: f"{NOME_DATASET[n]} ({METRICA_CURTA[n]})")
    for m in ab.METODOS:
        celulas = []
        for nome in benchmarks.DATASETS:
            for llm in TODOS:
                serie = medias[nome, llm]
                v = serie[m]
                texto = num(v)
                if m != "sem_rag" and sig[nome, llm][m][1] < 0.05:
                    texto += "*"
                if m in empatados_no_maior(serie):
                    texto = f"\\textbf{{{texto}}}"
                celulas.append(texto)
        linhas.append(f"{ab.NOME_METODO[m]} & " + " & ".join(celulas) + " \\\\")
        if m == "sem_rag":
            linhas.append("\\midrule")
    linhas += ["\\midrule", "Friedman $p$ & " + " & ".join(p_tex(friedman[n, l]).replace("$<$", "\\textless{}")  # mais estreito
                                                for n in benchmarks.DATASETS for l in TODOS) + " \\\\",
               "\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    escrever("tab_principal", linhas)
    return medias, sig


def tabela_tokens(df: pd.DataFrame):
    """Mesmo recorte da Tabela de resultados principal, com os
    tokens consumidos por pergunta no lugar da métrica: soma da entrada e da saída de todas as chamadas ao LLM feitas
    pelo método, em média por pergunta."""
    medias = {}
    for nome in benchmarks.DATASETS:
        col = ab.METRICA_PRINCIPAL[nome][0]
        for llm in TODOS:
            g = df[(df.dataset == nome) & (df.llm == llm) & (df.repeticao == 1) & df[col].notna()]
            medias[nome, llm] = g.groupby("metodo_rag")["tokens_totais"].mean()
    linhas = [
        "\\begin{table*}[!t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{1.5pt}",
        "\\caption{Tokens consumidos por pergunta, por método e LLM, nas mesmas perguntas da Tabela~\\ref{tab:principal} "
        "(econômicos: 100 perguntas por dataset). " + NOTA_INTERMEDIARIOS +
        " Cada valor é a média, por pergunta, da soma dos tokens de entrada e de saída de todas as chamadas ao LLM feitas pelo "
        "método. Os tokens são contados pelo tokenizador de cada modelo, de modo que a comparação entre LLMs reflete também a "
        "diferença entre tokenizadores.}",
        "\\label{tab:tokens}",
    ] + cabecalho_7llms(lambda n: NOME_DATASET[n])
    for m in ab.METODOS:
        celulas = [num(medias[nome, llm][m], 0) for nome in benchmarks.DATASETS for llm in TODOS]
        linhas.append(f"{ab.NOME_METODO[m]} & " + " & ".join(celulas) + " \\\\")
        if m == "sem_rag":
            linhas.append("\\midrule")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    escrever("tab_tokens", linhas)
    return medias


def tabela_custo_usd(df: pd.DataFrame):
    """Mesmo recorte da Tabela de tokens, com o custo em dólares no lugar dos tokens: US$ por 1.000 perguntas, somando
    todas as chamadas ao LLM feitas pelo método (tokens de entrada e de saída vezes o preço de tabela de cada um)."""
    medias = {}
    for nome in benchmarks.DATASETS:
        col = ab.METRICA_PRINCIPAL[nome][0]
        for llm in TODOS:
            g = df[(df.dataset == nome) & (df.llm == llm) & (df.repeticao == 1) & df[col].notna()]
            medias[nome, llm] = 1000 * g.groupby("metodo_rag")["custo_usd"].mean()
    linhas = [
        "\\begin{table*}[!t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{1.5pt}",
        "\\caption{Custo em US\\$ por 1.000 perguntas, por método e LLM, nas mesmas perguntas da Tabela~\\ref{tab:principal} "
        "(econômicos: 100 perguntas por dataset). " + NOTA_INTERMEDIARIOS +
        " Cada valor soma todas as chamadas ao LLM feitas pelo método: tokens de entrada e de saída vezes o preço de tabela de "
        "cada um (Tabela~\\ref{tab:llms}). Custo nominal, sem descontos de cache de prompt.}",
        "\\label{tab:custo_usd}",
    ] + cabecalho_7llms(lambda n: NOME_DATASET[n])
    for m in ab.METODOS:
        celulas = [num(medias[nome, llm][m]) for nome in benchmarks.DATASETS for llm in TODOS]
        linhas.append(f"{ab.NOME_METODO[m]} & " + " & ".join(celulas) + " \\\\")
        if m == "sem_rag":
            linhas.append("\\midrule")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    escrever("tab_custo_usd", linhas)
    return medias


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
              "dos seis métodos entre os LLMs (W de Kendall, com correção de empates: 1 = mesma ordem em todos; 0 = nenhuma concordância; entre parênteses, "
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
    # Da recuperação ao fim da geração. O tempo_total_s do log não serve: ele é medido depois do cálculo da similaridade
    # de cosseno com a referência (rodar_experimento), que é avaliação, não parte do RAG, e somava ~2 s à mediana.
    tempo = (g["tempo_recuperacao_s"] + g["tempo_geracao_s"]).groupby(g["metodo_rag"]).median()
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
              "executado três vezes nas 25 perguntas da camada de generalização. n. d.: não divulgado.}", "\\label{tab:llms}",
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
    """Econômico + melhor método × intermediário + Stuff, com o preço por token, os tokens gastos e o
    custo de cada configuração. A razão de custo é calculada com os custos sem arredondamento, por isso pode diferir um
    pouco da divisão dos custos impressos."""
    porte = ab.comparar_porte(df)
    precos = {l["rotulo"]: l["preco"] for l in pb.LLMS_DO_COMPARATIVO}
    linhas = ["\\begin{table*}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
              "\\caption{Modelo econômico com o seu melhor método (escolhido nas 100 perguntas da camada principal; em empate, o mais "
              "barato) contra o "
              "intermediário da mesma família com o Stuff, nas perguntas da camada de generalização (25). Preço: US\\$ por milhão de tokens de entrada e de saída (Tabela~\\ref{tab:llms}). Tokens: média "
              "por pergunta, somando a entrada e a saída de todas as chamadas do método, contados pelo tokenizador de cada modelo. "
              "Custo: US\\$ por 1.000 perguntas (tokens de entrada e de saída vezes o preço de cada um). Métrica: a principal de cada "
              "conjunto. $p$: McNemar exato (acurácia) ou Wilcoxon pareado entre as duas configurações. Razão: custo do "
              "intermediário dividido pelo do econômico, calculada antes do arredondamento.}", "\\label{tab:porte}",
              "\\begin{tabular}{lllcrrccc}", "\\toprule",
              "Dataset & Família & Configuração & Preço (US\\$/1M) & Tokens & Custo (US\\$/1.000) & Métrica & $p$ & Razão \\\\",
              "\\midrule"]
    anterior = None
    for r in porte.itertuples():
        if anterior is not None:
            linhas.append("\\midrule" if r.dataset != anterior else "\\addlinespace")
        conjunto = NOME_DATASET[r.dataset].replace(" SEC 10-Q", "") if r.dataset != anterior else ""
        anterior = r.dataset
        for k, (llm, metodo, media, tokens, custo) in enumerate([
                (r.economico, ab.NOME_METODO[r.melhor_metodo_economico], r.media_economico_melhor,
                 r.tokens_economico_melhor, r.custo_economico_melhor),
                (r.intermediario, "Stuff", r.media_intermediario_stuff, r.tokens_intermediario_stuff, r.custo_intermediario_stuff)]):
            entrada, saida = precos[llm]
            linhas.append(" & ".join([
                conjunto if k == 0 else "", r.familia if k == 0 else "", f"{NOME_LLM[llm]} + {metodo}",
                f"{num(entrada)} / {num(saida)}", num(tokens, 0), num(1000 * custo), num(media),
                p_tex(r.p) if k == 1 else "",
                f"{num(r.custo_intermediario_stuff / r.custo_economico_melhor, 1)}×" if k == 1 else ""]) + " \\\\")
    linhas += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    escrever("tab_porte", linhas)
    return porte


def tabela_ancora(df: pd.DataFrame) -> pd.DataFrame:
    bge = pd.DataFrame([r for r in map(json.loads, (METRICAS_DIR / "similaridade_bge_small.jsonl").read_text(encoding="utf-8").splitlines())
                        if ab.na_amostra(r)])
    med = bge.groupby(["dataset", "metodo_rag"])["similaridade_bge_small"].median().unstack()
    linhas = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
              "\\caption{Mediana da similaridade de cosseno com a resposta de referência: Tabela 1 de Şakar e Emekci "
              "\\cite{sakar2025rag} e este trabalho, com o bge-small-en-v1.5, um dos três embeddings daquele estudo, e todas as respostas dos 7 LLMs. "
              "Entre parênteses, a posição do método (valores iguais na terceira casa decimal dividem a posição).}", "\\label{tab:ancora}",
              "\\begin{tabular}{lcccc}", "\\toprule",
              " & \\multicolumn{2}{c}{Docugami SEC 10-Q} & \\multicolumn{2}{c}{MedQA} \\\\",
              "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", "Método & Artigo-base & Este trabalho & Artigo-base & Este trabalho \\\\", "\\midrule"]
    posicoes = {}
    for nome in ("docugami_10q", "medqa"):
        ref = pd.Series(ab.SIMILARIDADE_PAPER_ANCORA[nome])
        posicoes[nome, "ref"] = ref.rank(ascending=False).astype(int)
        impressos = med.loc[nome, ab.METODOS_RAG].map(lambda v: float(num(v, 3).replace(",", ".")))
        posicoes[nome, "nos"] = impressos.rank(ascending=False, method="min").astype(int)  # iguais na 3ª casa dividem a posição
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
        linhas_ctx = [r["ragas"] for r in map(json.loads, (METRICAS_DIR / f"ragas_contexto_{nome}.jsonl").read_text(encoding="utf-8").splitlines())
                      if ab.na_amostra(r)]
        ctx[nome] = (np.nanmean([l.get("context_precision", np.nan) for l in linhas_ctx]),
                     np.nanmean([l.get("context_recall", np.nan) for l in linhas_ctx]))
    linhas = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
              "\\caption{Recuperação ($k = 4$). Hit@4, Recall@4 e MRR@10 contra a origem da resposta marcada no gabarito (o relatório, no Docugami; o "
              "parágrafo, no FaQuAD), em todas as perguntas que a marcam (193 das 195 no Docugami); o MedQA não a marca. Context Precision (CP) e Context Recall (CR) do RAGAS, uma vez por pergunta "
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
            if ab.na_amostra(r):
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
              "BGE-M3. Top-4 idêntico: consultas em que os dois bancos entregam os mesmos quatro fragmentos. Na última "
              "coluna, a métrica principal do Stuff do gpt-4o-mini nas 100 perguntas, refazendo com o top-4 do FAISS as perguntas "
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
    """Por dataset e LLM: sem RAG, Stuff e o melhor método de RAG daquele LLM. Econômicos (círculos):
    camada principal. Intermediários (quadrados): as 25 perguntas da generalização."""
    fig, eixos = plt.subplots(1, 3, figsize=(7.1, 3.15), sharey=True)
    for ax, nome in zip(eixos, benchmarks.DATASETS):
        _estilo(ax)
        for y, llm in enumerate(TODOS):
            s = medias[nome, llm]
            rag = s[ab.METODOS_RAG]
            # Com empate no melhor valor na segunda casa decimal (comum na acurácia do MedQA), o rótulo diz quantos métodos
            # empataram, em vez de nomear só um deles.
            empatados = empatados_no_maior(rag)
            rotulo = ab.NOME_METODO[empatados[0]] if len(empatados) == 1 else f"{len(empatados)} empatados"
            forma, tamanho = ("s", 32) if llm in INTERMEDIARIOS else ("o", 34)
            ax.plot([s["sem_rag"], rag.max()], [y, y], color=GRADE, linewidth=2.2, zorder=1, solid_capstyle="round")
            ax.scatter(s["sem_rag"], y, s=tamanho, marker=forma, facecolor="white", edgecolor=CINZA, linewidth=1.4, zorder=3)
            ax.scatter(s["stuff"], y, s=tamanho, marker=forma, color=AZUL, edgecolor="white", linewidth=1.0, zorder=4)
            ax.scatter(rag.max(), y, s=tamanho, marker=forma, color=LARANJA, edgecolor="white", linewidth=1.0, zorder=5)
            ax.annotate(rotulo, (rag.max(), y), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=6.5, color=TEXTO_2)
        ax.axhline(len(PRINCIPAIS) - 0.5, color=TEXTO, linewidth=0.7, zorder=0)
        ax.set_title(f"{NOME_DATASET[nome]}\n({METRICA_CURTA[nome]})", fontsize=8, color=TEXTO, pad=4)
        ax.set_xlim(0, 1.15 if nome != "docugami_10q" else 0.45)
        ax.set_ylim(-0.6, len(TODOS) - 0.4)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    eixos[0].set_yticks(range(len(TODOS)),
                        [NOME_LLM[l] + ("$^\\dagger$" if l in INTERMEDIARIOS else "") for l in TODOS])
    eixos[0].invert_yaxis()
    marcadores = [plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=CINZA, markersize=5.5, label="Sem RAG"),
                  plt.Line2D([], [], marker="o", linestyle="", color=AZUL, markersize=5.5, label="Stuff"),
                  plt.Line2D([], [], marker="o", linestyle="", color=LARANJA, markersize=5.5, label="Melhor método de RAG do LLM"),
                  plt.Line2D([], [], marker="s", linestyle="", color=CINZA, markersize=5.0, label="Intermediário$^\\dagger$")]
    fig.legend(handles=marcadores, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    fig.savefig(GRAFICOS / "efeito_rag.pdf", bbox_inches="tight")
    plt.close(fig)


# Figura de custo × desempenho com todos os métodos: a cor identifica o método, e a sigla dentro do marcador, o LLM.
# As cores vêm da paleta categórica de referência (Stuff em azul, como na Fig. 1), atribuídas de modo que os métodos
# de custo parecido, cujos pontos ficam lado a lado, tenham as cores mais distantes entre si; o controle sem RAG é
# vazado, em cinza. A sigla usa branco ou quase preto, o de maior contraste com o preenchimento.
COR_METODO = {"stuff": "#2a78d6", "refine": "#eda100", "map_reduce": "#e87ba4", "map_rerank": "#008300",
              "query_step_down": "#6250d6", "reciprocal": "#1baf7a"}
TINTA_METODO = {"stuff": "white", "refine": TEXTO, "map_reduce": TEXTO, "map_rerank": "white",
                "query_step_down": "white", "reciprocal": TEXTO}
SIGLA_LLM = {"qwen": "Q", "deepseek": "D", "claude_haiku": "H", "gpt_luna": "L", "gpt_mini": "M",
             "claude_sonnet": "Sn", "gpt_sol": "Sl"}
# Limites e intervalo das marcas do eixo y de cada painel (as métricas têm escalas diferentes).
EIXO_Y_CUSTO = {"docugami_10q": (-0.02, 0.38, 0.1), "medqa": (0.56, 1.0, 0.1), "faquad": (0.3, 1.0, 0.1)}
DIAMETRO_MARCADOR = 10.0  # pt


FOLGA_MARCADORES = 0.8  # pt entre dois marcadores vizinhos


def _menor_distancia(G: np.ndarray, h: np.ndarray) -> np.ndarray:
    """O menor vetor x com G x >= h (least distance programming), resolvido de forma exata por mínimos quadrados não
    negativos (Lawson e Hanson, Solving Least Squares Problems, cap. 23)."""
    m, n = G.shape
    E = np.vstack([G.T, h[None, :]])
    f = np.zeros(n + 1)
    f[-1] = 1.0
    u, _ = nnls(E, f, maxiter=50 * m)
    r = E @ u - f
    if np.linalg.norm(r) < 1e-10:
        raise RuntimeError("afastar_marcadores: restrições incompatíveis")
    return -r[:n] / r[n]


def afastar_marcadores(ancoras: np.ndarray, raios: np.ndarray, caixa, max_rodadas: int = 40) -> np.ndarray:
    """Posições (em pontos) em que os marcadores não se sobrepõem, o mais perto possível das posições reais (menor soma
    dos quadrados dos deslocamentos), sem nunca inverter a ordem dos dados: um valor maior em x (ou em y) fica à
    direita (ou acima) do menor, ou no máximo alinhado com ele, e valores iguais continuam alinhados. Cada par que se
    sobreporia é afastado ao longo da reta que liga as duas posições reais, o que por si só mantém a ordem dos dois
    nos dois eixos. As restrições de afastamento entram em rodadas, até nenhum par se sobrepor.

    A versão anterior (empurrar os marcadores sobrepostos e puxá-los de volta, sem restrição de ordem) chegava a
    desenhar um valor maior abaixo de um menor quando os dois estavam a menos de um diâmetro de distância."""
    n = len(ancoras)
    a = ancoras.astype(float).copy()
    # Dois pontos idênticos nos dois eixos não teriam direção de afastamento: desempata em x, pela ordem de chegada.
    for i in range(n):
        for j in range(i + 1, n):
            if np.allclose(a[i], a[j], rtol=0, atol=1e-6):
                a[j, 0] += 1e-3
    fixas = []  # (linha, limite) de G p >= h
    for eixo in (0, 1):  # ordem: só entre vizinhos na ordenação; o resto vem por transitividade
        ordem = np.argsort(a[:, eixo], kind="stable")
        for i, j in zip(ordem[:-1], ordem[1:]):
            linha = np.zeros(2 * n)
            linha[2 * j + eixo], linha[2 * i + eixo] = 1.0, -1.0
            fixas.append((linha, 0.0))
            if np.isclose(a[i, eixo], a[j, eixo], rtol=0, atol=1e-6):
                fixas.append((-linha, 0.0))  # empate: as duas desigualdades fazem a igualdade
    (x0, y0), (x1, y1) = caixa
    for k, r in enumerate(raios):  # dentro da área do gráfico
        for eixo, (lo, hi) in enumerate(((x0 + r, x1 - r), (y0 + r, y1 - r))):
            linha = np.zeros(2 * n)
            linha[2 * k + eixo] = 1.0
            fixas += [(linha, lo), (-linha, -hi)]
    minimo = raios[:, None] + raios[None, :] + FOLGA_MARCADORES

    def sobrepostos(p):
        d = np.hypot(*(p[:, None, :] - p[None, :, :]).transpose(2, 0, 1))
        return {(i, j) for i, j in zip(*np.nonzero(d < minimo - 1e-3)) if i < j}

    pares, pos, alvo = set(), a.copy(), a.ravel()
    for _ in range(max_rodadas):
        novos = sobrepostos(pos) - pares
        if not novos:
            break
        pares |= novos
        linhas = list(fixas)
        for i, j in sorted(pares):
            u = (a[j] - a[i]) / np.hypot(*(a[j] - a[i]))
            linha = np.zeros(2 * n)
            linha[2 * j:2 * j + 2], linha[2 * i:2 * i + 2] = u, -u
            linhas.append((linha, minimo[i, j]))
        G, h = np.array([l for l, _ in linhas]), np.array([v for _, v in linhas])
        pos = (alvo + _menor_distancia(G, h - G @ alvo)).reshape(n, 2)
    if sobrepostos(pos):
        raise RuntimeError("afastar_marcadores: ainda há marcadores sobrepostos")
    parados = np.hypot(*(pos - ancoras).T) < 1e-3  # ruído numérico
    pos[parados] = ancoras[parados]
    return pos


def _marcador(ax, x, y, metodo: str | None, llm: str, zorder: float = 3):
    """Marcador com a sigla do LLM: círculo para os econômicos e quadrado para os intermediários, preenchido com a
    cor do método (vazado no controle sem RAG)."""
    intermediario = llm in INTERMEDIARIOS
    forma, tamanho = ("s", (0.93 * DIAMETRO_MARCADOR) ** 2) if intermediario else ("o", DIAMETRO_MARCADOR ** 2)
    cor = COR_METODO.get(metodo)
    if cor is None:
        ax.scatter(x, y, s=tamanho, marker=forma, facecolor="white", edgecolor=CINZA, linewidth=0.9, zorder=zorder)
        tinta = TEXTO_2 if metodo == "sem_rag" else TEXTO
    else:
        ax.scatter(x, y, s=tamanho, marker=forma, facecolor=cor, edgecolor="white", linewidth=0.7, zorder=zorder)
        tinta = TINTA_METODO[metodo]
    ax.text(x, y, SIGLA_LLM[llm], ha="center", va="center_baseline", fontsize=5.3 if intermediario else 5.8,
            fontweight="bold", color=tinta, zorder=zorder + 1)


def _linha_de_legenda(ax, y: float, titulo: str, itens: list):
    """Uma linha da legenda no topo da figura: o título e, em sequência, cada marcador seguido do seu nome. As
    posições são medidas no texto renderizado, para os nomes não encostarem no marcador seguinte."""
    renderizador = ax.figure.canvas.get_renderer()
    largura_eixo = ax.get_window_extent(renderizador).width
    raio = DIAMETRO_MARCADOR / 2 * ax.figure.dpi / 72 / largura_eixo
    t = ax.text(0, y, titulo, fontsize=7.5, color=TEXTO, va="center", fontweight="bold")
    x = 0.105
    for desenhar, rotulo in itens:
        desenhar(x + raio, y)
        t = ax.text(x + 2 * raio + 0.006, y, rotulo, fontsize=7.5, color=TEXTO, va="center")
        x += 2 * raio + 0.006 + t.get_window_extent(renderizador).width / largura_eixo + 0.03


def _eixo_y_metrica(ax, nome: str):
    y0, y1, passo = EIXO_Y_CUSTO[nome]
    ax.set_ylim(y0, y1)
    ax.yaxis.set_major_locator(plt.MultipleLocator(passo))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_title(f"{NOME_DATASET[nome]} ({METRICA_CURTA[nome]})", fontsize=8, color=TEXTO, loc="left", pad=4)


def _eixo_y_tokens(ax, nome: str):
    """Tokens por pergunta em escala log, com a mesma faixa nos três painéis (de 80 a 10.063 na Tabela de tokens)."""
    ax.set_yscale("log")
    ax.set_ylim(55, 16000)
    ax.set_yticks([100, 300, 1000, 3000, 10000])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", ".")))
    ax.set_title(NOME_DATASET[nome], fontsize=8, color=TEXTO, loc="left", pad=4)


def figura_custo_desempenho(medias: dict, custo_usd: dict):
    """Todos os métodos e LLMs: custo em US$ por 1.000 perguntas (escala log) × métrica principal, um painel por
    dataset, com os mesmos valores das Tabelas de resultado principal e de custo em dólares."""
    _figura_contra_custo(medias, custo_usd, _eixo_y_metrica, None, "custo_desempenho.pdf")


def figura_custo_tokens(tokens: dict, custo_usd: dict):
    """Igual à figura de custo × desempenho, com os tokens por pergunta (Tabela de tokens) no eixo y, em escala log,
    no lugar da métrica principal. Num mesmo LLM, os pontos ficam numa reta (o preço por token é quase fixo); a
    distância vertical entre LLMs com o mesmo custo mostra a diferença de preço do token."""
    _figura_contra_custo(tokens, custo_usd, _eixo_y_tokens, "Tokens por pergunta (escala logarítmica)",
                         "custo_tokens.pdf")


def _figura_contra_custo(valores_y: dict, custo_usd: dict, configurar_y, rotulo_y: str | None, arquivo,
                         conjuntos=None, figsize=(7.1, 7.5), ajuste=None, legenda_ret=None, dpi=None):
    """Um painel por dataset, com custo em US$ por 1.000 perguntas (escala log) no eixo x e `valores_y` no y. Onde
    dois marcadores se sobreporiam, um deles é deslocado o mínimo necessário, e uma linha fina o liga à posição real
    (um ponto). Os parâmetros opcionais (só alguns conjuntos, tamanho, margens, posição da legenda, resolução) servem
    para as versões dos slides; sem eles, sai a figura do artigo."""
    medias = valores_y
    conjuntos = conjuntos or benchmarks.DATASETS
    fig, eixos = plt.subplots(len(conjuntos), 1, figsize=figsize, squeeze=False)
    eixos = eixos[:, 0]
    esquerda = 0.105 if rotulo_y else 0.07  # espaço para o rótulo do eixo y e para as marcas "10.000"
    ajuste = ajuste or dict(left=esquerda, right=0.99, top=0.885, bottom=0.065, hspace=0.42)
    fig.subplots_adjust(**ajuste)
    esquerda = ajuste["left"]
    for ax, nome in zip(eixos, conjuntos):
        _estilo(ax)
        ax.yaxis.grid(True, color=GRADE, linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlim(0.015, 50)
        ax.set_xticks([0.03, 0.1, 0.3, 1, 3, 10, 30])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
        configurar_y(ax, nome)
        if rotulo_y:
            ax.set_ylabel(rotulo_y if ax is eixos[len(eixos) // 2] else "", fontsize=7.5, color=TEXTO_2)
    eixos[-1].set_xlabel("Custo (US\\$ por 1.000 perguntas, escala logarítmica)", fontsize=7.5, color=TEXTO_2)

    escala = 72 / fig.dpi  # pixels -> pontos
    for ax, nome in zip(eixos, conjuntos):
        pontos = [(m, llm, custo_usd[nome, llm][m], medias[nome, llm][m]) for m in ab.METODOS for llm in TODOS]
        dados = np.array([(x, y) for _, _, x, y in pontos])
        # Nenhum ponto pode cair fora dos eixos: o marcador seria empurrado para dentro e a posição real sumiria.
        (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
        fora = [(m, llm, x, y) for m, llm, x, y in pontos if not (xa < x < xb and ya < y < yb)]
        if fora:
            raise ValueError(f"{nome}: pontos fora dos limites do gráfico: {fora}")
        ancoras = ax.transData.transform(dados) * escala
        bb = ax.get_window_extent()
        caixa = ((bb.x0 * escala, bb.y0 * escala), (bb.x1 * escala, bb.y1 * escala))
        raios = np.array([DIAMETRO_MARCADOR / 2 + (0.6 if llm in INTERMEDIARIOS else 0) for _, llm, _, _ in pontos])
        afastadas = afastar_marcadores(ancoras, raios, caixa)
        no_lugar = (afastadas == ancoras).all(axis=1)
        posicoes = np.where(no_lugar[:, None], dados, ax.transData.inverted().transform(afastadas / escala))
        deslocados = np.hypot(*(afastadas - ancoras).T) > 0.05  # a linha até a posição real só aparece se houver o que ligar
        for (m, llm, x, y), (xm, ym), deslocado in zip(pontos, posicoes, deslocados):
            if deslocado:
                ax.plot([x, xm], [y, ym], color="#a9a8a2", linewidth=0.5, zorder=2, solid_capstyle="round")
                ax.scatter(x, y, s=5, color=COR_METODO.get(m, CINZA), linewidth=0, zorder=2.5)
            _marcador(ax, xm, ym, m, llm)

    legenda = fig.add_axes(legenda_ret or [esquerda, 0.915, 0.99 - esquerda, 0.08])
    legenda.axis("off")
    legenda.set_xlim(0, 1)
    legenda.set_ylim(0, 1)
    def metodo(m):
        return lambda x, y: legenda.scatter(x, y, s=DIAMETRO_MARCADOR ** 2, facecolor=COR_METODO.get(m, "white"),
                                            edgecolor=CINZA if m == "sem_rag" else "white",
                                            linewidth=0.9 if m == "sem_rag" else 0.7)
    _linha_de_legenda(legenda, 0.78, "Método", [(metodo(m), ab.NOME_METODO[m]) for m in ab.METODOS])
    _linha_de_legenda(legenda, 0.22, "Modelo", [(lambda x, y, l=llm: _marcador(legenda, x, y, None, l),
                                                 NOME_LLM[llm] + ("$^\\dagger$" if llm in INTERMEDIARIOS else ""))
                                                for llm in TODOS])
    fig.savefig(GRAFICOS / arquivo, **({"dpi": dpi} if dpi else {}))
    plt.close(fig)


def figura_ordem_metodos(resumo: dict, caminhos=None, casas: int = 2):
    """Generalização: métrica principal nas 25 perguntas, 7 LLMs × 6 métodos; cor = posição do método
    dentro do LLM (mais escuro = melhor). `caminhos` e `casas` servem para reaproveitar o desenho com
    outra métrica (figura_ordem_metodos_cosseno.py); sem eles, grava a figura do artigo."""
    cmap = LinearSegmentedColormap.from_list("seq_azul", SEQ_AZUL[::-1])
    fig, eixos = plt.subplots(1, 3, figsize=(7.1, 2.75), sharey=True)
    for ax, nome in zip(eixos, benchmarks.DATASETS):
        t = resumo[nome]["tabela7"].reindex(TODOS)
        # A cor vem da posição pelo valor impresso na célula: dois valores que o leitor vê iguais têm a mesma cor. O W do
        # título continua calculado com os valores exatos.
        impressos = t.apply(lambda coluna: coluna.map(lambda v: float(f"{v:.{casas}f}")))
        postos = impressos.rank(axis=1, ascending=False, method="average")
        ax.imshow(postos.values, cmap=cmap, vmin=1, vmax=6, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                escuro = postos.values[i, j] <= 3
                ax.text(j, i, num(t.values[i, j], casas), ha="center", va="center", fontsize=6.3,
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
    for caminho in caminhos or [GRAFICOS / "ordem_metodos.pdf"]:
        fig.savefig(caminho, bbox_inches="tight", **({"dpi": 300} if str(caminho).endswith(".png") else {}))
    plt.close(fig)


# Recusas (classificar_recusas.py): o veredito do juiz sobre o contexto que o LLM recebeu. Os dois tons de azul são as
# recusas justificadas (a informação não veio, ou veio só em parte); o laranja, as injustificadas (veio toda).
VEREDITOS_RECUSA = [("ausente", "#184f95", "Justificada: o contexto não tinha a informação"),
                    ("parcial", "#6da7ec", "Justificada: o contexto tinha só parte da informação"),
                    ("completa", LARANJA, "Injustificada: o contexto tinha toda a informação")]


def _barras_por_metodo_llm(contagem: pd.Series, categorias: list, rotulo_y: str, arquivo, conjuntos=None,
                           figsize=(7.1, 4.9), ajuste=None, legenda_ancora=(0.055, 1.0), dpi=None):
    """Barras empilhadas por método (grupos) e LLM (barras, pela sigla), um painel por dataset (Docugami e FaQuAD).
    `contagem` é indexada por (dataset, método, LLM, categoria); `categorias`, uma lista de (categoria, cor, rótulo).
    Os parâmetros opcionais servem para as versões dos slides; sem eles, sai a figura do artigo."""
    conjuntos = conjuntos or DATASETS_CONTEXTO
    fig, eixos = plt.subplots(len(conjuntos), 1, figsize=figsize, squeeze=False)
    eixos = eixos[:, 0]
    fig.subplots_adjust(**(ajuste or dict(left=0.065, right=0.995, top=0.86, bottom=0.07, hspace=0.42)))
    # Os intermediários (25 perguntas) ficam um pouco afastados dos econômicos (100) dentro de cada método.
    posicao = {llm: j + (0.45 if llm in INTERMEDIARIOS else 0) for j, llm in enumerate(TODOS)}
    largura = max(posicao.values()) + 1
    passo = largura + 1.6
    for ax, nome in zip(eixos, conjuntos):
        _estilo(ax)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color=GRADE, linewidth=0.6)
        for i, metodo in enumerate(ab.METODOS_RAG):
            for llm in TODOS:
                x, base = i * passo + posicao[llm], 0
                for categoria, cor, _ in categorias:
                    n = int(contagem.get((nome, metodo, llm, categoria), 0))
                    if n:
                        ax.bar(x, n, bottom=base, width=0.8, color=cor, edgecolor="white", linewidth=0.5)
                        base += n
                ax.text(x, -0.025, SIGLA_LLM[llm],
                        transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.3, color=TEXTO_2)
            ax.text(i * passo + (largura - 1) / 2, -0.15, ab.NOME_METODO[metodo], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.3, color=TEXTO)
        ax.set_xticks([])
        ax.set_xlim(-0.8, (len(ab.METODOS_RAG) - 1) * passo + largura - 0.2)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
        ax.set_ylabel(rotulo_y, fontsize=7.5, color=TEXTO_2)
        ax.set_title(NOME_DATASET[nome], fontsize=8, color=TEXTO, loc="left", pad=4)
    legenda = [plt.Rectangle((0, 0), 1, 1, color=cor, label=rotulo) for _, cor, rotulo in categorias]
    fig.legend(handles=legenda, loc="upper left", ncol=2, frameon=False, fontsize=7.3, bbox_to_anchor=legenda_ancora,
               handlelength=1.2, columnspacing=1.5)
    fig.savefig(GRAFICOS / arquivo, **({"dpi": dpi} if dpi else {}))
    plt.close(fig)


DATASETS_CONTEXTO = ["docugami_10q", "faquad"]
# O que o modelo fez quando o contexto não tinha nada da informação: azul, o comportamento pedido pelo prompt (recusar);
# laranja, responder assim mesmo -- as mesmas cores do certo e do errado na figura de recusas.
CONDUTAS_SEM_INFORMACAO = [("recusou", "#184f95", "Recusou (o que o prompt pede)"),
                           ("respondeu", LARANJA, "Respondeu mesmo sem a informação")]


def carregar_contextos() -> pd.DataFrame:
    """Uma linha por resposta dos métodos de RAG, com o veredito do juiz sobre o contexto (classificar_contextos.py)."""
    ctx = pd.read_json(METRICAS_DIR / "contextos_classificados.jsonl", lines=True)
    return ctx[[ab.na_amostra(r) for r in ctx[["dataset", "pergunta_id"]].to_dict("records")]].reset_index(drop=True)


def figura_recusas(ctx: pd.DataFrame) -> dict:
    """Recusas por método e LLM (repetição 1), empilhadas pelo veredito do juiz sobre o contexto, no Docugami e no
    FaQuAD. Devolve as contagens para o JSON de números do artigo."""
    rec = ctx[ctx.recusa]
    _barras_por_metodo_llm(rec.groupby(["dataset", "metodo_rag", "llm", "veredito_juiz"]).size(), VEREDITOS_RECUSA,
                           "Recusas", "recusas.pdf")
    saida = {}
    for (nome, metodo, llm), g in rec.groupby(["dataset", "metodo_rag", "llm"]):
        saida[f"{nome}|{llm}|{metodo}"] = {v: int((g.veredito_juiz == v).sum()) for v, _, _ in VEREDITOS_RECUSA}
    for (nome, metodo), g in rec[rec.llm.isin(PRINCIPAIS)].groupby(["dataset", "metodo_rag"]):
        saida[f"{nome}|principais|{metodo}"] = {v: int((g.veredito_juiz == v).sum()) for v, _, _ in VEREDITOS_RECUSA}
    for nome, g in rec.groupby("dataset"):
        saida[f"{nome}|todos"] = {v: int((g.veredito_juiz == v).sum()) for v, _, _ in VEREDITOS_RECUSA}
    fq = ctx[ctx.dataset == "faquad"]
    saida["validacao_faquad_recusas"] = pd.crosstab(fq[fq.recusa].veredito_juiz, fq[fq.recusa].resposta_no_contexto.astype(bool)).to_dict()
    saida["validacao_faquad_todas"] = pd.crosstab(fq.veredito_juiz, fq.resposta_no_contexto.astype(bool)).to_dict()
    return saida


def figura_sem_informacao(ctx: pd.DataFrame):
    """Respostas em que o contexto não tinha nada da informação (veredito "ausente"), por método e LLM: recusou ou
    respondeu assim mesmo. A altura da barra é o número de perguntas em que a informação não chegou ao modelo."""
    sem = ctx[ctx.veredito_juiz == "ausente"].assign(conduta=lambda d: np.where(d.recusa, "recusou", "respondeu"))
    _barras_por_metodo_llm(sem.groupby(["dataset", "metodo_rag", "llm", "conduta"]).size(), CONDUTAS_SEM_INFORMACAO,
                           "Respostas sem a informação\nno contexto", "sem_informacao.pdf")


def tabela_contexto(ctx: pd.DataFrame) -> dict:
    """Decomposição por método nos cinco econômicos (camada principal): com que frequência a informação chegou ao
    LLM, o que ele fez sem ela (rejeição negativa, como no RGB) e com ela (recusa injustificada), e a métrica
    principal em cada caso."""
    g = ctx[ctx.llm.isin(PRINCIPAIS)]
    saida, linhas_tex = {}, []
    for metodo in ab.METODOS_RAG:
        celulas = []
        for nome in DATASETS_CONTEXTO:
            h = g[(g.dataset == nome) & (g.metodo_rag == metodo)]
            completa, ausente = h[h.veredito_juiz == "completa"], h[h.veredito_juiz == "ausente"]
            r = {"n": len(h), "completa": (h.veredito_juiz == "completa").mean(), "parcial": (h.veredito_juiz == "parcial").mean(),
                 "ausente": (h.veredito_juiz == "ausente").mean(),
                 "rejeicao_negativa": ausente.recusa.mean(), "respondeu_sem_informacao": int((~ausente.recusa).sum()),
                 "recusa_injustificada": completa.recusa.mean(),
                 "metrica_com_informacao": completa.metrica_principal.mean(),
                 "metrica_respondeu_sem_informacao": ausente[~ausente.recusa].metrica_principal.mean()}
            saida[f"{nome}|{metodo}"] = {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in r.items()}
            celulas += [num(100 * r["completa"], 0), num(100 * r["ausente"], 0), num(100 * r["rejeicao_negativa"], 0),
                        num(100 * r["recusa_injustificada"], 0), num(r["metrica_com_informacao"]),
                        num(r["metrica_respondeu_sem_informacao"])]
        linhas_tex.append(f"{ab.NOME_METODO[metodo]} & " + " & ".join(celulas) + " \\\\")
    linha1 = " & " + " & ".join(["Inf.", "Inf.", "Rej.", "Recusa", "Métrica", "Métrica"] * 2) + " \\\\"
    linha2 = "Método & " + " & ".join(["completa", "ausente", "negativa", "injust.", "c/ inf.", "s/ inf."] * 2) + " \\\\"
    linhas = ["\\begin{table*}[!t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
              "\\caption{O que chegou ao LLM e o que ele fez, por método, nos cinco econômicos (camada principal, 500 respostas por "
              "método e conjunto), segundo o juiz. Inf. completa e Inf. ausente: porcentagem das respostas em que o contexto tinha toda a "
              "informação ou nada dela (o restante teve parte). Rej. negativa: dos casos sem a informação, a porcentagem em que o modelo "
              "recusou, como no RGB \\cite{chen2024rgb}; o restante respondeu sem a informação. Recusa injust.: dos casos com toda a "
              "informação, a porcentagem em que o modelo recusou. Métrica c/ inf.: métrica principal quando o contexto tinha toda a "
              "informação; Métrica s/ inf.: quando não tinha nada dela e o modelo respondeu assim mesmo.}", "\\label{tab:contexto}",
              "\\begin{tabular}{l" + "c" * 12 + "}", "\\toprule",
              " & \\multicolumn{6}{c}{Docugami SEC 10-Q (valores do gabarito)} & \\multicolumn{6}{c}{FaQuAD (cobertura de tokens)} \\\\",
              "\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}",
              linha1, linha2,
              "\\midrule"] + linhas_tex + ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    escrever("tab_contexto", linhas)
    return saida


def resumo_numeros(df, medias, sig, resumo, relativo, tokens, tempo, porte, med_bge, rag, bancos, stuff, recusas=None):
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
    if recusas:
        saida["recusas_por_veredito"] = recusas
    saida["ragas_faithfulness"] = rag.pivot_table(index=["dataset", "llm"], columns="metodo_rag", values="faithfulness").round(3).to_json()
    (TABELAS / "numeros_artigo.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")


def copiar_para_artigo():
    """Copia as tabelas (.tex) e as figuras (.pdf) para dentro da pasta do artigo, que assim não
    depende de nada fora dela (o Overleaf só enxerga a pasta enviada)."""
    for origem, destino, padrao in [(TABELAS, ARTIGO / "tabelas", "*.tex"), (GRAFICOS, ARTIGO / "figuras", "*.pdf")]:
        destino.mkdir(parents=True, exist_ok=True)
        for arquivo in origem.glob(padrao):
            shutil.copy2(arquivo, destino / arquivo.name)


def main():
    TABELAS.mkdir(parents=True, exist_ok=True)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "pdf.fonttype": 42})
    df = ab.carregar_execucoes()
    medias, sig = tabela_principal(df)
    tokens_por_pergunta = tabela_tokens(df)
    custo_usd = tabela_custo_usd(df)
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
    figura_custo_desempenho(medias, custo_usd)
    figura_custo_tokens(tokens_por_pergunta, custo_usd)
    ctx = carregar_contextos()
    recusas = figura_recusas(ctx)
    figura_sem_informacao(ctx)
    recusas["decomposicao_por_metodo"] = tabela_contexto(ctx)
    diagramas_rag.figura_pipeline(GRAFICOS)
    diagramas_rag.figura_metodos(GRAFICOS)
    resumo_numeros(df, medias, sig, resumo, relativo, tokens, tempo, porte, med_bge, rag, bancos, stuff, recusas)
    copiar_para_artigo()
    print(f"Escritos {TABELAS.relative_to(BASE_DIR)} e {GRAFICOS.relative_to(BASE_DIR)}, copiados para "
          f"{ARTIGO.name}/tabelas e {ARTIGO.name}/figuras")


if __name__ == "__main__":
    main()
