"""
Análise da rodada dos benchmarks (rodar_benchmarks.py + avaliar_recuperacao.py +
avaliar_ragas_benchmarks.py). Escreve resultados/analise_benchmarks.md com:

1. Recuperação (Recall@k, sem LLM).
2. Camada principal (5 modelos, 100 perguntas por dataset): métrica principal, cosseno, recusas,
   tokens, custo e tempo por LLM e método; Friedman entre os métodos por LLM.
3. Efeito do RAG: cada método contra o controle sem RAG.
4. Generalização (7 modelos, as mesmas 25 perguntas por dataset): concordância da ordem dos métodos
   entre os LLMs (W de Kendall).
5. Variação do modelo: desvio entre as 3 repetições do Stuff.
6. Comparação com a Tabela 1 de Şakar & Emekci (2025) no Docugami e no MedQA.
7. RAGAS: Faithfulness e Answer Relevancy por LLM e método; Context Precision/Recall por dataset, e a
   correlação delas com o Hit@4 onde há gabarito de recuperação.
8. Bancos vetoriais: ChromaDB (HNSW) × FAISS (busca exata).
9. Efeito do porte (FrugalGPT): o econômico com o melhor método contra o intermediário com o Stuff.

Métrica principal de cada dataset (comparada direto com o gabarito, sem juiz): acurácia no MedQA,
cobertura de tokens no FaQuAD (o F1 vai junto, mas pune as respostas em frase contra um gabarito de
poucas palavras) e, no Docugami, a fração dos valores do gabarito (US$, %, milhões...) que aparecem na
resposta. A similaridade de cosseno, métrica do artigo-base, continua nas tabelas e na comparação com
ele, mas não mede acerto no Docugami: resposta genérica, sem nenhum número, tira nota alta (ver
metricas_resposta.py).
"""

import json

import numpy as np
import pandas as pd
from scipy import stats

import benchmarks
import metricas_resposta
import pipeline_base as pb
from analisar_resultados import BASE_DIR, METRICAS_DIR, br, br_p, friedman_com_posthoc, holm, ic_bootstrap, md_tabela

SAIDA = BASE_DIR / "resultados" / "analise_benchmarks.md"
METODOS = ["sem_rag", "stuff", "refine", "map_reduce", "map_rerank", "query_step_down", "reciprocal"]
METODOS_RAG = METODOS[1:]
NOME_METODO = {"sem_rag": "Sem RAG", "stuff": "Stuff", "refine": "Refine", "map_reduce": "Map-Reduce",
               "map_rerank": "Map-Rerank", "query_step_down": "Step-Down", "reciprocal": "Reciprocal"}
METRICA_PRINCIPAL = {"docugami_10q": ("valores_gabarito", "Valores do gabarito na resposta"),
                     "medqa": ("acerto", "Acurácia"),
                     "faquad": ("cobertura_tokens", "Cobertura de tokens")}

# Tabela 1 de Şakar & Emekci (2025): mediana da similaridade de cosseno por método.
SIMILARIDADE_PAPER_ANCORA = {
    "docugami_10q": {"reciprocal": 0.971, "stuff": 0.895, "query_step_down": 0.890, "map_reduce": 0.873, "refine": 0.836, "map_rerank": 0.801},
    "medqa": {"query_step_down": 0.678, "reciprocal": 0.673, "stuff": 0.641, "refine": 0.609, "map_reduce": 0.569, "map_rerank": 0.148},
}


def carregar_execucoes() -> pd.DataFrame:
    precos = {l["rotulo"]: l["preco"] for l in pb.LLMS_DO_COMPARATIVO}
    linhas = []
    for nome in benchmarks.DATASETS:
        gabarito = {p["id"]: p for p in benchmarks.carregar_perguntas(nome)}
        for log in sorted(METRICAS_DIR.glob(f"benchmarks_{nome}_*.jsonl")):
            for texto in log.read_text(encoding="utf-8").splitlines():
                if not texto.strip():
                    continue
                r = json.loads(texto)
                r.setdefault("repeticao", 1)
                r.update(metricas_resposta.metricas(r, gabarito[r["pergunta_id"]]))
                preco_entrada, preco_saida = precos.get(r["llm"], (np.nan, np.nan))
                r["custo_usd"] = (r["tokens_prompt"] * preco_entrada + r["tokens_gerados"] * preco_saida) / 1e6
                linhas.append(r)
    df = pd.DataFrame(linhas)
    if not df.empty:
        df["uid"] = df["pergunta_id"]
    return df


def kendall_w(tabela: pd.DataFrame) -> float:
    """W de Kendall: linhas = avaliadores (LLMs), colunas = itens (métodos). 1 = mesma ordem em todos."""
    postos = tabela.rank(axis=1)
    m, n = postos.shape
    s = ((postos.sum(axis=0) - m * (n + 1) / 2) ** 2).sum()
    return float(12 * s / (m**2 * (n**3 - n)))


def kendall_w_p(tabela: pd.DataFrame) -> float:
    """Valor-p do W de Kendall pela aproximação qui-quadrado: m(n-1)W ~ χ² com n-1 graus de liberdade."""
    m, n = tabela.shape
    return float(stats.chi2.sf(m * (n - 1) * kendall_w(tabela), n - 1))


# Efeito do porte: em cada família fechada, um modelo econômico e um intermediário.
FAMILIAS = [("OpenAI", "gpt_luna", "gpt_sol"), ("Anthropic", "claude_haiku", "claude_sonnet")]


def _melhores(medias: pd.Series) -> list[str]:
    return [m for m, v in medias.items() if np.isclose(v, medias.max())]


def comparar_porte(df: pd.DataFrame) -> pd.DataFrame:
    """FrugalGPT (Chen, Zaharia & Zou 2023) aplicado aos métodos de RAG: o modelo econômico com o seu
    melhor método alcança o intermediário da mesma família com o Stuff? O melhor método do econômico é
    escolhido nas 100 perguntas da camada principal e avaliado nas 25 da camada de generalização (as
    únicas que o intermediário respondeu), pareado pergunta a pergunta. Também registra se o melhor
    método nas 25 perguntas muda do econômico pro intermediário."""
    linhas = []
    for nome in benchmarks.DATASETS:
        col = METRICA_PRINCIPAL[nome][0]
        ids = {p["id"] for p in benchmarks.amostra(benchmarks.carregar_perguntas(nome), benchmarks.PERGUNTAS_POR_CAMADA["generalizacao"])}
        g = df[(df["dataset"] == nome) & (df["repeticao"] == 1) & (df["metodo_rag"].isin(METODOS_RAG))]
        for familia, econ, inter in FAMILIAS:
            melhor = g[g["llm"] == econ].groupby("metodo_rag")[col].mean().idxmax()
            e = g[(g["llm"] == econ) & (g["metodo_rag"] == melhor) & g["pergunta_id"].isin(ids)].set_index("pergunta_id")
            i = g[(g["llm"] == inter) & (g["metodo_rag"] == "stuff") & g["pergunta_id"].isin(ids)].set_index("pergunta_id")
            par = pd.concat([e[col].rename("econ"), i[col].rename("inter")], axis=1).dropna()
            if par.empty:
                continue
            if col == "acerto":
                so_e, so_i = int(((par.econ == 1) & (par.inter == 0)).sum()), int(((par.econ == 0) & (par.inter == 1)).sum())
                p = stats.binomtest(so_e, so_e + so_i).pvalue if so_e + so_i else 1.0
            else:
                p = stats.wilcoxon(par.econ, par.inter).pvalue if (par.econ != par.inter).any() else 1.0
            g25 = g[g["pergunta_id"].isin(ids)]
            linhas.append({
                "dataset": nome, "familia": familia, "economico": econ, "intermediario": inter, "n": len(par),
                "melhor_metodo_economico": melhor, "media_economico_melhor": par.econ.mean(),
                "media_intermediario_stuff": par.inter.mean(), "p": p,
                "custo_economico_melhor": e.loc[par.index, "custo_usd"].mean(),
                "custo_intermediario_stuff": i.loc[par.index, "custo_usd"].mean(),
                "melhores_economico_25": _melhores(g25[g25["llm"] == econ].groupby("metodo_rag")[col].mean()),
                "melhores_intermediario_25": _melhores(g25[g25["llm"] == inter].groupby("metodo_rag")[col].mean()),
            })
    return pd.DataFrame(linhas)


def secao_porte(df: pd.DataFrame) -> list[str]:
    porte = comparar_porte(df)
    if porte.empty:
        return []
    linhas = [[r.dataset, r.familia, f"{r.economico} + {NOME_METODO[r.melhor_metodo_economico]}", br(r.media_economico_melhor),
               br(r.media_intermediario_stuff), br_p(r.p), br(r.custo_intermediario_stuff / r.custo_economico_melhor, 1) + "×",
               ", ".join(NOME_METODO[m] for m in r.melhores_economico_25), ", ".join(NOME_METODO[m] for m in r.melhores_intermediario_25)]
              for r in porte.itertuples()]
    return ["\n## 9. Efeito do porte (FrugalGPT): econômico com o melhor método × intermediário com o Stuff\n\n",
            md_tabela(["Dataset", "Família", "Econômico + melhor método (camada principal)", "Econômico", "Intermediário + Stuff",
                       "p", "Custo do intermediário / econômico", "Melhor(es) nas 25 (econômico)", "Melhor(es) nas 25 (intermediário)"], linhas),
            "\n\nMédias nas perguntas da camada de generalização (25; no Docugami, as com valor no gabarito), pareadas. "
            "p: McNemar exato (acurácia) ou Wilcoxon pareado. O melhor método do econômico vem das 100 perguntas da "
            "camada principal, que incluem essas 25.\n"]


def _fmt_media_ic(serie: pd.Series) -> str:
    ic = ic_bootstrap(serie.dropna())
    return f"{br(serie.mean())} [{br(ic[0])}; {br(ic[1])}]"


def secao_recuperacao() -> list[str]:
    caminho = METRICAS_DIR / "recuperacao_benchmarks_resumo.json"
    if not caminho.exists():
        return ["\n## 1. Recuperação\n\nAinda não calculada: rode `avaliar_recuperacao.py`.\n"]
    resumo = json.loads(caminho.read_text(encoding="utf-8"))
    linhas = [[d, str(r["n"]), br(r["hit@4"]), br(r["recall@4"]), br(r["mrr"]), br(r["hit@10"])] for d, r in resumo.items()]
    return ["\n## 1. Recuperação (sem LLM, todas as perguntas)\n\n",
            md_tabela(["Dataset", "Perguntas", "Hit@4", "Recall@4", "MRR@10", "Hit@10"], linhas),
            "\n\nO MedQA não tem trecho certo marcado: a busca dele é avaliada pela Context Precision/Recall (seção 7).\n"]


def secao_principal(df: pd.DataFrame) -> list[str]:
    partes = ["\n## 2. Camada principal (5 modelos econômicos, 100 perguntas por dataset)\n"]
    principais = [l["rotulo"] for l in pb.LLMS_DO_COMPARATIVO if l["camada"] == "principal"]
    for nome in benchmarks.DATASETS:
        col, rotulo_metrica = METRICA_PRINCIPAL[nome]
        g = df[(df["dataset"] == nome) & (df["llm"].isin(principais)) & (df["repeticao"] == 1)]
        if g.empty:
            continue
        partes.append(f"\n### {nome}\n\nMétrica principal: {rotulo_metrica}.\n\n")
        if col == "valores_gabarito":
            partes.append("Só entram as perguntas com algum valor numérico no gabarito (as outras são qualitativas). "
                          "Recusas parciais (\"I don't have verified figures\") contam como recusa.\n\n")
        linhas = []
        for (llm, metodo), h in g.groupby(["llm", "metodo_rag"], sort=False):
            linhas.append([llm, NOME_METODO[metodo], str(h[col].notna().sum()), _fmt_media_ic(h[col]), br(h["similaridade_cosseno"].mean()),
                           f"{br(100 * h['recusa'].mean(), 0)}%", br(h["tokens_totais"].mean(), 0),
                           br(h["custo_usd"].sum(), 4), br(h["tempo_total_s"].median(), 1)])
        partes.append(md_tabela(["LLM", "Método", "n", f"{rotulo_metrica} [IC 95%]", "Cosseno", "Recusas",
                                 "Tokens (média)", "Custo total US$", "Tempo s (mediana)"], linhas))
        testes = []
        for llm, h in g.groupby("llm", sort=False):
            metodos = [m for m in METODOS_RAG if m in set(h["metodo_rag"])]
            if len(metodos) < 3:
                continue  # rodada ainda em andamento: o Friedman precisa de 3 métodos ou mais
            r = friedman_com_posthoc(h, col, metodos)
            sig = [f"{NOME_METODO[p['a']]}×{NOME_METODO[p['b']]}" for p in r.get("pares", []) if p["p_holm"] < 0.05]
            testes.append([llm, str(r["n_blocos"]), br(r.get("chi2", np.nan), 2), br_p(r.get("p", np.nan)), ", ".join(sig) or "--"])
        partes.append("\n\nFriedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:\n\n")
        partes.append(md_tabela(["LLM", "Perguntas", "χ²", "p", "Pares com p_Holm < 0,05"], testes))
        partes.append("\n")
    return partes


def secao_efeito_rag(df: pd.DataFrame) -> list[str]:
    partes = ["\n## 3. Efeito do RAG (método − sem RAG, na métrica principal)\n"]
    for nome in benchmarks.DATASETS:
        col, rotulo_metrica = METRICA_PRINCIPAL[nome]
        g = df[(df["dataset"] == nome) & (df["repeticao"] == 1)]
        if g.empty or "sem_rag" not in set(g["metodo_rag"]):
            continue
        linhas = []
        for llm, h in g.groupby("llm", sort=False):
            tabela = h.pivot_table(index="uid", columns="metodo_rag", values=col, aggfunc="first")
            if "sem_rag" not in tabela:
                continue
            metodos = [m for m in METODOS_RAG if m in tabela]
            pvals, difs = [], []
            for m in metodos:
                par = tabela[["sem_rag", m]].dropna()
                diferenca = par[m] - par["sem_rag"]
                difs.append(diferenca.mean())
                pvals.append(stats.wilcoxon(par[m], par["sem_rag"]).pvalue if (diferenca != 0).any() else 1.0)
            ajustados = holm(pvals)
            linhas.append([llm, br(tabela["sem_rag"].mean())] + [
                f"{'+' if d >= 0 else ''}{br(d)}{'*' if p < 0.05 else ''}" for d, p in zip(difs, ajustados)])
        if linhas:
            partes.append(f"\n### {nome} ({rotulo_metrica})\n\n")
            partes.append(md_tabela(["LLM", "Sem RAG"] + [NOME_METODO[m] for m in metodos], linhas))
            partes.append("\n\n\\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).\n")
    return partes


def secao_generalizacao(df: pd.DataFrame) -> list[str]:
    partes = ["\n## 4. Generalização (7 modelos, as mesmas 25 perguntas por dataset)\n"]
    for nome in benchmarks.DATASETS:
        col, rotulo_metrica = METRICA_PRINCIPAL[nome]
        ids = {p["id"] for p in benchmarks.amostra(benchmarks.carregar_perguntas(nome), benchmarks.PERGUNTAS_POR_CAMADA["generalizacao"])}
        g = df[(df["dataset"] == nome) & (df["repeticao"] == 1) & (df["pergunta_id"].isin(ids)) & (df["metodo_rag"].isin(METODOS_RAG))]
        if g.empty:
            continue
        tabela = g.pivot_table(index="llm", columns="metodo_rag", values=col, aggfunc="mean")
        tabela = tabela.reindex(columns=[m for m in METODOS_RAG if m in tabela]).dropna()
        n = g.dropna(subset=[col])["pergunta_id"].nunique()
        partes.append(f"\n### {nome} ({rotulo_metrica}, média nas {n} perguntas"
                      f"{' com valor no gabarito' if col == 'valores_gabarito' else ''})\n\n")
        partes.append(md_tabela(["LLM"] + [NOME_METODO[m] for m in tabela.columns],
                                [[llm] + [br(v) for v in linha] for llm, linha in tabela.iterrows()]))
        if len(tabela) >= 2:
            partes.append(f"\n\nW de Kendall (concordância da ordem dos métodos entre os {len(tabela)} LLMs): **{br(kendall_w(tabela))}** "
                          f"(p = {br_p(kendall_w_p(tabela))}; 1 = mesma ordem em todos; 0 = nenhuma concordância).\n")
    return partes


def secao_repeticoes(df: pd.DataFrame) -> list[str]:
    g = df[(df["metodo_rag"] == "stuff")]
    repetidas = g.groupby(["dataset", "llm", "pergunta_id"])["repeticao"].nunique()
    if repetidas.empty or repetidas.max() < 2:
        return ["\n## 5. Variação do modelo\n\nAinda sem repetições: rode `rodar_benchmarks.py --repeticoes`.\n"]
    partes = ["\n## 5. Variação do modelo (desvio-padrão entre as 3 execuções do Stuff, por pergunta)\n\n"]
    linhas = []
    for (nome, llm), h in g.groupby(["dataset", "llm"]):
        col = METRICA_PRINCIPAL[nome][0]
        dp = h.groupby("pergunta_id")[col].std().dropna()
        if len(dp):
            linhas.append([nome, llm, str(len(dp)), br(dp.mean())])
    partes.append(md_tabela(["Dataset", "LLM", "Perguntas", "Desvio médio na métrica principal"], linhas))
    partes.append("\n\nCompare com as diferenças entre métodos da seção 2: diferenças menores que esse desvio não se distinguem da variação do próprio modelo.\n")
    return partes


def secao_paper_ancora(df: pd.DataFrame) -> list[str]:
    partes = ["\n## 6. Comparação com Şakar & Emekci (2025), Tabela 1 (mediana da similaridade de cosseno)\n"]
    for nome, ref in SIMILARIDADE_PAPER_ANCORA.items():
        g = df[(df["dataset"] == nome) & (df["repeticao"] == 1)]
        if g.empty:
            continue
        medianas = g.groupby(["metodo_rag", "llm"])["similaridade_cosseno"].median()
        llms = list(dict.fromkeys(g["llm"]))
        partes.append(f"\n### {nome}\n\n")
        partes.append(md_tabela(["Método", "Paper-âncora"] + llms,
                                [[NOME_METODO[m], br(v)] + [br(medianas.get((m, l), np.nan)) for l in llms] for m, v in ref.items()]))
        partes.append("\n")
    partes.append("\nO paper-âncora agrega vários LLMs, embeddings e bancos vetoriais, e não informa o embedding da "
                  "similaridade: a comparação é indicativa.\n")
    caminho = METRICAS_DIR / "similaridade_bge_small.jsonl"
    if caminho.exists():
        bge = pd.DataFrame([json.loads(t) for t in caminho.read_text(encoding="utf-8").splitlines()])
        partes.append("\n### Com o bge-small-en-v1.5, o embedding do artigo-base (mediana de todas as respostas, 7 LLMs)\n\n")
        linhas = []
        for m in METODOS_RAG:
            linha = [NOME_METODO[m]]
            for nome, ref in SIMILARIDADE_PAPER_ANCORA.items():
                h = bge[(bge["dataset"] == nome) & (bge["metodo_rag"] == m)]["similaridade_bge_small"]
                linha += [br(ref[m]), br(h.median())]
            linhas.append(linha)
        partes.append(md_tabela(["Método"] + [c for nome in SIMILARIDADE_PAPER_ANCORA for c in (f"{nome}: artigo", f"{nome}: este trabalho")], linhas))
        partes.append("\n\nCalculada por `similaridade_bge_small.py`, com o mesmo cálculo da similaridade dos experimentos.\n")
    return partes


def secao_ragas() -> list[str]:
    partes = ["\n## 7. RAGAS (juiz gpt-4o-mini; métricas complementares)\n"]
    respostas = []
    for caminho in sorted(METRICAS_DIR.glob("ragas_*.jsonl")):
        if caminho.name.startswith("ragas_contexto_"):
            continue
        for texto in caminho.read_text(encoding="utf-8").splitlines():
            r = json.loads(texto)
            respostas.append({**{k: r[k] for k in ("dataset", "llm", "metodo_rag", "pergunta_id")}, **r["ragas"]})
    if respostas:
        rag = pd.DataFrame(respostas)
        for coluna, rotulo in (("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer Relevancy")):
            if coluna not in rag:
                continue
            tabela = rag.pivot_table(index=["dataset", "llm"], columns="metodo_rag", values=coluna, aggfunc="mean")
            tabela = tabela.reindex(columns=[m for m in METODOS if m in tabela]).dropna(how="all")
            partes.append(f"\n### {rotulo} (média)\n\n")
            partes.append(md_tabela(["Dataset", "LLM"] + [NOME_METODO[m] for m in tabela.columns],
                                    [[d, l] + [br(v) for v in linha] for (d, l), linha in tabela.iterrows()]))
            partes.append("\n")
        partes.append("\nO gpt-4o-mini é juiz e também um dos modelos testados (viés de autopreferência; Panickssery et al., 2024): "
                      "as notas dele não entram na comparação entre LLMs.\n")
    recuperacao = METRICAS_DIR / "recuperacao_benchmarks.jsonl"
    hit4 = {}
    if recuperacao.exists():
        for texto in recuperacao.read_text(encoding="utf-8").splitlines():
            r = json.loads(texto)
            hit4[(r["dataset"], r["pergunta_id"])] = r["hit@4"]
    linhas = []
    for nome in benchmarks.DATASETS:
        caminho = METRICAS_DIR / f"ragas_contexto_{nome}.jsonl"
        if not caminho.exists():
            continue
        ctx = pd.DataFrame([{"pergunta_id": json.loads(t)["pergunta_id"], **json.loads(t)["ragas"]}
                            for t in caminho.read_text(encoding="utf-8").splitlines()])
        ctx["hit4"] = [hit4.get((nome, p)) for p in ctx["pergunta_id"]]
        pares = ctx.dropna(subset=["context_precision", "hit4"])
        rho = stats.spearmanr(pares["context_precision"], pares["hit4"]) if len(pares) > 2 and pares["hit4"].nunique() > 1 else None
        linhas.append([nome, str(len(ctx)), br(ctx["context_precision"].mean()), br(ctx["context_recall"].mean()),
                       f"{br(rho.statistic)} (p={br_p(rho.pvalue)})" if rho else "--"])
    if linhas:
        partes.append("\n### Context Precision e Context Recall (uma vez por pergunta)\n\n")
        partes.append(md_tabela(["Dataset", "Perguntas", "Context Precision", "Context Recall", "Spearman com o Hit@4"], linhas))
        partes.append("\n")
    if len(partes) == 1:
        partes.append("\nAinda não calculado: rode `avaliar_ragas_benchmarks.py respostas` e `... contexto`.\n")
    return partes


def secao_bancos() -> list[str]:
    caminho = METRICAS_DIR / "comparativo_vectorstores_benchmarks.json"
    if not caminho.exists():
        return ["\n## 8. Bancos vetoriais\n\nAinda não calculado: rode `comparar_bancos_passo1_chroma.py` e `comparar_bancos_passo2_faiss.py`.\n"]
    linhas = []
    for r in json.loads(caminho.read_text(encoding="utf-8")):
        c, f = r["chromadb"], r["faiss"]
        linhas.append([f"{r['dataset']} ({r['n_vetores']:,})".replace(",", "."),
                       br(c["tempo_indexacao_s"], 2), br(f["tempo_indexacao_s"], 3),
                       br(c["tempo_consulta_medio_ms"], 2), br(f["tempo_consulta_medio_ms"], 2),
                       br(c["memoria_indexacao_mb"], 1), br(f["memoria_indexacao_mb"], 1),
                       f"{br(100 * r['concordancia_top4'], 0)}%", f"{br(100 * r['sobreposicao_media_top4'], 0)}%",
                       " × ".join(br(r["hit@4"][b]) for b in ("chromadb", "faiss")) if r["hit@4"]["chromadb"] is not None else "--"])
    partes = ["\n## 8. Bancos vetoriais: ChromaDB (HNSW, aproximado) × FAISS (IndexFlatL2, exato)\n\n",
              md_tabela(["Dataset (vetores)", "Index. Chroma (s)", "Index. FAISS (s)", "Consulta Chroma (ms)",
                         "Consulta FAISS (ms)", "Mem. Chroma (MB)", "Mem. FAISS (MB)", "Top-4 idêntico",
                         "Sobreposição top-4", "Hit@4 Chroma × FAISS"], linhas),
              "\n\nTop-4 idêntico: fração das consultas em que os dois bancos entregam os mesmos 4 fragmentos ao LLM "
              "(nelas, a resposta do RAG seria a mesma com o FAISS). Medido num índice Chroma recriado só pro "
              "comparativo; o HNSW não é determinístico, então a tabela abaixo compara com o índice dos experimentos.\n"]
    caminho_stuff = METRICAS_DIR / "comparativo_stuff_faiss.json"
    if caminho_stuff.exists():
        linhas = []
        for r in json.loads(caminho_stuff.read_text(encoding="utf-8")):
            p = r.get("p_mcnemar", r.get("p_wilcoxon"))
            linhas.append([r["dataset"], r["metrica_rotulo"], f"{br(100 * r['top4_identico_ao_experimento'], 0)}%",
                           str(r["perguntas_refeitas"]), br(r["media_chroma"]), br(r["media_faiss"]),
                           " × ".join(br(r[k]) for k in ("media_chroma_nas_refeitas", "media_faiss_nas_refeitas"))
                           if r["perguntas_refeitas"] else "--",
                           f"{r['acerta_so_chroma']} × {r['acerta_so_faiss']}" if "acerta_so_chroma" in r else "--",
                           br_p(p) if p is not None else "--"])
        partes += ["\n### Efeito na resposta: Stuff do gpt-4o-mini com o top-4 do FAISS (100 perguntas)\n\n",
                   md_tabela(["Dataset", "Métrica", "Top-4 igual ao dos experimentos", "Perguntas refeitas",
                              "Chroma", "FAISS", "Nas refeitas (Chroma × FAISS)", "Só um acerta (Chroma × FAISS)", "p"], linhas),
                   "\n\nSó as perguntas com top-4 diferente são respondidas de novo com os fragmentos do FAISS; nas outras, "
                   "o LLM recebe a mesma entrada. p: McNemar exato (acurácia) ou Wilcoxon pareado, nas perguntas refeitas. "
                   "O gpt-4o-mini roda com temperatura 0, mas a API não garante saída idêntica: parte da diferença pode ser "
                   "variação do próprio modelo.\n"]
    return partes


def main():
    df = carregar_execucoes()
    partes = ["# Análise da rodada dos benchmarks\n", f"\n{len(df)} execuções nos logs `benchmarks_*.jsonl`.\n"]
    partes += secao_recuperacao()
    if not df.empty:
        partes += secao_principal(df) + secao_efeito_rag(df) + secao_generalizacao(df) + secao_repeticoes(df) + secao_paper_ancora(df)
    partes += secao_ragas() + secao_bancos()
    if not df.empty:
        partes += secao_porte(df)
    SAIDA.write_text("".join(partes), encoding="utf-8")
    print(f"Escrito {SAIDA.relative_to(BASE_DIR)} ({len(df)} execuções)")


if __name__ == "__main__":
    main()
