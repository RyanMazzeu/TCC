"""
Baixa os benchmarks em dados/brutos/ (só o que falta) e converte pro formato dos scripts do TCC
(dados/processados/):

- docugami_10q/   -> docugami_10q.json     (LlamaIndex: 195 perguntas, 20 relatórios 10-Q)
- medqa/          -> medqa.json            (MedQA-USMLE, teste: 1.273 questões de 4 alternativas; corpus
                                           dos 18 livros-texto que acompanham o MedQA)
- faquad/         -> faquad.json           (FaQuAD: 900 perguntas, 249 parágrafos originais em PT-BR)
- ragbench/       -> ragbench_amostra_juiz.jsonl (amostra pra validar o juiz RAGAS)

Docugami e MedQA são dois dos três datasets do paper-âncora (Şakar & Emekci 2025). O Docugami vem de
github.com/run-llama/llama-datasets; os PDFs viram texto com `pdftotext -layout`: sem o -layout, o
pdftotext separa o rótulo de cada linha das tabelas dos valores dela (ex.: "Total net sales" de
"117,154"), e as perguntas do 10-Q são quase todas sobre esses valores.

Llama2 Paper (o terceiro dataset do paper-âncora, com gabarito todo gerado pelo GPT-4) e Pirá 2.0
(documentos traduzidos automaticamente) foram testados e saíram do desenho.
"""

import json
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
BRUTOS = BASE_DIR / "dados" / "brutos"
PROCESSADOS = BASE_DIR / "dados" / "processados"

RAGBENCH_SUBCONJUNTOS = ["delucionqa", "covidqa", "pubmedqa", "finqa", "tatqa", "techqa", "emanual"]
RAGBENCH_POR_CLASSE = 15  # por subconjunto: 15 respostas fiéis ao contexto + 15 com alucinação
SEED = 42

_LLAMA = "https://media.githubusercontent.com/media/run-llama/llama-datasets/main/llama_datasets"
_HF = "https://huggingface.co/api/datasets"
_RELATORIOS_10Q = [f"{t} {e}" for t in ["2022 Q3", "2023 Q1", "2023 Q2", "2023 Q3"] for e in ["AAPL", "AMZN", "INTC", "MSFT", "NVDA"]]
ARQUIVOS = {
    "docugami_10q/rag_dataset.json": f"{_LLAMA}/docugami_kg_rag/sec_10_q/rag_dataset.json",
    **{f"docugami_10q/{r}.pdf": f"{_LLAMA}/docugami_kg_rag/sec_10_q/source_files/{r.replace(' ', '%20')}.pdf" for r in _RELATORIOS_10Q},
    "faquad/dataset.json": "https://raw.githubusercontent.com/liafacom/faquad/master/data/dataset.json",
    # MedQA-USMLE de 4 alternativas (Jin et al. 2021), no espelho do Hugging Face com a divisão de teste
    # oficial; os livros-texto são os do próprio MedQA, já em trechos, no espelho do MedRAG (Xiong et al. 2024).
    "medqa/test.parquet": f"{_HF}/GBaker/MedQA-USMLE-4-options/parquet/default/test/0.parquet",
    "medqa/textbooks.parquet": f"{_HF}/MedRAG/textbooks/parquet/default/train/0.parquet",
    **{f"ragbench/{n}_test.parquet": f"{_HF}/galileo-ai/ragbench/parquet/{n}/test/0.parquet" for n in RAGBENCH_SUBCONJUNTOS},
}


def baixar():
    """Baixa só o que ainda não está em dados/brutos/."""
    for destino, url in ARQUIVOS.items():
        caminho = BRUTOS / destino
        if caminho.exists():
            continue
        caminho.parent.mkdir(parents=True, exist_ok=True)
        resposta = requests.get(url, timeout=600)
        resposta.raise_for_status()
        caminho.write_bytes(resposta.content)
        print(f"baixado: {destino}")


def pdf_para_texto(pdf: Path) -> Path:
    destino = pdf.with_suffix(".txt")
    bruto = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    # O -layout alinha colunas com dezenas de espaços: reduz cada sequência a dois espaços, que
    # bastam pra separar rótulo e valores sem gastar o fragmento de 1.000 caracteres com espaço.
    linhas = [re.sub(r" {2,}", "  ", linha).rstrip() for linha in bruto.replace("\f", "\n\n").splitlines()]
    destino.write_text(re.sub(r"\n{3,}", "\n\n", "\n".join(linhas)).strip() + "\n", encoding="utf-8")
    return destino


def _salvar(nome: str, dados: dict):
    caminho = PROCESSADOS / nome
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{caminho.relative_to(BASE_DIR)}: {len(dados['perguntas'])} perguntas")


def preparar_docugami():
    pasta = BRUTOS / "docugami_10q"
    documentos = []
    for pdf in sorted(pasta.glob("*.pdf")):
        txt = pdf_para_texto(pdf)
        documentos.append({"document_id": pdf.stem, "arquivo": str(txt.relative_to(BRUTOS))})
    exemplos = json.loads((pasta / "rag_dataset.json").read_text(encoding="utf-8"))["examples"]
    perguntas = []
    for i, e in enumerate(exemplos, start=1):
        # A resposta de referência cita cada relatório usado como "(SOURCE: 2023 Q1 AAPL.pdf)": é o
        # documento-ouro da pergunta. 2 das 195 não citam nenhum (ficam fora do Recall@k).
        fontes = list(dict.fromkeys(re.findall(r"(20\d\d Q\d [A-Z]+)\.pdf", e["reference_answer"])))
        perguntas.append({
            "id": f"docugami_q{i:03d}",
            "pergunta": e["query"],
            "resposta_esperada": e["reference_answer"],
            "documentos_ouro": fontes,
        })
    _salvar("docugami_10q.json", {
        "fonte": "LlamaIndex Llama Datasets — Docugami KG-RAG SEC 10-Q (Şakar & Emekci 2025 usam o mesmo)",
        "url": "https://github.com/run-llama/llama-datasets/tree/main/llama_datasets/docugami_kg_rag/sec_10_q",
        "observacao": "Perguntas escritas por humanos; respostas de referência do GPT-4-turbo com revisão humana. "
                      "'documentos_ouro' vem das citações (SOURCE: ...) da própria resposta de referência.",
        "documentos": documentos,
        "perguntas": perguntas,
    })


def preparar_medqa():
    """MedQA-USMLE (Jin et al. 2021), divisão de teste: 1.273 questões reais do exame de licenciamento
    médico dos EUA, com 4 alternativas e a alternativa certa oficial.

    A pergunta enviada ao RAG é o enunciado seguido das alternativas ("A. ...", uma por linha). A
    `resposta_esperada` é o texto da alternativa certa (referência da similaridade de cosseno e do
    RAGAS); a acurácia usa a `letra_correta`. O corpus são os 18 livros-texto que acompanham o MedQA,
    remontados livro a livro (os trechos do MedRAG vêm em ordem, "<livro>_<n>") pra serem refragmentados
    com os mesmos 1.000 caracteres dos outros datasets."""
    pasta = BRUTOS / "medqa"
    livros = pd.read_parquet(pasta / "textbooks.parquet", columns=["id", "title", "content"])
    livros["ordem"] = livros["id"].str.rsplit("_", n=1).str[-1].astype(int)
    corpus = [
        {"document_id": titulo, "texto": "\n".join(grupo.sort_values("ordem")["content"])}
        for titulo, grupo in livros.groupby("title", sort=True)
    ]
    teste = pd.read_parquet(pasta / "test.parquet")
    perguntas = []
    for i, linha in enumerate(teste.itertuples(), start=1):
        opcoes = {letra: linha.options[letra] for letra in "ABCD"}
        perguntas.append({
            "id": f"medqa_q{i:04d}",
            "pergunta": linha.question.strip() + "\n\n" + "\n".join(f"{l}. {t}" for l, t in opcoes.items()),
            "resposta_esperada": opcoes[linha.answer_idx],
            "letra_correta": linha.answer_idx,
            "alternativas": opcoes,
        })
    _salvar("medqa.json", {
        "fonte": "MedQA-USMLE, 4 alternativas, teste (Jin et al. 2021); livros-texto do MedQA via MedRAG/textbooks",
        "url": "https://github.com/jind11/MedQA",
        "observacao": "Questões reais do USMLE com a alternativa certa oficial. O gabarito não marca de onde, nos "
                      "livros, sai a resposta: não há Recall@k neste dataset.",
        "corpus": corpus,
        "perguntas": perguntas,
    })
    print(f"  corpus: {len(corpus)} livros, {sum(len(d['texto']) for d in corpus):,} caracteres")


def preparar_faquad():
    """FaQuAD (Sayama et al., BRACIS 2019): 900 perguntas escritas por pessoas sobre 249 parágrafos de
    18 documentos oficiais de uma universidade federal e 21 artigos da Wikipédia sobre ensino superior.

    No formato SQuAD, cada pergunta vem junto do parágrafo de onde saiu. Aqui os 249 parágrafos viram
    a base do RAG (cada um é um documento, pra o Recall@k conferir se o parágrafo certo foi
    recuperado) e o parágrafo de cada pergunta vira o documento-ouro. Usa o dataset.json (as 900
    perguntas, na grafia original); o dev.json do repositório está todo em minúsculas."""
    dados = json.loads((BRUTOS / "faquad" / "dataset.json").read_text(encoding="utf-8"))["data"]
    corpus, perguntas = [], []
    for doc in dados:
        for paragrafo in doc["paragraphs"]:
            document_id = f"faquad_p{len(corpus) + 1:03d}"
            corpus.append({"document_id": document_id, "titulo": doc["title"], "texto": paragrafo["context"]})
            for qa in paragrafo["qas"]:
                # Mais de um anotador respondeu algumas perguntas: todas as respostas valem (como no
                # SQuAD, o F1 usa a melhor delas). A primeira é a referência da similaridade de cosseno.
                respostas = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                perguntas.append({
                    "id": qa["id"],
                    "pergunta": qa["question"],
                    "resposta_esperada": respostas[0],
                    "respostas_aceitas": respostas,
                    "documentos_ouro": [document_id],
                    "titulo_documento": doc["title"],
                })
    _salvar("faquad.json", {
        "fonte": "FaQuAD — Sayama, Araujo & Fernandes (2019), BRACIS",
        "url": "https://github.com/liafacom/faquad",
        "observacao": "Perguntas e respostas escritas por pessoas, sobre parágrafos originais em PT-BR. As "
                      "respostas são trechos copiados do parágrafo (formato SQuAD).",
        "corpus": corpus,
        "perguntas": perguntas,
    })


def preparar_ragbench():
    rng = random.Random(SEED)
    saida = PROCESSADOS / "ragbench_amostra_juiz.jsonl"
    contagem = defaultdict(int)
    with open(saida, "w", encoding="utf-8") as f:
        for nome in RAGBENCH_SUBCONJUNTOS:
            df = pd.read_parquet(BRUTOS / "ragbench" / f"{nome}_test.parquet")
            df = df[df["adherence_score"].notna()]
            for classe in (True, False):
                idxs = sorted(df.index[df["adherence_score"] == classe])
                for i in sorted(rng.sample(idxs, min(RAGBENCH_POR_CLASSE, len(idxs)))):
                    linha = df.loc[i]
                    f.write(json.dumps({
                        "subconjunto": nome,
                        "id": str(linha["id"]),
                        "pergunta": linha["question"],
                        "documentos": list(linha["documents"]),
                        "resposta": linha["response"],
                        "modelo_gerador": linha["generation_model_name"],
                        "adherence_score": bool(linha["adherence_score"]),
                        "relevance_score": _float_ou_none(linha["relevance_score"]),
                        "utilization_score": _float_ou_none(linha["utilization_score"]),
                        "completeness_score": _float_ou_none(linha["completeness_score"]),
                        # Faithfulness do RAGAS que os próprios autores do RAGBench calcularam.
                        "ragas_faithfulness_ragbench": _float_ou_none(linha["ragas_faithfulness"]),
                    }, ensure_ascii=False) + "\n")
                    contagem[(nome, classe)] += 1
    print(f"{saida.relative_to(BASE_DIR)}: {sum(contagem.values())} respostas")
    for nome in RAGBENCH_SUBCONJUNTOS:
        print(f"  {nome}: {contagem[(nome, True)]} fiéis, {contagem[(nome, False)]} com alucinação")


def _float_ou_none(v):
    return None if pd.isna(v) else float(v)


if __name__ == "__main__":
    baixar()
    preparar_docugami()
    preparar_medqa()
    preparar_faquad()
    preparar_ragbench()
