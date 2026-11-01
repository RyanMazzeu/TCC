"""
Pipeline RAG base — piloto reduzido do escopo original.

Réplica em escala menor de Şakar & Emekci (2025), "Maximizing RAG efficiency"
(ver Referências/Maximizando-Eficiência-RAG). Implementa os 6 métodos de RAG
do paper-âncora — Stuff, Refine, Map-Reduce, Map-Rerank, Query Step-Down e
Reciprocal RAG (ver _METODOS_RAG / _METODOS_NIVEL_QUERY) — com ChromaDB
local e embedding BGE-small (inglês) ou BGE-M3 (multilíngue, configurável).
Ainda faltam: FAISS/Pinecone como bancos vetoriais alternativos, outros
embeddings, e filtros de compressão contextual.

chunk_size=1000 / chunk_overlap=100 seguem o paper âncora, para manter os
resultados comparáveis com os dele.
"""

import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
import psutil
import requests
import tiktoken
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel, FlagModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstores"
METRICAS_DIR = BASE_DIR / "resultados" / "metricas"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# LLM gerador — trocar via .env (LLM_BACKEND + <BACKEND>_MODEL) ou, num script que compara
# vários LLMs no mesmo processo, via configurar_llm():
#   "ollama"    -> modelo open-source local, grátis (OLLAMA_MODEL)
#   "openai"    -> GPT (OPENAI_MODEL, OPENAI_REASONING_EFFORT)
#   "openrouter"-> qualquer modelo (Qwen, DeepSeek, Claude...) por uma chave e um saldo só, formato
#                  OpenAI (OPENROUTER_MODEL, OPENROUTER_API_KEY)
#   "deepseek"  -> API da DeepSeek, formato OpenAI (DEEPSEEK_MODEL, DEEPSEEK_API_KEY)
#   "anthropic" -> Claude, SDK oficial da Anthropic (ANTHROPIC_MODEL, ANTHROPIC_API_KEY; ver backend_anthropic.py)
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
# Modelos com raciocínio (ex.: qwen3.5) pensam por padrão; OLLAMA_THINK=false desliga. Sem a
# variável, o campo nem é enviado (modelos sem raciocínio, como o qwen2.5, recusam o parâmetro).
OLLAMA_THINK = os.getenv("OLLAMA_THINK")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Modelos de raciocínio da OpenAI (ex.: gpt-6-astra) recebem reasoning_effort no lugar de
# temperature. Sem a variável, a chamada segue como no gpt-4o-mini (temperature=0).
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT")
_openai_client = None

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
_deepseek_client = None

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.5-9b")
_openrouter_client = None

# Parâmetros extras da chamada (temperature, raciocínio, provedor) do LLM configurado por
# configurar_llm_por_rotulo; None = os padrões de cada backend.
_PARAMETROS_LLM = None

# Raciocínio desligado em todos os modelos que permitem: pra comparar os LLMs com os métodos de RAG
# e não com o quanto cada um "pensa" antes de responder (e porque raciocínio é cobrado como saída).
_SEM_RACIOCINIO = {"reasoning": {"effort": "none"}}

# Os LLMs do comparativo, em duas camadas (ver plano_execucao_benchmarks.md):
#   - principal: os 5 modelos de faixa de preço econômica, nos 6 métodos e em 100 perguntas por dataset;
#   - generalizacao: os 2 intermediários, nos 6 métodos e só nas 25 primeiras dessas perguntas.
# Raciocínio desligado em todos, pra comparar os métodos de RAG e não o quanto cada modelo "pensa".
#   - Qwen: precisão cheia (bf16), pra resposta não depender de qual provedor quantizado o OpenRouter
#     escolheu naquela chamada.
#   - Claude e GPT-6 não aceitam temperature (a API devolve erro); os demais rodam com temperature 0.
#     `aceita_temperatura` marca quem precisa das repetições do Stuff.
#   - Sonnet 5 e GPT-6 Sol no lugar do Sonnet 5.5 e do GPT-6.1 Sol: os mais novos não deixam desligar
#     o raciocínio.
#   - gpt-4o-mini fica como ponte: é um dos LLMs do paper-âncora e o das rodadas anteriores do TCC.
# `preco`: US$ por 1M de tokens (entrada, saída), consultados em 25/10/2026 -- usados só pra estimar
# o custo de cada execução na análise.
LLMS_DO_COMPARATIVO = [
    {"rotulo": "qwen", "camada": "principal", "backend": "openrouter", "modelo": "qwen/qwen3.5-9b",
     "aceita_temperatura": True, "preco": (0.17, 0.25),
     "parametros": {"temperature": 0.0, "extra_body": {**_SEM_RACIOCINIO, "provider": {"quantizations": ["bf16"]}}}},
    {"rotulo": "deepseek", "camada": "principal", "backend": "openrouter", "modelo": "deepseek/deepseek-v4-pro",
     "aceita_temperatura": True, "preco": (0.209, 0.418),
     "parametros": {"temperature": 0.0, "extra_body": _SEM_RACIOCINIO}},
    {"rotulo": "claude_haiku", "camada": "principal", "backend": "openrouter", "modelo": "anthropic/claude-haiku-5.5",
     "aceita_temperatura": False, "preco": (0.10, 0.50),
     "parametros": {"extra_body": _SEM_RACIOCINIO}},
    {"rotulo": "gpt_luna", "camada": "principal", "backend": "openai", "modelo": "gpt-6-luna",
     "aceita_temperatura": False, "preco": (0.10, 0.50),
     "parametros": {"reasoning_effort": "none"}},
    {"rotulo": "gpt_mini", "camada": "principal", "backend": "openai", "modelo": "gpt-4o-mini",
     "aceita_temperatura": True, "preco": (0.15, 0.60),
     "parametros": {"temperature": 0.0}},
    {"rotulo": "claude_sonnet", "camada": "generalizacao", "backend": "openrouter", "modelo": "anthropic/claude-sonnet-5",
     "aceita_temperatura": False, "preco": (2.0, 10.0),
     "parametros": {"extra_body": _SEM_RACIOCINIO}},
    {"rotulo": "gpt_sol", "camada": "generalizacao", "backend": "openai", "modelo": "gpt-6-sol",
     "aceita_temperatura": False, "preco": (2.0, 10.0),
     "parametros": {"reasoning_effort": "none"}},
]

# Versão do modelo que a API de fato usou na última chamada (ex.: o snapshot por trás de um
# apelido como gpt-4o-mini). Vai pro log em rodar_experimento, pra não depender só do apelido.
_modelo_llm_versao = None

# bge-small-en-v1.5 (default): mesmo modelo do paper-âncora, mas só serve inglês.
# Pra corpus em português, trocar via .env pra EMBEDDING_MODEL_NAME=BAAI/bge-m3 (multilíngue,
# já validado nos notebooks Busca_Semantica/Busca_Hibrida_RRF).
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
_EH_MULTILINGUE = "bge-m3" in EMBEDDING_MODEL_NAME.lower()

_encoder = tiktoken.get_encoding("cl100k_base")
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # fp16 só com GPU: em CPU ele é ~2x mais lento que o fp32 (medido no corpus do MedQA), e os
        # vetores saem iguais (cosseno médio 1,000 entre os dois).
        import torch

        meia_precisao = torch.cuda.is_available()
        if _EH_MULTILINGUE:
            _embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=meia_precisao)
        else:
            _embedding_model = FlagModel(
                EMBEDDING_MODEL_NAME,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=meia_precisao,
            )
    return _embedding_model


def _encode_documentos(modelo, textos: list[str]):
    if _EH_MULTILINGUE:
        return modelo.encode(textos, return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"]
    return modelo.encode(textos)


def _encode_query(modelo, query: str):
    if _EH_MULTILINGUE:
        return modelo.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"]
    return modelo.encode_queries([query])


def carregar_e_chunkar(caminho_documento: str, document_id: str) -> list[dict]:
    return chunkar_texto(Path(caminho_documento).read_text(encoding="utf-8"), document_id)


def chunkar_texto(texto: str, document_id: str) -> list[dict]:
    """Igual a `carregar_e_chunkar`, pra textos que não estão num arquivo próprio (ex.: os 692
    resumos do Pirá, guardados juntos num JSON)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    partes = splitter.split_text(texto)
    return [
        {"document_id": document_id, "chunk_index": i, "text": parte}
        for i, parte in enumerate(partes)
    ]


def construir_indice(chunks: list[dict], colecao_nome: str, vetores=None, diretorio: Path = VECTORSTORE_DIR):
    """Cria (ou recria) uma coleção Chroma persistente a partir dos chunks.

    `vetores` é opcional: passe embeddings já calculados (ex.: em benchmarks que precisam medir
    só o tempo de indexação em si, sem pagar o custo de reencodar tudo de novo).
    `diretorio`: onde o Chroma persiste a coleção (os índices dos benchmarks ficam fora do git)."""
    client = chromadb.PersistentClient(path=str(diretorio))
    try:
        client.delete_collection(colecao_nome)
    except Exception:
        pass
    colecao = client.create_collection(colecao_nome)

    textos = [c["text"] for c in chunks]
    if vetores is None:
        modelo = get_embedding_model()
        vetores = _encode_documentos(modelo, textos)

    # Em lotes: o Chroma recusa um add maior que o limite dele (alguns milhares de itens), e os 20
    # relatórios 10-Q do Docugami passam disso.
    lote = client.get_max_batch_size()
    for inicio in range(0, len(chunks), lote):
        parte = chunks[inicio : inicio + lote]
        colecao.add(
            ids=[f"{c['document_id']}#{c['chunk_index']}" for c in parte],
            embeddings=vetores[inicio : inicio + lote].tolist(),
            documents=textos[inicio : inicio + lote],
            metadatas=[{"document_id": c["document_id"], "chunk_index": c["chunk_index"]} for c in parte],
        )
    return colecao


def consultar_indice(colecao, query: str, k: int = 4, vetor_query=None) -> dict:
    """Busca crua no Chroma: devolve ids, textos e similaridades dos k chunks mais próximos. A
    similaridade de cosseno é derivada da distância L2 que o Chroma já calcula internamente
    (cos_sim = 1 - L2²/2, válido para vetores normalizados -- é o caso do BGE), o que evita
    reencodar documentos só para pontuar similaridade.

    `vetor_query` é opcional: passe um embedding de pergunta já calculado (mesmo motivo do
    parâmetro `vetores` em construir_indice)."""
    if vetor_query is None:
        modelo = get_embedding_model()
        vetor_query = _encode_query(modelo, query)
    resultado = colecao.query(query_embeddings=vetor_query.tolist(), n_results=k)
    return {
        "ids": resultado["ids"][0],
        "documentos": resultado["documents"][0],
        "similaridades": [1 - d / 2 for d in resultado["distances"][0]],
    }


def recuperar(colecao, query: str, k: int = 4, vetor_query=None) -> list[str]:
    return consultar_indice(colecao, query, k, vetor_query)["documentos"]


def recuperar_com_similaridade(colecao, query: str, k: int = 4, vetor_query=None) -> list[tuple[str, float]]:
    """Igual a `recuperar`, mas devolve também a similaridade de cosseno de cada chunk com a
    pergunta. Usado pelo filtro de compressão contextual."""
    resultado = consultar_indice(colecao, query, k, vetor_query)
    return list(zip(resultado["documentos"], resultado["similaridades"]))


def aplicar_compressao_contextual(chunks_com_similaridade: list[tuple[str, float]], limiar: float) -> list[str]:
    """Filtro de compressão contextual: descarta chunks com similaridade abaixo do limiar
    (limiar=0.0 equivale a "sem filtro", mantém todos os k chunks recuperados). Nunca deixa a
    lista vazia -- se todos ficarem abaixo do limiar, mantém ao menos o mais similar."""
    filtrados = [texto for texto, sim in chunks_com_similaridade if sim >= limiar]
    return filtrados if filtrados else [chunks_com_similaridade[0][0]]


# Prompts dos métodos, um conjunto por tipo de dataset; rodar_benchmarks.py escolhe o conjunto do
# dataset (CONJUNTO_PROMPTS):
#   - "pt" e "en": a primeira tentativa foi um conjunto só, em português, com a instrução "Responda no
#     mesmo idioma da pergunta": o gpt-6-astra obedeceu, mas o gpt-6-luna e o gpt-4o-mini responderam
#     em português ~80% das perguntas em inglês, e a similaridade com a resposta de referência (em
#     inglês) caía por causa do idioma, não do conteúdo. O conjunto em inglês é a tradução do
#     português, instrução por instrução, pra o método ser o mesmo nos dois idiomas. O idioma vem
#     escrito ("Answer in English."), e não "no mesmo idioma da pergunta": no piloto, com essa
#     instrução, o gpt-4o-mini respondeu perguntas em inglês em espanhol e em holandês.
#   - "en_multipla_escolha" (MedQA): os outros conjuntos mandam dizer "não sei" quando a resposta não
#     está no contexto, e os livros-texto raramente trazem a resposta exata de uma questão de prova --
#     a acurácia mediria recusa, não conhecimento. Aqui o modelo usa os trechos como apoio e sempre
#     escolhe uma alternativa, como no MIRAGE (Xiong et al., 2024). Só o passo "map" do Map-Reduce
#     continua podendo dizer que o trecho não ajuda; o "reduce" é que escolhe a alternativa.
# "sem_rag" é o grupo de controle: a mesma pergunta, sem nenhum trecho recuperado.
CONJUNTO_PROMPTS = "pt"

_LETRA_PRIMEIRO = (
    "Always choose exactly one option. Start your answer with the letter of the chosen option, "
    "followed by a short justification."
)

PROMPTS = {
    "pt": {
        "stuff": (
            "Responda à pergunta usando apenas o contexto abaixo. Se a resposta não estiver no contexto, "
            "diga que não sabe. Responda em português.\n\n"
            "Contexto:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
        ),
        "refine_inicial": (
            "Responda à pergunta usando o contexto abaixo. Se a resposta não estiver no contexto, "
            "diga que não sabe. Responda em português.\n\n"
            "Contexto:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
        ),
        "refine_passo": (
            "Sua tarefa é refinar uma resposta existente, apenas se o novo contexto trouxer informação "
            "relevante que ainda não esteja nela. Se o novo contexto não ajudar, repita a resposta existente "
            "sem mudanças. Responda em português.\n\n"
            "Pergunta original: {pergunta}\n\nResposta existente: {resposta}\n\n"
            "Novo contexto:\n{contexto}\n\nResposta refinada:"
        ),
        "map": (
            "Responda à pergunta usando apenas o contexto abaixo. Se a resposta não estiver no contexto, "
            "diga apenas 'Não há informação relevante.' Responda em português.\n\n"
            "Contexto:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
        ),
        "reduce": (
            "Combine as respostas parciais abaixo (geradas a partir de diferentes trechos de um documento) "
            "em uma única resposta final, coerente e não repetitiva, para a pergunta original. Ignore "
            "respostas parciais que digam não haver informação relevante. Responda em português.\n\n"
            "Pergunta original: {pergunta}\n\nRespostas parciais:\n{respostas}\n\nResposta final combinada:"
        ),
        "confianca": (
            "Responda à pergunta usando apenas o contexto abaixo, e avalie sua própria confiança de que a "
            "resposta está correta e plenamente sustentada pelo contexto. Responda EXATAMENTE neste formato, "
            "em duas linhas:\nRESPOSTA: <sua resposta>\nCONFIANCA: <número inteiro de 0 a 100>\n"
            "Responda em português.\n\nContexto:\n{contexto}\n\nPergunta: {pergunta}"
        ),
        "alternativas": (
            "Com base no contexto abaixo e na pergunta original do usuário, gere exatamente {n} perguntas "
            "alternativas, mais específicas e menos ambíguas, que ajudem a esclarecer a intenção da pergunta "
            "original. Responda APENAS com as {n} perguntas, uma por linha, numeradas, sem texto adicional, "
            "em português.\n\nContexto:\n{contexto}\n\nPergunta original: {pergunta}"
        ),
        "combinacao": (
            "Combine as respostas abaixo (geradas a partir de variações da pergunta original) em uma única "
            "resposta final, coerente e não repetitiva, para a pergunta original. Responda em português.\n\nPergunta original: {pergunta}\n\nRespostas das variações:\n{respostas}\n\n"
            "Resposta final combinada:"
        ),
        "sem_rag": (
            "Responda à pergunta. Se não souber a resposta, diga que não sabe. Responda em português.\n\nPergunta: {pergunta}\nResposta:"
        ),
    },
    "en": {
        "stuff": (
            "Answer the question using only the context below. If the answer is not in the context, "
            "say that you don't know. Answer in English.\n\n"
            "Context:\n{contexto}\n\nQuestion: {pergunta}\nAnswer:"
        ),
        "refine_inicial": (
            "Answer the question using the context below. If the answer is not in the context, "
            "say that you don't know. Answer in English.\n\n"
            "Context:\n{contexto}\n\nQuestion: {pergunta}\nAnswer:"
        ),
        "refine_passo": (
            "Your task is to refine an existing answer, only if the new context brings relevant information "
            "that is not in it yet. If the new context does not help, repeat the existing answer without "
            "changes. Answer in English.\n\n"
            "Original question: {pergunta}\n\nExisting answer: {resposta}\n\n"
            "New context:\n{contexto}\n\nRefined answer:"
        ),
        "map": (
            "Answer the question using only the context below. If the answer is not in the context, "
            "say only 'There is no relevant information.' Answer in English.\n\n"
            "Context:\n{contexto}\n\nQuestion: {pergunta}\nAnswer:"
        ),
        "reduce": (
            "Combine the partial answers below (generated from different excerpts of a document) into a "
            "single final answer, coherent and not repetitive, to the original question. Ignore partial "
            "answers that say there is no relevant information. Answer in English.\n\n"
            "Original question: {pergunta}\n\nPartial answers:\n{respostas}\n\nCombined final answer:"
        ),
        "confianca": (
            "Answer the question using only the context below, and rate your own confidence that the answer "
            "is correct and fully supported by the context. Answer EXACTLY in this format, in two lines:\n"
            "ANSWER: <your answer>\nCONFIDENCE: <integer from 0 to 100>\n"
            "Answer in English.\n\nContext:\n{contexto}\n\nQuestion: {pergunta}"
        ),
        "alternativas": (
            "Based on the context below and on the user's original question, generate exactly {n} alternative "
            "questions, more specific and less ambiguous, that help clarify the intent of the original "
            "question. Answer ONLY with the {n} questions, one per line, numbered, with no additional text, "
            "in English.\n\nContext:\n{contexto}\n\nOriginal question: {pergunta}"
        ),
        "combinacao": (
            "Combine the answers below (generated from variations of the original question) into a single "
            "final answer, coherent and not repetitive, to the original question. Answer in English.\n\nOriginal question: {pergunta}\n\nAnswers to the variations:\n{respostas}\n\n"
            "Combined final answer:"
        ),
        "sem_rag": (
            "Answer the question. If you don't know the answer, say that you don't know. Answer in English.\n\nQuestion: {pergunta}\nAnswer:"
        ),
    },
    "en_multipla_escolha": {
        "stuff": (
            "Answer the multiple-choice question below. Use the context as support; if it does not contain "
            "the answer, use your own medical knowledge. " + _LETRA_PRIMEIRO + "\n\n"
            "Context:\n{contexto}\n\nQuestion: {pergunta}\nAnswer:"
        ),
        "refine_inicial": (
            "Answer the multiple-choice question below. Use the context as support; if it does not contain "
            "the answer, use your own medical knowledge. " + _LETRA_PRIMEIRO + "\n\n"
            "Context:\n{contexto}\n\nQuestion: {pergunta}\nAnswer:"
        ),
        "refine_passo": (
            "Your task is to refine an existing answer to the multiple-choice question below, only if the new "
            "context brings relevant information that is not in it yet. If the new context does not help, "
            "repeat the existing answer without changes. " + _LETRA_PRIMEIRO + "\n\n"
            "Original question: {pergunta}\n\nExisting answer: {resposta}\n\n"
            "New context:\n{contexto}\n\nRefined answer:"
        ),
        "map": (
            "Answer the multiple-choice question below using only the context below. If the context is not "
            "relevant to the question, say only 'There is no relevant information.' Otherwise, start your "
            "answer with the letter of the chosen option, followed by a short justification.\n\n"
            "Context:\n{contexto}\n\nQuestion: {pergunta}\nAnswer:"
        ),
        "reduce": (
            "Combine the partial answers below (generated from different excerpts of a document) into a "
            "single final answer to the original multiple-choice question. Ignore partial answers that say "
            "there is no relevant information; if all of them say so, use your own medical knowledge. "
            + _LETRA_PRIMEIRO + "\n\n"
            "Original question: {pergunta}\n\nPartial answers:\n{respostas}\n\nCombined final answer:"
        ),
        "confianca": (
            "Answer the multiple-choice question below. Use the context as support; if it does not contain "
            "the answer, use your own medical knowledge. Then rate your own confidence that the answer is "
            "correct. Answer EXACTLY in this format, in two lines:\n"
            "ANSWER: <the letter of the chosen option, followed by a short justification>\n"
            "CONFIDENCE: <integer from 0 to 100>\n\nContext:\n{contexto}\n\nQuestion: {pergunta}"
        ),
        "alternativas": (
            "Based on the context below and on the original multiple-choice question, generate exactly {n} "
            "alternative versions of the question stem, more specific and less ambiguous, that help clarify "
            "its intent. Do not include the answer options: they are added automatically. Answer ONLY with "
            "the {n} questions, one per line, numbered, with no additional text.\n\n"
            "Context:\n{contexto}\n\nOriginal question: {pergunta}"
        ),
        "combinacao": (
            "Combine the answers below (generated from variations of the original question) into a single "
            "final answer to the original multiple-choice question. " + _LETRA_PRIMEIRO + "\n\n"
            "Original question: {pergunta}\n\nAnswers to the variations:\n{respostas}\n\nCombined final answer:"
        ),
        "sem_rag": (
            "Answer the multiple-choice question below using your own medical knowledge. " + _LETRA_PRIMEIRO
            + "\n\nQuestion: {pergunta}\nAnswer:"
        ),
    },
}

# Nas questões de múltipla escolha, as alternativas vêm no fim da pergunta, a partir de "\n\nA. ".
_INICIO_ALTERNATIVAS = re.compile(r"\n\nA\. ")


def _prompt(nome: str, **campos) -> str:
    return PROMPTS[CONJUNTO_PROMPTS][nome].format(**campos)


def _montar_prompt_stuff(query: str, contextos: list[str]) -> str:
    return _prompt("stuff", contexto="\n\n---\n\n".join(contextos), pergunta=query)


def _gerar_via_ollama(prompt: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42},
    }
    if OLLAMA_THINK is not None:
        payload["think"] = OLLAMA_THINK.lower() == "true"
    # Em CPU, um modelo de ~9B leva bem mais que 2 min nos prompts longos do Refine/Step-Down.
    resposta = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=900)
    dados = resposta.json()
    return {
        "resposta_texto": dados.get("response", ""),
        "tokens_prompt": dados.get("prompt_eval_count", len(_encoder.encode(prompt))),
        "tokens_gerados": dados.get("eval_count", 0),
        "modelo_llm": OLLAMA_MODEL,
        "modelo_llm_versao": dados.get("model", OLLAMA_MODEL),
    }


def _gerar_via_api_openai(cliente: OpenAI, modelo: str, prompt: str, **parametros) -> dict:
    """Chamada no formato Chat Completions, compartilhada pela OpenAI e pela DeepSeek (que expõe
    a mesma API). `tokens_gerados` inclui os tokens de raciocínio, que também são cobrados."""
    resposta = cliente.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt}],
        **parametros,
    )
    # O OpenRouter informa qual provedor serviu a chamada; OpenAI e DeepSeek não têm esse campo.
    provedor = getattr(resposta, "provider", None)
    return {
        "resposta_texto": resposta.choices[0].message.content or "",
        "tokens_prompt": resposta.usage.prompt_tokens,
        "tokens_gerados": resposta.usage.completion_tokens,
        "modelo_llm": modelo,
        "modelo_llm_versao": f"{resposta.model}@{provedor}" if provedor else resposta.model,
    }


def _gerar_via_openai(prompt: str) -> dict:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()  # lê OPENAI_API_KEY do ambiente (.env)
    if _PARAMETROS_LLM is not None:
        parametros = _PARAMETROS_LLM
    elif OPENAI_REASONING_EFFORT:
        parametros = {"reasoning_effort": OPENAI_REASONING_EFFORT}
    else:
        parametros = {"temperature": 0.0}
    return _gerar_via_api_openai(_openai_client, OPENAI_MODEL, prompt, **parametros)


def _gerar_via_deepseek(prompt: str) -> dict:
    """Raciocínio desligado: no modo com raciocínio (o padrão da DeepSeek), temperature é ignorada."""
    global _deepseek_client
    if _deepseek_client is None:
        _deepseek_client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL)
    return _gerar_via_api_openai(
        _deepseek_client, DEEPSEEK_MODEL, prompt,
        temperature=0.0, extra_body={"thinking": {"type": "disabled"}},
    )


def _gerar_via_openrouter(prompt: str) -> dict:
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=OPENROUTER_BASE_URL)
    parametros = _PARAMETROS_LLM if _PARAMETROS_LLM is not None else {"temperature": 0.0, "extra_body": _SEM_RACIOCINIO}
    return _gerar_via_api_openai(_openrouter_client, OPENROUTER_MODEL, prompt, **parametros)


def _gerar_via_anthropic(prompt: str) -> dict:
    import backend_anthropic  # import tardio: o SDK da Anthropic só é exigido por quem usa Claude

    return backend_anthropic.gerar(prompt, ANTHROPIC_MODEL)


_BACKENDS_LLM = {
    "ollama": _gerar_via_ollama,
    "openai": _gerar_via_openai,
    "openrouter": _gerar_via_openrouter,
    "deepseek": _gerar_via_deepseek,
    "anthropic": _gerar_via_anthropic,
}


def configurar_llm(backend: str, modelo: str | None = None, reasoning_effort: str | None = None):
    """Troca o LLM gerador em tempo de execução -- pra rodar vários LLMs no mesmo script, com o
    mesmo índice vetorial, sem reencodar os documentos. `modelo` None mantém o do .env;
    `reasoning_effort` só vale pro backend openai (None = temperature 0)."""
    global LLM_BACKEND, OLLAMA_MODEL, OPENAI_MODEL, OPENROUTER_MODEL, DEEPSEEK_MODEL, ANTHROPIC_MODEL
    global OPENAI_REASONING_EFFORT, _PARAMETROS_LLM
    if backend not in _BACKENDS_LLM:
        raise ValueError(f"backend desconhecido: {backend} (opções: {', '.join(_BACKENDS_LLM)})")
    LLM_BACKEND = backend
    _PARAMETROS_LLM = None
    if modelo:
        if backend == "ollama":
            OLLAMA_MODEL = modelo
        elif backend == "openai":
            OPENAI_MODEL = modelo
        elif backend == "openrouter":
            OPENROUTER_MODEL = modelo
        elif backend == "deepseek":
            DEEPSEEK_MODEL = modelo
        else:
            ANTHROPIC_MODEL = modelo
    if backend == "openai":
        OPENAI_REASONING_EFFORT = reasoning_effort


def configurar_llm_por_rotulo(rotulo: str):
    """Configura um LLM de LLMS_DO_COMPARATIVO pelo rótulo (ex.: "claude_haiku"), com os parâmetros dele."""
    global _PARAMETROS_LLM
    llm = llm_por_rotulo(rotulo)
    configurar_llm(llm["backend"], llm["modelo"])
    _PARAMETROS_LLM = llm["parametros"]


def llm_por_rotulo(rotulo: str) -> dict:
    for llm in LLMS_DO_COMPARATIVO:
        if llm["rotulo"] == rotulo:
            return llm
    raise ValueError(f"rótulo desconhecido: {rotulo} (opções: {', '.join(l['rotulo'] for l in LLMS_DO_COMPARATIVO)})")


def _chamar_llm(prompt: str) -> dict:
    global _modelo_llm_versao
    resultado = _BACKENDS_LLM[LLM_BACKEND](prompt)
    _modelo_llm_versao = resultado.pop("modelo_llm_versao", None)
    return resultado


def gerar_resposta_stuff(query: str, contextos: list[str]) -> dict:
    """Método Stuff: injeta todos os contextos recuperados de uma vez no prompt (1 chamada ao LLM)."""
    prompt = _montar_prompt_stuff(query, contextos)
    return _chamar_llm(prompt)


def gerar_resposta_refine(query: str, contextos: list[str]) -> dict:
    """Método Refine: percorre os contextos em sequência, refinando a resposta a cada novo chunk
    (1 chamada ao LLM por contexto recuperado — mais chamadas e tokens que o Stuff)."""
    prompt_inicial = _prompt("refine_inicial", contexto=contextos[0], pergunta=query)
    resultado = _chamar_llm(prompt_inicial)
    resposta_atual = resultado["resposta_texto"]
    tokens_prompt_total = resultado["tokens_prompt"]
    tokens_gerados_total = resultado["tokens_gerados"]
    modelo_llm = resultado["modelo_llm"]

    for contexto in contextos[1:]:
        prompt_refine = _prompt("refine_passo", pergunta=query, resposta=resposta_atual, contexto=contexto)
        resultado = _chamar_llm(prompt_refine)
        resposta_atual = resultado["resposta_texto"]
        tokens_prompt_total += resultado["tokens_prompt"]
        tokens_gerados_total += resultado["tokens_gerados"]

    return {
        "resposta_texto": resposta_atual,
        "tokens_prompt": tokens_prompt_total,
        "tokens_gerados": tokens_gerados_total,
        "modelo_llm": modelo_llm,
    }


def gerar_resposta_map_reduce(query: str, contextos: list[str]) -> dict:
    """Método Map-Reduce: gera uma resposta parcial pra cada contexto isoladamente (map),
    depois combina todas as respostas parciais em uma resposta final (reduce). k+1 chamadas."""
    tokens_prompt_total = 0
    tokens_gerados_total = 0
    modelo_llm = None
    respostas_parciais = []

    for contexto in contextos:
        prompt_map = _prompt("map", contexto=contexto, pergunta=query)
        resultado = _chamar_llm(prompt_map)
        respostas_parciais.append(resultado["resposta_texto"])
        tokens_prompt_total += resultado["tokens_prompt"]
        tokens_gerados_total += resultado["tokens_gerados"]
        modelo_llm = resultado["modelo_llm"]

    respostas_concatenadas = "\n\n".join(f"- {r}" for r in respostas_parciais)
    prompt_reduce = _prompt("reduce", pergunta=query, respostas=respostas_concatenadas)
    resultado_final = _chamar_llm(prompt_reduce)
    tokens_prompt_total += resultado_final["tokens_prompt"]
    tokens_gerados_total += resultado_final["tokens_gerados"]

    return {
        "resposta_texto": resultado_final["resposta_texto"],
        "tokens_prompt": tokens_prompt_total,
        "tokens_gerados": tokens_gerados_total,
        "modelo_llm": modelo_llm,
    }


def _responder_com_confianca(query: str, contextos: list[str]) -> dict:
    """1 chamada ao LLM: responde e avalia a própria confiança (0-100) de que a resposta está
    correta e sustentada pelo contexto. Bloco compartilhado por Map-Rerank (por chunk) e
    Reciprocal RAG (por pergunta alternativa)."""
    contexto_concatenado = "\n\n---\n\n".join(contextos)
    prompt = _prompt("confianca", contexto=contexto_concatenado, pergunta=query)
    resultado = _chamar_llm(prompt)
    texto = resultado["resposta_texto"]
    match_resposta = re.search(r"(?:RESPOSTA|ANSWER):\s*(.*?)(?:\n\s*(?:CONFIAN|CONFIDENCE)|\Z)", texto, re.DOTALL | re.IGNORECASE)
    match_confianca = re.search(r"(?:CONFIAN[CÇ]A|CONFIDENCE):\s*(\d+)", texto, re.IGNORECASE)
    resultado["resposta_texto"] = match_resposta.group(1).strip() if match_resposta else texto.strip()
    resultado["confianca"] = int(match_confianca.group(1)) if match_confianca else 0
    return resultado


def gerar_resposta_map_rerank(query: str, contextos: list[str]) -> dict:
    """Método Map-Rerank: gera uma resposta + nota de confiança pra cada contexto
    isoladamente, e mantém só a resposta de maior confiança. k chamadas."""
    tokens_prompt_total = 0
    tokens_gerados_total = 0
    modelo_llm = None
    candidatos = []  # (confianca, resposta_texto)

    for contexto in contextos:
        resultado = _responder_com_confianca(query, [contexto])
        tokens_prompt_total += resultado["tokens_prompt"]
        tokens_gerados_total += resultado["tokens_gerados"]
        modelo_llm = resultado["modelo_llm"]
        candidatos.append((resultado["confianca"], resultado["resposta_texto"]))

    _, melhor_resposta = max(candidatos, key=lambda c: c[0])

    return {
        "resposta_texto": melhor_resposta,
        "tokens_prompt": tokens_prompt_total,
        "tokens_gerados": tokens_gerados_total,
        "modelo_llm": modelo_llm,
    }


def _gerar_perguntas_alternativas(query: str, contexto_semente: list[str], n: int = 4) -> dict:
    """'Generator chain' (Şakar & Emekci 2025, seç. 2.5, GCi=1 chamada): gera n perguntas
    alternativas, mais específicas e menos ambíguas, a partir da pergunta original e de um
    contexto semente (recuperado com a própria pergunta original)."""
    contexto_concatenado = "\n\n---\n\n".join(contexto_semente)
    prompt = _prompt("alternativas", n=n, contexto=contexto_concatenado, pergunta=query)
    resultado = _chamar_llm(prompt)
    linhas = [linha.strip() for linha in resultado["resposta_texto"].splitlines() if linha.strip()]
    perguntas = [re.sub(r"^\d+[\.\):]\s*", "", linha).strip() for linha in linhas][:n]
    # Múltipla escolha: o LLM reescreve só o enunciado, e as alternativas da pergunta original são
    # coladas em cada versão -- assim cada resposta intermediária também escolhe uma letra.
    inicio_alternativas = _INICIO_ALTERNATIVAS.search(query)
    if CONJUNTO_PROMPTS == "en_multipla_escolha" and inicio_alternativas:
        perguntas = [f"{p}{query[inicio_alternativas.start():]}" for p in perguntas]
    resultado["perguntas_alternativas"] = perguntas if perguntas else [query]
    return resultado


def gerar_resposta_query_step_down(query: str, colecao, k: int = 4, n_alternativas: int = 4) -> dict:
    """Query Step-Down (Şakar & Emekci 2025, seç. 2.5.2): gera n perguntas alternativas a partir
    de um contexto semente (1 chamada), responde cada uma isoladamente com Stuff (n chamadas —
    até aqui, GCi+Qj = 1+n chamadas, igual ao exemplo do paper), e combina as n respostas em uma
    resposta final coerente (+1 chamada nossa: o paper não deixa 100% explícito se essa
    combinação usa uma chamada de LLM à parte ou não — decidimos que sim, para produzir uma
    resposta final coerente; é uma escolha de implementação documentada, não uma certeza)."""
    contexto_semente = recuperar(colecao, query, k=k)
    resultado_gerador = _gerar_perguntas_alternativas(query, contexto_semente, n=n_alternativas)

    tokens_prompt_total = resultado_gerador["tokens_prompt"]
    tokens_gerados_total = resultado_gerador["tokens_gerados"]
    modelo_llm = resultado_gerador["modelo_llm"]

    respostas_por_pergunta = []
    for pergunta_alt in resultado_gerador["perguntas_alternativas"]:
        contexto_alt = recuperar(colecao, pergunta_alt, k=k)
        resultado = gerar_resposta_stuff(pergunta_alt, contexto_alt)
        respostas_por_pergunta.append(resultado["resposta_texto"])
        tokens_prompt_total += resultado["tokens_prompt"]
        tokens_gerados_total += resultado["tokens_gerados"]

    respostas_concatenadas = "\n\n".join(f"- {r}" for r in respostas_por_pergunta)
    prompt_combinacao = _prompt("combinacao", pergunta=query, respostas=respostas_concatenadas)
    resultado_final = _chamar_llm(prompt_combinacao)
    tokens_prompt_total += resultado_final["tokens_prompt"]
    tokens_gerados_total += resultado_final["tokens_gerados"]

    return {
        "resposta_texto": resultado_final["resposta_texto"],
        "tokens_prompt": tokens_prompt_total,
        "tokens_gerados": tokens_gerados_total,
        "modelo_llm": modelo_llm,
        "perguntas_alternativas": resultado_gerador["perguntas_alternativas"],
    }


def gerar_resposta_reciprocal(query: str, colecao, k: int = 4, n_alternativas: int = 4) -> dict:
    """Reciprocal RAG (Şakar & Emekci 2025, seç. 2.5.3): mesmo primeiro passo do Query Step-Down
    (n perguntas alternativas a partir de um contexto semente), mas em vez de combinar as n
    respostas, avalia a confiança de cada uma (mesmo mecanismo do Map-Rerank) e mantém só a mais
    confiável — sem chamada extra de combinação. O próprio paper contrasta os dois métodos por
    esse motivo: Step-Down agrega, Reciprocal só filtra e retém a mais pertinente."""
    contexto_semente = recuperar(colecao, query, k=k)
    resultado_gerador = _gerar_perguntas_alternativas(query, contexto_semente, n=n_alternativas)

    tokens_prompt_total = resultado_gerador["tokens_prompt"]
    tokens_gerados_total = resultado_gerador["tokens_gerados"]
    modelo_llm = resultado_gerador["modelo_llm"]
    candidatos = []

    for pergunta_alt in resultado_gerador["perguntas_alternativas"]:
        contexto_alt = recuperar(colecao, pergunta_alt, k=k)
        resultado = _responder_com_confianca(pergunta_alt, contexto_alt)
        tokens_prompt_total += resultado["tokens_prompt"]
        tokens_gerados_total += resultado["tokens_gerados"]
        candidatos.append((resultado["confianca"], resultado["resposta_texto"]))

    _, melhor_resposta = max(candidatos, key=lambda c: c[0])

    return {
        "resposta_texto": melhor_resposta,
        "tokens_prompt": tokens_prompt_total,
        "tokens_gerados": tokens_gerados_total,
        "modelo_llm": modelo_llm,
        "perguntas_alternativas": resultado_gerador["perguntas_alternativas"],
    }


def gerar_resposta_sem_rag(query: str, contextos: list[str]) -> dict:
    """Grupo de controle: a mesma pergunta, sem nenhum trecho recuperado (1 chamada ao LLM). Sem ele
    não dá pra separar o que o RAG acrescentou do que o modelo já sabia (ex.: questões do MedQA)."""
    return _chamar_llm(_prompt("sem_rag", pergunta=query))


_METODOS_RAG = {
    "sem_rag": gerar_resposta_sem_rag,
    "stuff": gerar_resposta_stuff,
    "refine": gerar_resposta_refine,
    "map_reduce": gerar_resposta_map_reduce,
    "map_rerank": gerar_resposta_map_rerank,
}

# Métodos que fazem a própria recuperação internamente (várias vezes: contexto semente + 1 por
# pergunta alternativa), em vez de receber um `contextos` já pronto — por isso ficam fora do
# dict acima e são despachados à parte em `rodar_experimento`.
_METODOS_NIVEL_QUERY = {
    "query_step_down": gerar_resposta_query_step_down,
    "reciprocal": gerar_resposta_reciprocal,
}


def calcular_similaridade(resposta_gerada: str, resposta_esperada: str) -> float:
    modelo = get_embedding_model()
    vetores = _encode_documentos(modelo, [resposta_gerada, resposta_esperada])
    return float(cosine_similarity([vetores[0]], [vetores[1]])[0][0])


def rodar_experimento(
    query: str,
    colecao,
    document_id: str,
    resposta_esperada: str | None = None,
    k: int = 4,
    metodo: str = "stuff",
    log_path: Path = METRICAS_DIR / "pilot_stuff.jsonl",
    extra_campos: dict | None = None,
    limiar_compressao: float = 0.0,
):
    """`limiar_compressao`: filtro de compressão contextual (0.0 = sem filtro, mantém todos os k
    chunks; > 0 descarta chunks com similaridade abaixo do limiar). Só se aplica aos métodos que
    recebem um `contextos` pronto (Stuff/Refine/Map-Reduce/Map-Rerank) -- Query Step-Down e
    Reciprocal fazem recuperações próprias internamente e não passam por aqui."""
    inicio_total = time.time()
    eh_metodo_nivel_query = metodo in _METODOS_NIVEL_QUERY
    # Query Step-Down e Reciprocal fazem sua própria recuperação (várias vezes: contexto semente
    # + 1 por pergunta alternativa), então não há um único `contextos` prévio pra medir aqui —
    # o tempo dessas recuperações internas fica contabilizado dentro de tempo_geracao_s.
    if eh_metodo_nivel_query:
        contextos = None
        ids_recuperados = None
    elif metodo == "sem_rag":
        contextos = []
        ids_recuperados = None
    else:
        busca = consultar_indice(colecao, query, k=k)
        chunks_com_similaridade = list(zip(busca["documentos"], busca["similaridades"]))
        contextos = aplicar_compressao_contextual(chunks_com_similaridade, limiar_compressao)
        # Ids "<document_id>#<chunk_index>" do top-k antes da compressão: é o que permite calcular
        # Recall@k contra o documento-ouro de cada pergunta, sem juiz LLM.
        ids_recuperados = busca["ids"]
    tempo_recuperacao_s = round(time.time() - inicio_total, 3)

    cpu_antes = psutil.cpu_percent(interval=None)
    mem_antes = psutil.virtual_memory().percent
    inicio_geracao = time.time()

    if eh_metodo_nivel_query:
        resultado_geracao = _METODOS_NIVEL_QUERY[metodo](query, colecao, k=k)
    else:
        resultado_geracao = _METODOS_RAG[metodo](query, contextos)

    resultado_geracao["tempo_geracao_s"] = round(time.time() - inicio_geracao, 3)
    resultado_geracao["cpu_percent"] = round((cpu_antes + psutil.cpu_percent(interval=None)) / 2, 2)
    resultado_geracao["mem_percent"] = round((mem_antes + psutil.virtual_memory().percent) / 2, 2)

    similaridade = (
        calcular_similaridade(resultado_geracao["resposta_texto"], resposta_esperada)
        if resposta_esperada
        else None
    )

    registro = {
        "experimento_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "document_id": document_id,
        "metodo_rag": metodo,
        "banco_vetorial": "chromadb",
        "modelo_embedding": EMBEDDING_MODEL_NAME,
        "llm_backend": LLM_BACKEND,
        "modelo_llm": resultado_geracao["modelo_llm"],
        "modelo_llm_versao": _modelo_llm_versao,
        "compressao_contextual": {"limiar": limiar_compressao, "n_chunks_mantidos": len(contextos) if contextos is not None else None},
        "pergunta": query,
        "contextos_recuperados": contextos if contextos is not None else resultado_geracao.get("perguntas_alternativas"),
        "ids_recuperados": ids_recuperados,
        "resposta_gerada": resultado_geracao["resposta_texto"],
        "resposta_esperada": resposta_esperada,
        "similaridade_cosseno": similaridade,
        "tokens_prompt": resultado_geracao["tokens_prompt"],
        "tokens_gerados": resultado_geracao["tokens_gerados"],
        "tokens_totais": resultado_geracao["tokens_prompt"] + resultado_geracao["tokens_gerados"],
        "tempo_recuperacao_s": tempo_recuperacao_s,
        "tempo_geracao_s": resultado_geracao["tempo_geracao_s"],
        "tempo_total_s": round(time.time() - inicio_total, 3),
        "cpu_percent": resultado_geracao["cpu_percent"],
        "mem_percent": resultado_geracao["mem_percent"],
    }
    if extra_campos:
        registro.update(extra_campos)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    return registro


if __name__ == "__main__":
    # Teste de fumaça com um texto placeholder — troque por dados/brutos/llama2_arxiv.txt
    # assim que o dataset piloto estiver definido e baixado.
    texto_exemplo = BASE_DIR / "dados" / "brutos" / "_exemplo.txt"
    texto_exemplo.parent.mkdir(parents=True, exist_ok=True)
    if not texto_exemplo.exists():
        texto_exemplo.write_text(
            "O Trabalho de Conclusão de Curso (TCC) é uma etapa fundamental na formação acadêmica. "
            "Ele permite ao aluno aplicar os conhecimentos adquiridos durante o curso em um projeto "
            "prático ou teórico. Para avaliar um sistema de RAG, métricas comuns incluem similaridade "
            "de cosseno, tempo de resposta, uso de CPU e memória, e contagem de tokens.",
            encoding="utf-8",
        )

    chunks = carregar_e_chunkar(str(texto_exemplo), document_id="exemplo")
    colecao = construir_indice(chunks, colecao_nome="piloto_exemplo")

    registro = rodar_experimento(
        query="Quais métricas eu posso usar para avaliar meu sistema?",
        colecao=colecao,
        document_id="exemplo",
        resposta_esperada="Similaridade de cosseno, tempo de resposta, uso de CPU e memória, e tokens.",
    )

    print(json.dumps(registro, indent=2, ensure_ascii=False))
