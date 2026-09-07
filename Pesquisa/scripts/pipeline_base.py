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

# "ollama" (local, grátis) ou "openai" (pago por token) — trocar via .env, sem mexer no código.
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_openai_client = None

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
        if _EH_MULTILINGUE:
            _embedding_model = BGEM3FlagModel(EMBEDDING_MODEL_NAME, use_fp16=True)
        else:
            _embedding_model = FlagModel(
                EMBEDDING_MODEL_NAME,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
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
    texto = Path(caminho_documento).read_text(encoding="utf-8")
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


def construir_indice(chunks: list[dict], colecao_nome: str, vetores=None):
    """Cria (ou recria) uma coleção Chroma persistente a partir dos chunks.

    `vetores` é opcional: passe embeddings já calculados (ex.: em benchmarks que precisam medir
    só o tempo de indexação em si, sem pagar o custo de reencodar tudo de novo)."""
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    try:
        client.delete_collection(colecao_nome)
    except Exception:
        pass
    colecao = client.create_collection(colecao_nome)

    textos = [c["text"] for c in chunks]
    if vetores is None:
        modelo = get_embedding_model()
        vetores = _encode_documentos(modelo, textos)

    colecao.add(
        ids=[f"{c['document_id']}#{c['chunk_index']}" for c in chunks],
        embeddings=vetores.tolist(),
        documents=textos,
        metadatas=[{"document_id": c["document_id"], "chunk_index": c["chunk_index"]} for c in chunks],
    )
    return colecao


def recuperar(colecao, query: str, k: int = 4, vetor_query=None) -> list[str]:
    """`vetor_query` é opcional: passe um embedding de pergunta já calculado (mesmo motivo do
    parâmetro `vetores` em construir_indice)."""
    if vetor_query is None:
        modelo = get_embedding_model()
        vetor_query = _encode_query(modelo, query)
    resultado = colecao.query(query_embeddings=vetor_query.tolist(), n_results=k)
    return resultado["documents"][0]


def recuperar_com_similaridade(colecao, query: str, k: int = 4, vetor_query=None) -> list[tuple[str, float]]:
    """Igual a `recuperar`, mas devolve também uma similaridade de cosseno aproximada de cada
    chunk com a pergunta, derivada da distância L2 que o Chroma já calcula internamente
    (cos_sim = 1 - L2²/2, válido para vetores normalizados -- é o caso do BGE). Evita reencodar
    documentos só para pontuar similaridade. Usado pelo filtro de compressão contextual."""
    if vetor_query is None:
        modelo = get_embedding_model()
        vetor_query = _encode_query(modelo, query)
    resultado = colecao.query(query_embeddings=vetor_query.tolist(), n_results=k)
    documentos = resultado["documents"][0]
    similaridades = [1 - d / 2 for d in resultado["distances"][0]]
    return list(zip(documentos, similaridades))


def aplicar_compressao_contextual(chunks_com_similaridade: list[tuple[str, float]], limiar: float) -> list[str]:
    """Filtro de compressão contextual: descarta chunks com similaridade abaixo do limiar
    (limiar=0.0 equivale a "sem filtro", mantém todos os k chunks recuperados). Nunca deixa a
    lista vazia -- se todos ficarem abaixo do limiar, mantém ao menos o mais similar."""
    filtrados = [texto for texto, sim in chunks_com_similaridade if sim >= limiar]
    return filtrados if filtrados else [chunks_com_similaridade[0][0]]


def _montar_prompt_stuff(query: str, contextos: list[str]) -> str:
    contexto_concatenado = "\n\n---\n\n".join(contextos)
    return (
        "Responda à pergunta usando apenas o contexto abaixo. "
        "Se a resposta não estiver no contexto, diga que não sabe.\n\n"
        f"Contexto:\n{contexto_concatenado}\n\nPergunta: {query}\nResposta:"
    )


def _gerar_via_ollama(prompt: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42},
    }
    resposta = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
    dados = resposta.json()
    return {
        "resposta_texto": dados.get("response", ""),
        "tokens_prompt": dados.get("prompt_eval_count", len(_encoder.encode(prompt))),
        "tokens_gerados": dados.get("eval_count", 0),
        "modelo_llm": OLLAMA_MODEL,
    }


def _gerar_via_openai(prompt: str) -> dict:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()  # lê OPENAI_API_KEY do ambiente (.env)

    resposta = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return {
        "resposta_texto": resposta.choices[0].message.content or "",
        "tokens_prompt": resposta.usage.prompt_tokens,
        "tokens_gerados": resposta.usage.completion_tokens,
        "modelo_llm": OPENAI_MODEL,
    }


def _chamar_llm(prompt: str) -> dict:
    if LLM_BACKEND == "openai":
        return _gerar_via_openai(prompt)
    return _gerar_via_ollama(prompt)


def gerar_resposta_stuff(query: str, contextos: list[str]) -> dict:
    """Método Stuff: injeta todos os contextos recuperados de uma vez no prompt (1 chamada ao LLM)."""
    prompt = _montar_prompt_stuff(query, contextos)
    return _chamar_llm(prompt)


def gerar_resposta_refine(query: str, contextos: list[str]) -> dict:
    """Método Refine: percorre os contextos em sequência, refinando a resposta a cada novo chunk
    (1 chamada ao LLM por contexto recuperado — mais chamadas e tokens que o Stuff)."""
    prompt_inicial = (
        "Responda à pergunta usando o contexto abaixo. Se a resposta não estiver no contexto, "
        f"diga que não sabe.\n\nContexto:\n{contextos[0]}\n\nPergunta: {query}\nResposta:"
    )
    resultado = _chamar_llm(prompt_inicial)
    resposta_atual = resultado["resposta_texto"]
    tokens_prompt_total = resultado["tokens_prompt"]
    tokens_gerados_total = resultado["tokens_gerados"]
    modelo_llm = resultado["modelo_llm"]

    for contexto in contextos[1:]:
        prompt_refine = (
            "Sua tarefa é refinar uma resposta existente, apenas se o novo contexto trouxer "
            "informação relevante que ainda não esteja nela. Se o novo contexto não ajudar, "
            "repita a resposta existente sem mudanças.\n\n"
            f"Pergunta original: {query}\n\nResposta existente: {resposta_atual}\n\n"
            f"Novo contexto:\n{contexto}\n\nResposta refinada:"
        )
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
        prompt_map = (
            "Responda à pergunta usando apenas o contexto abaixo. Se a resposta não estiver no "
            "contexto, diga apenas 'Não há informação relevante.'\n\n"
            f"Contexto:\n{contexto}\n\nPergunta: {query}\nResposta:"
        )
        resultado = _chamar_llm(prompt_map)
        respostas_parciais.append(resultado["resposta_texto"])
        tokens_prompt_total += resultado["tokens_prompt"]
        tokens_gerados_total += resultado["tokens_gerados"]
        modelo_llm = resultado["modelo_llm"]

    respostas_concatenadas = "\n\n".join(f"- {r}" for r in respostas_parciais)
    prompt_reduce = (
        "Combine as respostas parciais abaixo (geradas a partir de diferentes trechos de um "
        "documento) em uma única resposta final, coerente e não repetitiva, para a pergunta "
        "original. Ignore respostas parciais que digam não haver informação relevante.\n\n"
        f"Pergunta original: {query}\n\nRespostas parciais:\n{respostas_concatenadas}\n\n"
        "Resposta final combinada:"
    )
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
    prompt = (
        "Responda à pergunta usando apenas o contexto abaixo, e avalie sua própria confiança "
        "de que a resposta está correta e plenamente sustentada pelo contexto. Responda "
        "EXATAMENTE neste formato, em duas linhas:\nRESPOSTA: <sua resposta>\n"
        "CONFIANCA: <número inteiro de 0 a 100>\n\n"
        f"Contexto:\n{contexto_concatenado}\n\nPergunta: {query}"
    )
    resultado = _chamar_llm(prompt)
    texto = resultado["resposta_texto"]
    match_resposta = re.search(r"RESPOSTA:\s*(.*?)(?:\n\s*CONFIAN|\Z)", texto, re.DOTALL | re.IGNORECASE)
    match_confianca = re.search(r"CONFIAN[CÇ]A:\s*(\d+)", texto, re.IGNORECASE)
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
    prompt = (
        f"Com base no contexto abaixo e na pergunta original do usuário, gere exatamente {n} "
        "perguntas alternativas, mais específicas e menos ambíguas, que ajudem a esclarecer a "
        f"intenção da pergunta original. Responda APENAS com as {n} perguntas, uma por linha, "
        "numeradas, sem texto adicional.\n\n"
        f"Contexto:\n{contexto_concatenado}\n\nPergunta original: {query}"
    )
    resultado = _chamar_llm(prompt)
    linhas = [linha.strip() for linha in resultado["resposta_texto"].splitlines() if linha.strip()]
    perguntas = [re.sub(r"^\d+[\.\):]\s*", "", linha).strip() for linha in linhas][:n]
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
    prompt_combinacao = (
        "Combine as respostas abaixo (geradas a partir de variações da pergunta original) em uma "
        "única resposta final, coerente e não repetitiva, para a pergunta original.\n\n"
        f"Pergunta original: {query}\n\nRespostas das variações:\n{respostas_concatenadas}\n\n"
        "Resposta final combinada:"
    )
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


_METODOS_RAG = {
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
    else:
        chunks_com_similaridade = recuperar_com_similaridade(colecao, query, k=k)
        contextos = aplicar_compressao_contextual(chunks_com_similaridade, limiar_compressao)
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
        "modelo_llm": resultado_geracao["modelo_llm"],
        "compressao_contextual": {"limiar": limiar_compressao, "n_chunks_mantidos": len(contextos) if contextos is not None else None},
        "pergunta": query,
        "contextos_recuperados": contextos if contextos is not None else resultado_geracao.get("perguntas_alternativas"),
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
