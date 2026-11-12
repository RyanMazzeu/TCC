"""
Carrega os benchmarks de dados/processados/ (gerados por preparar_benchmarks.py) no formato que
pipeline_base.rodar_experimento usa, guarda o índice vetorial de cada um e sorteia as amostras de
perguntas das duas camadas do experimento (ver plano_execucao_benchmarks.md).
"""

import hashlib
import json
import random
import time

import chromadb
import numpy as np

import pipeline_base as pb

DADOS_DIR = pb.BASE_DIR / "dados"
# Docugami e MedQA: dois dos três datasets do paper-âncora (Şakar & Emekci 2025). FaQuAD: o dataset em
# português, com perguntas e respostas escritas por pessoas sobre textos originais em PT-BR.
DATASETS = ["docugami_10q", "medqa", "faquad"]
# Conjunto de prompts de cada dataset (pipeline_base.PROMPTS).
CONJUNTO_PROMPTS = {"docugami_10q": "en", "medqa": "en_multipla_escolha", "faquad": "pt"}
# Datasets em que o gabarito marca de onde sai a resposta (documento ou parágrafo): só neles dá
# Recall@k. O MedQA não marca, e a busca dele é avaliada pela Context Precision/Recall do RAGAS.
DATASETS_COM_TRECHO_CERTO = ["docugami_10q", "faquad"]

# Desenho em duas camadas: a principal (5 modelos econômicos) usa 100 perguntas por dataset, e a de
# generalização (os 7 modelos) usa as 25 primeiras dessas 100 -- todos os 7 respondem às mesmas 25.
PERGUNTAS_POR_CAMADA = {"principal": 100, "generalizacao": 25}
SEED_AMOSTRA = 42

# Os índices dos benchmarks ficam fora do git: o do MedQA tem dezenas de milhares de vetores.
VECTORSTORE_DIR = pb.BASE_DIR / "vectorstores_benchmarks"
# Embeddings calculados em blocos e guardados em disco: o MedQA leva horas em CPU, e uma queda no
# meio não pode obrigar a recomeçar do zero.
_BLOCO_EMBEDDINGS = 512


def _ler_dataset(nome: str) -> dict:
    return json.loads((DADOS_DIR / "processados" / f"{nome}.json").read_text(encoding="utf-8"))


def carregar_perguntas(nome: str) -> list[dict]:
    """Só as perguntas, sem fragmentar o corpus (o do MedQA leva alguns segundos)."""
    return _ler_dataset(nome)["perguntas"]


def carregar(nome: str) -> tuple[list[dict], list[dict]]:
    dados = _ler_dataset(nome)
    if "corpus" in dados:
        chunks = [c for d in dados["corpus"] for c in pb.chunkar_texto(d["texto"], d["document_id"])]
    else:
        chunks = [
            c for d in dados["documentos"]
            for c in pb.carregar_e_chunkar(str(DADOS_DIR / "brutos" / d["arquivo"]), d["document_id"])
        ]
    return chunks, dados["perguntas"]


def amostra(perguntas: list[dict], n: int) -> list[dict]:
    """As n primeiras perguntas de uma permutação fixa (seed 42), na ordem original do dataset.
    Como a permutação é a mesma pra qualquer n, a amostra de 25 está sempre dentro da de 100.
    n=0 ou n >= total: todas."""
    if n <= 0 or n >= len(perguntas):
        return perguntas
    permutacao = random.Random(SEED_AMOSTRA).sample(range(len(perguntas)), len(perguntas))
    return [perguntas[i] for i in sorted(permutacao[:n])]


def _nome_colecao(nome: str) -> str:
    """O nome inclui embedding e tamanho de fragmento, pra uma troca de configuração nunca
    reaproveitar um índice velho."""
    modelo = pb.EMBEDDING_MODEL_NAME.split("/")[-1].lower()
    return f"bench_{nome}_{modelo}_c{pb.CHUNK_SIZE}"


def assinatura_fragmentos(textos: list[str]) -> str:
    """Hash dos textos dos fragmentos, na ordem: garante que vetores calculados em outra máquina (ex.:
    no Colab, por colab/colab_embeddings.py) correspondem exatamente aos fragmentos daqui."""
    return hashlib.sha256("\x00".join(textos).encode("utf-8")).hexdigest()


def _vetores_em_blocos(chunks: list[dict], cache_dir) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifesto = cache_dir / "manifesto.json"
    assinatura = assinatura_fragmentos([c["text"] for c in chunks])
    if manifesto.exists():
        esperado = json.loads(manifesto.read_text(encoding="utf-8"))
        if esperado["assinatura"] != assinatura or esperado["n_fragmentos"] != len(chunks):
            raise RuntimeError(f"Os vetores em {cache_dir} são de outros fragmentos (mudou o corpus ou a "
                               "fragmentação): apague a pasta pra recalcular.")
    else:
        manifesto.write_text(json.dumps({"n_fragmentos": len(chunks), "assinatura": assinatura,
                                         "bloco": _BLOCO_EMBEDDINGS}), encoding="utf-8")
    modelo = None
    blocos, inicio_relogio = [], time.time()
    for n_bloco, inicio in enumerate(range(0, len(chunks), _BLOCO_EMBEDDINGS)):
        arquivo = cache_dir / f"bloco_{n_bloco:05d}.npy"
        if not arquivo.exists():
            modelo = modelo or pb.get_embedding_model()
            textos = [c["text"] for c in chunks[inicio : inicio + _BLOCO_EMBEDDINGS]]
            np.save(arquivo, pb._encode_documentos(modelo, textos))
            feitos = inicio + len(textos)
            print(f"  embeddings: {feitos}/{len(chunks)} ({time.time() - inicio_relogio:.0f}s nesta execução)", flush=True)
        # Os blocos vindos do Colab estão em float16, pra baixar metade do tamanho.
        blocos.append(np.load(arquivo).astype(np.float32))
    return np.concatenate(blocos)


def obter_indice(nome: str, chunks: list[dict]):
    """Reaproveita a coleção do Chroma se ela já tem esses chunks; senão calcula os embeddings (com
    retomada pelo cache em disco) e cria a coleção."""
    colecao_nome = _nome_colecao(nome)
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    try:
        colecao = client.get_collection(colecao_nome)
        if colecao.count() == len(chunks):
            return colecao
    except Exception:
        pass
    print(f"[{nome}] criando o índice: {len(chunks)} fragmentos")
    vetores = _vetores_em_blocos(chunks, VECTORSTORE_DIR / "_cache_embeddings" / colecao_nome)
    return pb.construir_indice(chunks, colecao_nome=colecao_nome, vetores=vetores, diretorio=VECTORSTORE_DIR)


def documento_do_chunk(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0]
