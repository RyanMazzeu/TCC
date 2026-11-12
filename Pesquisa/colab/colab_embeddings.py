"""
Roda NO GOOGLE COLAB (com GPU): calcula os embeddings BGE-M3 dos fragmentos exportados por
scripts/exportar_fragmentos.py e empacota tudo em embeddings_colab.zip, no formato que
scripts/benchmarks.py lê (blocos de 512 vetores + manifesto). Leva ~10-20 min numa T4.

No Colab:
  1. Ambiente de execução > Alterar o tipo de ambiente de execução > GPU (T4).
  2. Envie este arquivo e os fragmentos_*.jsonl.gz pelo painel Arquivos (ícone de pasta, à esquerda).
  3. Numa célula:  !pip install -q FlagEmbedding==1.4.0
  4. Noutra célula: !python colab_embeddings.py
  5. Baixe o embeddings_colab.zip pelo painel Arquivos (três pontinhos > Fazer download).
"""

import glob
import gzip
import hashlib
import json
import os
import shutil
import time

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

BLOCO = 512  # o mesmo de benchmarks._BLOCO_EMBEDDINGS
SAIDA = "embeddings_colab"

assert torch.cuda.is_available(), "Sem GPU: troque o tipo de ambiente de execução para GPU (T4)."
modelo = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

for arquivo in sorted(glob.glob("fragmentos_*.jsonl.gz")):
    with gzip.open(arquivo, "rt", encoding="utf-8") as f:
        cabecalho = json.loads(f.readline())
        textos = [json.loads(linha) for linha in f]
    assert len(textos) == cabecalho["n_fragmentos"], f"{arquivo} incompleto"
    pasta = os.path.join(SAIDA, cabecalho["colecao"])
    os.makedirs(pasta, exist_ok=True)
    inicio_relogio = time.time()
    for n_bloco, inicio in enumerate(range(0, len(textos), BLOCO)):
        vetores = modelo.encode(textos[inicio : inicio + BLOCO], batch_size=64, return_dense=True,
                                return_sparse=False, return_colbert_vecs=False)["dense_vecs"]
        np.save(os.path.join(pasta, f"bloco_{n_bloco:05d}.npy"), np.asarray(vetores, dtype=np.float16))
        if n_bloco % 20 == 0:
            print(f"{cabecalho['dataset']}: {inicio + len(vetores)}/{len(textos)} ({time.time() - inicio_relogio:.0f}s)", flush=True)
    assinatura = hashlib.sha256("\x00".join(textos).encode("utf-8")).hexdigest()
    with open(os.path.join(pasta, "manifesto.json"), "w") as f:
        json.dump({"n_fragmentos": len(textos), "assinatura": assinatura, "bloco": BLOCO}, f)
    print(f"{cabecalho['dataset']}: pronto, {len(textos)} vetores em {time.time() - inicio_relogio:.0f}s", flush=True)

shutil.make_archive(SAIDA, "zip", SAIDA)
print(f"Baixe o arquivo {SAIDA}.zip ({os.path.getsize(SAIDA + '.zip') / 1e6:.0f} MB)")
