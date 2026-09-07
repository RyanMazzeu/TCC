"""
Primeiro comparativo real: Stuff vs Refine, nos 3 domínios PT-BR já pilotados
(Revalida, Grendene, paper IFES/SBC). Reaproveita os mesmos documentos e
perguntas dos pilotos individuais — só varia o método de RAG.
"""

import json
import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import pipeline_base as pb

DADOS_DIR = pb.BASE_DIR / "dados"
LOG_PATH = pb.METRICAS_DIR / "comparativo_6_metodos_v2.jsonl"
METODOS = ["stuff", "refine", "map_reduce", "map_rerank", "query_step_down", "reciprocal"]


def carregar_revalida():
    dataset = json.loads((DADOS_DIR / "processados" / "revalida_2024_2_completo.json").read_text(encoding="utf-8"))
    nomes_arquivos = {
        "codigo_etica_medica_cfm": "cfm_codigo_etica",
        "sbp_consulta_adolescente": "sbp_consulta_adolescente",
        "revalida_ref_tev_scielo": "tev_scielo",
        "revalida_ref_parto_febrasgo": "parto_febrasgo",
        "revalida_ref_meningococica_ms": "meningococica_ms",
    }
    chunks = []
    for arquivo, document_id in nomes_arquivos.items():
        chunks += pb.carregar_e_chunkar(str(DADOS_DIR / "brutos" / f"{arquivo}.txt"), document_id=document_id)
    perguntas = [
        {"id": item["id"], "query": f"{item['vinheta']}\n\n{item['pergunta']}", "esperada": item["resposta_esperada"]}
        for item in dataset["perguntas"]
    ]
    return "revalida", chunks, perguntas


def carregar_grendene():
    dataset = json.loads((DADOS_DIR / "processados" / "grendene_itr_3t2025.json").read_text(encoding="utf-8"))
    chunks = pb.carregar_e_chunkar(str(DADOS_DIR / "brutos" / "grendene_itr_3t2025.txt"), document_id="grendene_itr_3t2025")
    perguntas = [{"id": item["id"], "query": item["pergunta"], "esperada": item["resposta_esperada"]} for item in dataset["perguntas"]]
    return "grendene", chunks, perguntas


def carregar_sbc():
    dataset = json.loads((DADOS_DIR / "processados" / "sbc_embeddings_llms.json").read_text(encoding="utf-8"))
    chunks = pb.carregar_e_chunkar(str(DADOS_DIR / "brutos" / "sbc_embeddings_llms.txt"), document_id="sbc_embeddings_llms")
    perguntas = [{"id": item["id"], "query": item["pergunta"], "esperada": item["resposta_esperada"]} for item in dataset["perguntas"]]
    return "sbc", chunks, perguntas


def main():
    dominios = [carregar_revalida(), carregar_grendene(), carregar_sbc()]

    for nome, chunks, perguntas in dominios:
        colecao = pb.construir_indice(chunks, colecao_nome=f"comparativo_{nome}")
        for metodo in METODOS:
            for p in perguntas:
                registro = pb.rodar_experimento(
                    query=p["query"],
                    colecao=colecao,
                    document_id=nome,
                    resposta_esperada=p["esperada"],
                    metodo=metodo,
                    log_path=LOG_PATH,
                    extra_campos={"dominio": nome, "pergunta_id": p["id"]},
                )
                print(
                    f"[{nome}/{metodo}] {p['id']}: sim={registro['similaridade_cosseno']:.4f} "
                    f"tokens={registro['tokens_totais']} t_ger={registro['tempo_geracao_s']:.2f}s"
                )


if __name__ == "__main__":
    main()
