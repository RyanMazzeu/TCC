"""
Roda os 6 métodos de RAG nas perguntas do gabarito humano do CFM (Ryan + Afonso, conforme
disponível -- ver converter_gabarito_cfm.py). Usa o mesmo corpus combinado de 5 documentos do
domínio "revalida" (o Código de Ética Médica já está nele), já que o objetivo é justamente
testar se o retriever acha o documento certo entre vários.
"""

import json
import os

os.environ.setdefault("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

import pipeline_base as pb

DADOS_DIR = pb.BASE_DIR / "dados"
LOG_PATH = pb.METRICAS_DIR / "gabarito_cfm_6metodos.jsonl"
METODOS = ["stuff", "refine", "map_reduce", "map_rerank", "query_step_down", "reciprocal"]


def main():
    dataset = json.loads((DADOS_DIR / "processados" / "gabarito_cfm_completo.json").read_text(encoding="utf-8"))
    perguntas = dataset["perguntas"]
    print(f"{len(perguntas)} perguntas carregadas do gabarito humano do CFM.")

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
    print(f"{len(chunks)} chunks no corpus combinado (5 documentos).")

    colecao = pb.construir_indice(chunks, colecao_nome="gabarito_cfm")

    # Retomada: pula os pares (método, pergunta) que já estão no log -- assim as perguntas novas
    # são acrescentadas no fim sem rodar de novo as antigas, e o avaliar_ragas.py (que retoma
    # contando linhas) avalia só as linhas novas.
    ja_feitos = set()
    if LOG_PATH.exists():
        for linha in LOG_PATH.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                r = json.loads(linha)
                ja_feitos.add((r["metodo_rag"], r["pergunta_id"]))
        print(f"{len(ja_feitos)} execuções já no log, puladas.")

    for metodo in METODOS:
        for p in perguntas:
            if (metodo, p["id"]) in ja_feitos:
                continue
            registro = pb.rodar_experimento(
                query=p["pergunta"],
                colecao=colecao,
                document_id="revalida",
                resposta_esperada=p["resposta_esperada"],
                metodo=metodo,
                log_path=LOG_PATH,
                extra_campos={"dominio": "revalida", "pergunta_id": p["id"], "tipo_pergunta": p["tipo"]},
            )
            print(
                f"[{metodo}] {p['id']}: sim={registro['similaridade_cosseno']:.4f} "
                f"tokens={registro['tokens_totais']} t_ger={registro['tempo_geracao_s']:.2f}s"
            )


if __name__ == "__main__":
    main()
