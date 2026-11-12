"""
Teste de fumaça dos LLMs do comparativo (pb.LLMS_DO_COMPARATIVO): manda o mesmo prompt Stuff
curto pra cada um e mostra resposta, tokens, tempo e a versão do modelo que a API usou. Serve pra
conferir chaves e parâmetros antes de uma rodada longa (custa centavos). LLMs sem chave no .env
são pulados.

Chaves no .env: OPENAI_API_KEY, OPENROUTER_API_KEY, DEEPSEEK_API_KEY, ANTHROPIC_API_KEY.

Uso: python testar_llms.py [rotulo ...]   (sem argumentos, testa todos)
"""

import os
import sys
import time

import requests

import pipeline_base as pb

_CHAVE_POR_BACKEND = {
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

CONTEXTO = [
    "Art. 73 do Código de Ética Médica: é vedado ao médico revelar fato de que tenha conhecimento "
    "em virtude do exercício de sua profissão, salvo por motivo justo, dever legal ou consentimento, "
    "por escrito, do paciente.",
]
PERGUNTA = "Em que casos o médico pode revelar fato sigiloso do paciente?"


def motivo_para_pular(backend: str) -> str | None:
    """Devolve o motivo de pular o LLM, ou None se dá pra chamar."""
    if backend == "ollama":
        try:
            requests.get(pb.OLLAMA_BASE_URL, timeout=3)
        except requests.ConnectionError:
            return f"Ollama não está rodando em {pb.OLLAMA_BASE_URL}"
        return None
    chave = _CHAVE_POR_BACKEND[backend]
    return None if os.getenv(chave) else f"{chave} ausente no .env"


def main(rotulos: list[str]):
    for llm in pb.LLMS_DO_COMPARATIVO:
        rotulo, modelo = llm["rotulo"], llm["modelo"]
        if rotulos and rotulo not in rotulos:
            continue
        motivo = motivo_para_pular(llm["backend"])
        if motivo:
            print(f"[{rotulo}] pulado: {motivo}\n")
            continue
        pb.configurar_llm_por_rotulo(rotulo)
        inicio = time.time()
        try:
            resultado = pb.gerar_resposta_stuff(PERGUNTA, CONTEXTO)
        except Exception as e:
            print(f"[{rotulo}] ERRO {type(e).__name__}: {e}\n")
            continue
        print(
            f"[{rotulo}] {modelo} -> versão {pb._modelo_llm_versao} | {time.time() - inicio:.1f}s | "
            f"tokens in={resultado['tokens_prompt']} out={resultado['tokens_gerados']}\n"
            f"  {resultado['resposta_texto'].strip()}\n"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
