"""
Backend Claude (Anthropic) do pipeline_base.py, com o SDK oficial `anthropic`. Fica num arquivo
à parte para que só quem usa Claude precise do SDK instalado (`pip install anthropic`).

Diferenças em relação aos outros backends, que precisam constar na metodologia:
- O Claude Opus 4.8 não aceita temperature/top_p (a API devolve 400), então a geração não é
  fixada em temperatura 0 como nos outros LLMs.
- Sem o parâmetro `thinking`, o Opus 4.8 responde sem raciocínio estendido -- o mesmo regime do
  DeepSeek com o raciocínio desligado.

Lê a chave de ANTHROPIC_API_KEY (no .env).
"""

import anthropic

_cliente = None


def gerar(prompt: str, modelo: str) -> dict:
    global _cliente
    if _cliente is None:
        _cliente = anthropic.Anthropic(max_retries=8)

    resposta = _cliente.messages.create(
        model=modelo,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
    )
    # Recusa por política (stop_reason "refusal") volta sem bloco de texto: fica registrada como
    # resposta vazia, e não como erro, pra não derrubar a rodada inteira.
    texto = "".join(bloco.text for bloco in resposta.content if bloco.type == "text")
    return {
        "resposta_texto": texto,
        "tokens_prompt": resposta.usage.input_tokens,
        "tokens_gerados": resposta.usage.output_tokens,
        "modelo_llm": modelo,
        "modelo_llm_versao": resposta.model,
    }
