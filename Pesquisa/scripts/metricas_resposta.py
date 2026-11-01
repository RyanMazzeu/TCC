"""
Métricas de resposta calculadas direto contra o gabarito, sem juiz LLM:

- FaQuAD: F1 e Exact Match no formato do SQuAD (Rajpurkar et al. 2016), a métrica oficial do FaQuAD,
  contra todas as respostas aceitas (vale a melhor). A normalização segue o SQuAD (minúsculas, sem
  pontuação, espaços colapsados), mas sem remover artigos: em português, "a", "o" e "as" também são
  preposição e pronome, e removê-los muda o sentido -- o mesmo argumento do Pirá 2.0 (Pirozelli et
  al. 2024). Como as respostas do FaQuAD são trechos curtos e as dos LLMs são frases, também vai a
  cobertura de tokens (fração dos tokens da resposta certa que aparecem na resposta gerada), que não
  pune a resposta por ser mais longa.
- MedQA: a letra escolhida (A-D) e a acurácia.
- Todos: se a resposta é uma recusa ("não sei", "I don't know"...).
"""

import re
import string
import unicodedata
from collections import Counter

_PONTUACAO = set(string.punctuation) | {"“", "”", "‘", "’", "–", "—", "…"}

RECUSA = re.compile(
    r"^\s*n[ãa]o sei\b|n[ãa]o h[áa] informa[çc][ãa]o relevante|o contexto n[ãa]o (?:fornece|menciona|cont[ée]m|apresenta)"
    r"|^\s*i don'?t know\b|i do not know|there is no relevant information"
    r"|the (?:provided )?context does not (?:provide|mention|contain|include)",
    re.IGNORECASE,
)

# A letra escolhida, na ordem de preferência: no começo da resposta ("B.", "(B)", "B:", "Answer: B"),
# depois em "the answer is B" / "option B" em qualquer ponto.
# Depois da letra, exige pontuação ou fim de linha: "A patient with..." começa com o artigo "A", não
# com a alternativa A.
_LETRA_NO_INICIO = re.compile(r"^\s*(?:\*\*)?(?:answer\s*[:\-]\s*)?(?:\*\*)?\(?([A-D])(?:\)|\*\*|[.:\-]|\s+[-–—]|\s*$|\s*\n)", re.IGNORECASE)
_LETRA_NO_TEXTO = re.compile(r"\b(?:answer|option|choice)\s+(?:is\s+)?\(?([A-D])\)?\b", re.IGNORECASE)


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto.lower())
    texto = "".join(ch for ch in texto if ch not in _PONTUACAO)
    return " ".join(texto.split())


def _tokens(texto: str) -> list[str]:
    return _normalizar(texto).split()


def exact_match(gerada: str, aceitas: list[str]) -> float:
    return float(any(_normalizar(gerada) == _normalizar(a) for a in aceitas))


def _f1(gerada: list[str], certa: list[str]) -> float:
    comuns = sum((Counter(gerada) & Counter(certa)).values())
    if comuns == 0:
        return 0.0
    precisao, revocacao = comuns / len(gerada), comuns / len(certa)
    return 2 * precisao * revocacao / (precisao + revocacao)


def f1(gerada: str, aceitas: list[str]) -> float:
    return max(_f1(_tokens(gerada), _tokens(a)) for a in aceitas)


def cobertura_tokens(gerada: str, aceitas: list[str]) -> float:
    gerados = Counter(_tokens(gerada))
    return max(
        sum((gerados & Counter(_tokens(a))).values()) / len(_tokens(a)) if _tokens(a) else 0.0
        for a in aceitas
    )


def letra_escolhida(resposta: str) -> str | None:
    m = _LETRA_NO_INICIO.search(resposta) or _LETRA_NO_TEXTO.search(resposta)
    return m.group(1).upper() if m else None


def eh_recusa(resposta: str) -> bool:
    return bool(RECUSA.search(resposta or ""))


def metricas(registro: dict, pergunta: dict) -> dict:
    """Métricas de uma linha de log, conforme o que o gabarito da pergunta tem."""
    resposta = registro.get("resposta_gerada") or ""
    saida = {"recusa": eh_recusa(resposta)}
    if "respostas_aceitas" in pergunta:
        aceitas = pergunta["respostas_aceitas"]
        saida.update(f1=f1(resposta, aceitas), exact_match=exact_match(resposta, aceitas),
                     cobertura_tokens=cobertura_tokens(resposta, aceitas))
    if "letra_correta" in pergunta:
        letra = letra_escolhida(resposta)
        saida.update(letra_escolhida=letra, acerto=float(letra == pergunta["letra_correta"]))
    return saida


if __name__ == "__main__":
    # Casos conhecidos (rodar com `python metricas_resposta.py`): para se algum falhar.
    assert letra_escolhida("B. Ceftriaxone, because...") == "B"
    assert letra_escolhida("**C** - the drug of choice") == "C"
    assert letra_escolhida("(D) Tell the attending") == "D"
    assert letra_escolhida("Answer: A. Metformin") == "A"
    assert letra_escolhida("Based on the context, the answer is C, since...") == "C"
    assert letra_escolhida("I cannot determine the answer.") is None
    assert letra_escolhida("A patient with fever...") is None
    assert letra_escolhida("B") == "B"
    assert letra_escolhida("B - Ceftriaxone") == "B"
    assert exact_match("Informática.", ["informática", "Tecnologia da Informação"]) == 1.0
    assert f1("a informática", ["Informática"]) == 2 * 0.5 * 1 / 1.5
    assert f1("nada a ver", ["Informática"]) == 0.0
    assert cobertura_tokens("O componente é a Informática ou Tecnologia da Informação.", ["Informática ou Tecnologia da Informação"]) == 1.0
    assert eh_recusa("Não sei.") and eh_recusa("I don't know.") and not eh_recusa("Informática")
    print("ok: todos os casos conhecidos passaram")
