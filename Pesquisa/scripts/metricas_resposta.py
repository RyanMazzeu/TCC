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
- Docugami: a fração dos valores do gabarito (US$, %, milhões, bilhões...) que aparecem na resposta.
  A similaridade de cosseno do artigo-base não mede acerto aqui: uma resposta genérica, sem nenhum
  número ("Apple's R&D spending has generally risen... I don't have verified figures"), tira ~0,86
  contra um gabarito cheio de valores. Datas, anos, itens de lista e números de processo não contam
  como valor; a comparação aceita arredondamento e troca de escala ($7,442 million = $7.44 billion).
- Todos: se a resposta é uma recusa, total ou parcial ("não sei", "I don't know", "I don't have
  verified figures"...).
"""

import re
import string
import unicodedata
from collections import Counter

_PONTUACAO = set(string.punctuation) | {"“", "”", "‘", "’", "–", "—", "…"}

_NAO = r"(?:do not|don['’]?t)"
RECUSA = re.compile(
    r"^\s*n[ãa]o sei\b|n[ãa]o h[áa] informa[çc][ãa]o relevante|o contexto n[ãa]o (?:fornece|menciona|cont[ée]m|apresenta)"
    r"|n[ãa]o (?:tenho|possuo) (?:acesso|informa[çc][õo]es|dados|os (?:dados|n[úu]meros))"
    r"|n[ãa]o (?:consigo|posso) (?:confirmar|afirmar|determinar|informar|responder)"
    rf"|^\s*i {_NAO} know\b|i do not know|\bi {_NAO} know which|there is no relevant information"
    r"|the (?:provided )?context (?:does not|doesn['’]?t) (?:provide|mention|contain|include)"
    # Recusas parciais dos modelos sem os documentos: respondem genericamente e avisam que não têm os dados.
    rf"|\bi {_NAO} have (?:verified|specific|access|enough|reliable|the (?:specific|exact|actual|latest|verified))"
    r"|\bi (?:can['’]?t|cannot) (?:reliably|accurately|confirm|verify|determine|provide (?:the|specific|exact|a reliable))",
    re.IGNORECASE,
)

# Valor numérico: com moeda, %, escala ou formato de número (separador de milhar, casa decimal).
_ESCALA = {"thousand": 1e3, "million": 1e6, "millions": 1e6, "billion": 1e9, "billions": 1e9, "trillion": 1e12}
_VALOR = re.compile(
    r"(?P<moeda>\$)?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)"
    r"(?:\s?(?P<pct>%|percent\b)|\s(?P<escala>thousand|millions?|billions?|trillion)\b)?",
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


def valores(texto: str) -> list[tuple[float, float, str]]:
    """(valor, meia unidade da última casa, "%" ou "") de cada valor numérico do texto. Sem moeda, % ou
    escala, só conta número com separador de milhar ou casa decimal -- o que deixa de fora datas
    ("July 1, 2023"), anos ("2022 Q3"), itens de lista ("1.") e números de processo ("4:18-cv-07669")."""
    saida = []
    for m in _VALOR.finditer(texto or ""):
        num, rotulado = m["num"], bool(m["moeda"] or m["pct"] or m["escala"])
        if not rotulado:
            vizinhos = texto[max(0, m.start() - 1):m.start()] + texto[m.end():m.end() + 1]
            if not ("," in num or "." in num) or any(c in vizinhos for c in ":-/"):
                continue
        casas = len(num.split(".")[1]) if "." in num else 0
        escala = _ESCALA[m["escala"].lower()] if m["escala"] else 1.0
        saida.append((float(num.replace(",", "")) * escala, 0.5 * 10**-casas * escala, "%" if m["pct"] else ""))
    return saida


def recall_valores(gerada: str, esperada: str) -> float:
    """Fração dos valores distintos do gabarito que aparecem na resposta (NaN se o gabarito não tem
    valor). Aceita o arredondamento da resposta ($7.4 billion para $7,442 million), até 2% de
    diferença, e sempre 0,5%."""
    ouro = list({(round(v, 6), tipo): (v, tipo) for v, _, tipo in valores(esperada)}.values())
    if not ouro:
        return float("nan")
    resposta = valores(gerada)

    def aparece(v_ouro: float, tipo: str) -> bool:
        return any(t == tipo and abs(v - v_ouro) <= max(0.005 * abs(v_ouro), min(meia, 0.02 * abs(v_ouro)))
                   for v, meia, t in resposta)

    return sum(aparece(v, tipo) for v, tipo in ouro) / len(ouro)


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
    if "respostas_aceitas" not in pergunta and "letra_correta" not in pergunta:
        saida["valores_gabarito"] = recall_valores(resposta, pergunta.get("resposta_esperada") or "")
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
    assert eh_recusa("Apple's R&D has risen. I don’t have verified figures for its most recent quarter.")
    assert eh_recusa("The context doesn’t provide NVIDIA’s Q1 2023 operating expenses.")
    assert not eh_recusa("R&D expenses were $7,442 million, up from $6,797 million.")
    ouro = "R&D was $7,442 million in the quarter ended July 1, 2023, versus $6,797 million (1. see note)."
    assert [v for v, _, _ in valores(ouro)] == [7.442e9, 6.797e9]
    assert recall_valores("R&D was $7.44 billion, up from $6.8 billion.", ouro) == 1.0
    assert recall_valores("It was $7.4 billion.", ouro) == 0.5
    assert recall_valores("It was about $7 billion.", ouro) == 0.0  # 6% de diferença: não é arredondamento
    assert recall_valores("Apple's R&D spending has generally risen.", ouro) == 0.0
    assert valores("Case 4:18-cv-07669-HSG, filed December 21, 2018; 2022 Q3 MSFT.pdf; margin 44.5%") == [(44.5, 0.05, "%")]
    assert recall_valores("Margin was 45%.", "Gross margin was 44.5%.") == 1.0
    assert recall_valores("qualquer coisa", "Filed on December 21, 2018.") != recall_valores("x", "y")  # NaN
    print("ok: todos os casos conhecidos passaram")
