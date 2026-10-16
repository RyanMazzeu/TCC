"""
Converte Pesquisa/dados/gabaritos_cfm/perguntas_cfm_para_gabarito.txt (preenchido à mão por
Ryan e Afonso) para o formato JSON usado pelos experimentos.

Processa só os blocos [Pxx] que já têm RESPOSTA preenchida e STATUS != DESCARTAR -- pode ser
rodado a qualquer momento, mesmo com só parte das 50 perguntas preenchidas, e pega
automaticamente o que estiver pronto (sem precisar editar o script quando o resto chegar).
"""

import json
import re
from pathlib import Path

ENTRADA = Path(__file__).resolve().parent.parent / "dados" / "gabaritos_cfm" / "perguntas_cfm_para_gabarito.txt"
SAIDA = Path(__file__).resolve().parent.parent / "dados" / "processados" / "gabarito_cfm_completo.json"

PADRAO_BLOCO = re.compile(r"\[P(\d+)\]\n(.*?)(?=\n\[P\d+\]|\n##|\Z)", re.DOTALL)
PADRAO_CAMPO = re.compile(
    r"TIPO:\s*(?P<tipo>.*?)\s*\|\s*ONDE PROCURAR:\s*(?P<onde>.*?)\n"
    r"PERGUNTA:\s*(?P<pergunta>.*?)\n"
    r"RESPOSTA:\s*(?P<resposta>.*?)\n"
    r"TRECHO:\s*(?P<trecho>.*?)\n"
    r"P[ÁA]GINA:\s*(?P<pagina>.*?)\n"
    r"STATUS:\s*(?P<status>.*?)\s*$",
    re.DOTALL,
)


def main():
    texto = ENTRADA.read_text(encoding="utf-8")
    perguntas = []
    ignoradas_vazias = 0
    ignoradas_descartadas = 0

    for match in PADRAO_BLOCO.finditer(texto):
        numero = match.group(1)
        campos = PADRAO_CAMPO.match(match.group(2).strip() + "\n")
        if not campos:
            print(f"AVISO: P{numero} não bateu com o padrão esperado, pulando.")
            continue

        resposta = campos["resposta"].strip()
        status = campos["status"].strip()

        if status.upper().startswith("DESCARTAR"):
            ignoradas_descartadas += 1
            continue
        if not resposta:
            ignoradas_vazias += 1
            continue

        perguntas.append(
            {
                "id": f"cfm_p{numero}",
                "tipo": campos["tipo"].strip(),
                "pergunta": campos["pergunta"].strip(),
                "resposta_esperada": resposta,
                "trecho_fonte": campos["trecho"].strip(),
                "pagina_pdf": campos["pagina"].strip(),
            }
        )

    saida = {
        "fonte": "Gabarito humano (Ryan + Afonso) sobre o Código de Ética Médica (Resolução CFM nº 2.217/2018)",
        "observacao": "P01-P25: respostas e trechos escritos manualmente pelos autores, sem apoio de IA. P26-P50: redigidas com apoio de IA a partir do texto do Código e revisadas manualmente pelos autores contra o documento. Ground truth para validar as métricas automáticas (cosseno, RAGAS) contra julgamento humano.",
        "corpus": "Mesmo corpus combinado de 5 documentos já usado no domínio 'revalida' (cfm_codigo_etica + sbp + tev + parto_febrasgo + meningococica).",
        "perguntas": perguntas,
    }

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(perguntas)} perguntas convertidas (de 50 esperadas).")
    print(f"{ignoradas_vazias} ainda sem resposta (pendente), {ignoradas_descartadas} marcadas DESCARTAR.")
    print(f"Salvo em {SAIDA}")


if __name__ == "__main__":
    main()
