#!/usr/bin/env bash
# 02/11: fecha o Claude Sonnet 5 (camada de generalização) depois da recarga: espera a geração dele (rodando
# em processos por dataset/método), faz uma segunda passada, as repetições do Stuff, espera a cadeia dos
# 3 modelos (continuar_openrouter.sh) terminar, e roda o RAGAS do Sonnet e a análise final com os 7 modelos.
set -eu
cd "$(dirname "$0")"
PY=../../.venv/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
etapa() { echo; echo "=== $(date '+%d/%m %H:%M') $*"; }
trap 'echo; echo "=== $(date "+%d/%m %H:%M") PAROU: a etapa acima falhou; depois de corrigir, rode este script de novo"' ERR
etapa "esperando a geração do claude_sonnet"
while pgrep -f "llms claude_sonnet" > /dev/null; do sleep 30; done
etapa "geração do claude_sonnet: segunda passada"
$PY -u rodar_benchmarks.py --camada generalizacao --llms claude_sonnet
etapa "repetições do Stuff do claude_sonnet"
$PY -u rodar_benchmarks.py --repeticoes --llms claude_sonnet
etapa "esperando a cadeia dos 3 modelos (continuar_openrouter.sh)"
while pgrep -f "continuar_openrouter" > /dev/null; do sleep 30; done
etapa "RAGAS do claude_sonnet"
$PY -u avaliar_ragas_benchmarks.py respostas --llms claude_sonnet --por-minuto 600 --concorrencia 40
etapa "análise final (7 modelos)"
$PY -u analisar_benchmarks.py
etapa "fim"
