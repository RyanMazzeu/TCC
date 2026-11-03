#!/usr/bin/env bash
# Wrapper de 02/11: espera o Haiku terminar a geração, roda as repetições dele, espera o RAGAS adiantado
# e então chama continuar_openrouter.sh (que espera os processos "rodar_benchmarks.py --llms" do MedQA).
cd "$(dirname "$0")"
PY=../../.venv/bin/python
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
while kill -0 22440 2>/dev/null; do sleep 30; done
echo "=== $(date '+%d/%m %H:%M') repetições do Stuff do claude_haiku"
$PY -u rodar_benchmarks.py --llms claude_haiku --repeticoes >> rodada_repeticoes_claude_haiku.log 2>&1
while pgrep -f avaliar_ragas_benchmarks > /dev/null; do sleep 30; done
exec bash continuar_openrouter.sh
