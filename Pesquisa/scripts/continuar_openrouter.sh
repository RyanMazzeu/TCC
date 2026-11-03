#!/usr/bin/env bash
# Passos 5 a 10 do plano para os 4 modelos do OpenRouter (Qwen3.5-9B, DeepSeek V4 Pro e Claude Haiku
# 5.5 na camada principal; Claude Sonnet 5 na de generalização), em sequência, sem precisar de ninguém
# olhando: geração, repetições do Stuff (Haiku e Sonnet, que não aceitam temperatura 0), RAGAS (juiz
# gpt-4o-mini, na chave da OpenAI) e a análise. Cada etapa retoma de onde parou, então rodar este
# script de novo depois de uma queda (ou de recarregar crédito) é seguro.
#
# Uso (em Pesquisa/scripts): nohup bash continuar_openrouter.sh [--com-sonnet] >> continuar_openrouter.log 2>&1 &
# Sem --com-sonnet, roda só os 3 da camada principal: o Sonnet sai ~US$ 7 no piloto de 02/11 (o dobro
# do previsto: o Claude conta ~1,7x mais tokens de entrada que o GPT e escreve ~3,6x mais) e depende de
# recarga de crédito.
set -eu
COM_SONNET=0; [ "${1:-}" = "--com-sonnet" ] && COM_SONNET=1
cd "$(dirname "$0")"
PY=../../.venv/bin/python
# 4 processos com PyTorch (embedding das perguntas e da similaridade) num notebook de 12 núcleos: sem
# limite, cada um abre 12 threads e a máquina trava (load 33 no piloto de 02/11).
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

etapa() { echo; echo "=== $(date '+%d/%m %H:%M') $*"; }
trap 'echo; echo "=== $(date "+%d/%m %H:%M") PAROU: a etapa acima falhou (ver os rodada_*.log; erro 402 = acabou o crédito do OpenRouter); depois de corrigir, rode este script de novo"' ERR

# Impede o notebook de suspender sozinho por inatividade; desligar pelo menu ou fechar a tampa ainda param tudo.
systemd-inhibit --what=sleep:idle --who="TCC benchmarks" --why="continuar_openrouter.sh rodando" --mode=block \
    tail --pid=$$ -f /dev/null > /dev/null 2>&1 &

etapa "esperando o piloto terminar"
while pgrep -f "rodar_benchmarks.py --llms" > /dev/null; do sleep 30; done

etapa "geração em paralelo: qwen, deepseek, claude_haiku (principal)$([ $COM_SONNET = 1 ] && echo ' e claude_sonnet (generalização)')"
pids=()
for m in qwen deepseek claude_haiku; do
    $PY -u rodar_benchmarks.py --camada principal --llms $m >> rodada_principal_$m.log 2>&1 & pids+=($!)
done
if [ $COM_SONNET = 1 ]; then
    $PY -u rodar_benchmarks.py --camada generalizacao --llms claude_sonnet >> rodada_generalizacao_claude_sonnet.log 2>&1 & pids+=($!)
fi
falhou=0
for p in "${pids[@]}"; do wait "$p" || falhou=1; done
[ "$falhou" = 0 ] || false

etapa "geração: segunda passada (pega o que tiver ficado pra trás)"
$PY -u rodar_benchmarks.py --camada principal --llms qwen deepseek claude_haiku
[ $COM_SONNET = 1 ] && $PY -u rodar_benchmarks.py --camada generalizacao --llms claude_sonnet
etapa "repetições do Stuff (modelos sem temperatura 0)"
$PY -u rodar_benchmarks.py --repeticoes --llms claude_haiku $([ $COM_SONNET = 1 ] && echo claude_sonnet)

# Conta da OpenAI no tier 2 (2 mi tokens/min no gpt-4o-mini): ver continuar_openai.sh.
etapa "RAGAS: Faithfulness e Answer Relevancy por resposta"
$PY -u avaliar_ragas_benchmarks.py respostas --llms qwen deepseek claude_haiku $([ $COM_SONNET = 1 ] && echo claude_sonnet) --por-minuto 600 --concorrencia 40

etapa "análise"
$PY -u analisar_benchmarks.py
etapa "fim"
