#!/usr/bin/env bash
# Passos 5 a 10 do plano que dependem só da chave da OpenAI, em sequência, sem precisar de ninguém
# olhando: espera a geração (camada principal do gpt-4o-mini e do GPT-6 Luna; generalização do
# GPT-6 Sol) terminar, roda as repetições do Stuff, o RAGAS, o comparativo de bancos vetoriais e a
# análise. Cada etapa retoma de onde parou, então rodar este script de novo depois de uma queda é seguro.
#
# Uso (em Pesquisa/scripts): nohup bash continuar_openai.sh > continuar_openai.log 2>&1 &
# Se uma etapa falhar (ex.: acabou o crédito da API), a cadeia para ali, em vez de seguir pra RAGAS e
# análise com dados pela metade.
set -eu
cd "$(dirname "$0")"
PY=../../.venv/bin/python

etapa() { echo; echo "=== $(date '+%d/%m %H:%M') $*"; }
trap 'echo; echo "=== $(date "+%d/%m %H:%M") PAROU: a etapa acima falhou (ver os rodada_*.log); depois de corrigir, rode este script de novo"' ERR

# Impede o notebook de suspender sozinho por inatividade enquanto a cadeia roda; o bloqueio some
# quando este script termina. Fechar a tampa ainda suspende.
systemd-inhibit --what=sleep:idle --who="TCC benchmarks" --why="continuar_openai.sh rodando" --mode=block \
    tail --pid=$$ -f /dev/null > /dev/null 2>&1 &

etapa "esperando a geração em andamento terminar"
while pgrep -f "rodar_benchmarks.py --camada" > /dev/null; do sleep 60; done

# Os 3 modelos em paralelo (cada um num processo, com o próprio log); o que já está no log é pulado.
etapa "geração em paralelo: gpt_mini, gpt_luna (principal) e gpt_sol (generalização)"
pids=()
$PY -u rodar_benchmarks.py --camada principal --llms gpt_mini >> rodada_principal_gpt_mini.log 2>&1 & pids+=($!)
$PY -u rodar_benchmarks.py --camada principal --llms gpt_luna >> rodada_principal_gpt_luna.log 2>&1 & pids+=($!)
$PY -u rodar_benchmarks.py --camada generalizacao --llms gpt_sol >> rodada_generalizacao_gpt_sol.log 2>&1 & pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

# Uma segunda passada pega o que tiver ficado pra trás (ex.: um processo que caiu); o que já está no
# log é pulado.
etapa "geração: camada principal (gpt_mini, gpt_luna)"
$PY -u rodar_benchmarks.py --camada principal --llms gpt_mini gpt_luna
etapa "geração: camada de generalização (gpt_sol)"
$PY -u rodar_benchmarks.py --camada generalizacao --llms gpt_sol
etapa "repetições do Stuff (modelos sem temperatura 0: gpt_luna, gpt_sol)"
$PY -u rodar_benchmarks.py --repeticoes --llms gpt_luna gpt_sol

etapa "RAGAS: Faithfulness e Answer Relevancy por resposta"
$PY -u avaliar_ragas_benchmarks.py respostas --llms gpt_mini gpt_luna gpt_sol --por-minuto 120
etapa "RAGAS: Context Precision e Context Recall, uma vez por pergunta"
$PY -u avaliar_ragas_benchmarks.py contexto --por-minuto 120

etapa "comparativo ChromaDB × FAISS (passo 1: Chroma)"
$PY -u comparar_bancos_passo1_chroma.py
etapa "comparativo ChromaDB × FAISS (passo 2: FAISS)"
$PY -u comparar_bancos_passo2_faiss.py

etapa "análise"
$PY -u analisar_benchmarks.py
etapa "fim"
