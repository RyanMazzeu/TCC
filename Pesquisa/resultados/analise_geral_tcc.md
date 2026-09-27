# Análise geral — onde o TCC está e o que falta

Data: 2026-09-27. Substitui a versão de 2026-09-06 (que continua no histórico do git, commit
`64671d8`). Raio-x único do que existe, do que está pronto e do que ainda depende de vocês.

## 1. Inventário

| Peça | Onde está | Estado |
|---|---|---|
| Anteprojeto (TCC01) | `Plano_Trabalho/ECO_plano_de_trabalho_TCC01/` | Aprovado pela banca (junho/2026) |
| Pipeline experimental | `Pesquisa/scripts/pipeline_base.py` | 6 métodos de RAG, ChromaDB + FAISS, BGE-M3, gpt-4o-mini |
| Comparativo dos 6 métodos | `resultados/metricas/comparativo_6_metodos_v3.jsonl` | 162 execuções (27 perguntas × 6 métodos) |
| RAGAS sobre o comparativo | `comparativo_6_metodos_v3_ragas.jsonl` | 162/162 linhas, 4 métricas, sem falhas |
| Validação do juiz RAGAS | `..._ragas_reteste_mini.jsonl`, `..._ragas_juiz_cohere.jsonl` | Amostra de 54: reteste do 4o-mini + juiz Cohere Command A |
| Compressão contextual | `comparativo_compressao_contextual_v2.jsonl`, `comparativo_compressao_calibrada.jsonl` | Limiares fixos e calibrados por domínio, 27 perguntas cada |
| Bancos vetoriais | `comparativo_vectorstores.json` | ChromaDB × FAISS (infraestrutura) |
| Análise estatística | `resultados/analise_estatistica.md` (gerado por `scripts/analisar_resultados.py`) | Friedman + Wilcoxon/Holm, IC bootstrap, Spearman |
| Tabelas e figuras | `resultados/tabelas/*.tex`, `resultados/graficos/*.pdf` | Geradas pelo mesmo script, prontas pro LaTeX |
| Artigo do TCC02 | `Artigo_TCC02/` | Rascunho completo, compila com `latexmk -pdf main.tex` |
| Relatório de pesquisa | `resultados/piloto_relatorio.md` | Histórico de todas as rodadas |
| Notebook de testes | `notebooks/Testar_RAG.ipynb` | Pra testar perguntas avulsas |

## 2. Situação das lacunas do anteprojeto

| Prometido no anteprojeto | Situação |
|---|---|
| Datasets SEC 10Q / Llama2 / MedQA | Trocados por Revalida / Grendene / SBC (PT-BR), validado com o orientador. Justificativa no artigo, seção 3.8 |
| ChromaDB, FAISS, Pinecone | ChromaDB e FAISS cobertos; Pinecone fora (justificado no artigo) |
| 3 embeddings | Só BGE-M3 (justificado no artigo) |
| Similaridade de cosseno + RAGAS | As duas cobertas, com validação do juiz |
| Compressão contextual | Coberta, com limiares fixos e calibrados |
| Gráficos e estatística | Feitos (seção 1) |
| Hardware (CPU/RAM) | Medido, mas não analisado: com LLM via API, a CPU local não reflete o custo do método (limitação declarada no artigo) |

## 3. O que ainda depende de vocês

1. **Template oficial do TCC02.** O artigo foi escrito num `article` genérico porque o template
   da coordenação não está no repositório (o `Plano_Trabalho/LEIAM-ME.txt` só diz que o TCC02 é um
   artigo). As seções estão em arquivos separados justamente pra serem movidas pro template.
2. **Revisar o texto inteiro.** É um rascunho: conferir cada afirmação, ajustar a voz de vocês e
   verificar as regras do curso sobre uso de IA na redação.
3. **Pendências marcadas no texto.** Trocar `\usepackage[disable]{todonotes}` por
   `\usepackage{todonotes}` no `main.tex` mostra as marcações (máquina dos experimentos, versão
   do modelo, redação da justificativa com o orientador).
4. **Referências novas.** Conferir os metadados das entradas adicionadas no fim do
   `Artigo_TCC02/referencias.bib` (principalmente Medeiros & Oliveira 2025, sem paginação).
5. **Orientador.** Levar o artigo, principalmente a seção de diferenças em relação ao plano.
6. **Apresentação (F.5 do cronograma).** Ainda não existe.
