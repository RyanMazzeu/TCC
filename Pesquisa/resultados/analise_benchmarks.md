# Análise da rodada dos benchmarks

730 execuções nos logs `benchmarks_*.jsonl`.

## 1. Recuperação (sem LLM, todas as perguntas)

| Dataset | Perguntas | Hit@4 | Recall@4 | MRR@10 | Hit@10 |
|---|---|---|---|---|---|
| docugami_10q | 193 | 0,819 | 0,754 | 0,625 | 0,969 |
| faquad | 900 | 0,906 | 0,906 | 0,812 | 0,967 |

O MedQA não tem trecho certo marcado: a busca dele é avaliada pela Context Precision/Recall (seção 7).

## 2. Camada principal (5 modelos econômicos, 100 perguntas por dataset)

### docugami_10q

Métrica principal: Similaridade de cosseno.

| LLM | Método | n | Similaridade de cosseno [IC 95%] | Cosseno | Recusas | Tokens (média) | Custo total US$ | Tempo s (mediana) |
|---|---|---|---|---|---|---|---|---|
| gpt_luna | Sem RAG | 100 | 0,808 [0,792; 0,823] | 0,808 | 0% | 140 | 0,0048 | 6,1 |
| gpt_luna | Stuff | 100 | 0,777 [0,757; 0,797] | 0,777 | 23% | 997 | 0,0126 | 15,0 |
| gpt_luna | Refine | 71 | 0,786 [0,763; 0,809] | 0,786 | 0% | 1.722 | 0,0209 | 16,4 |
| gpt_mini | Sem RAG | 100 | 0,568 [0,525; 0,612] | 0,568 | 57% | 108 | 0,0040 | 4,8 |
| gpt_mini | Stuff | 100 | 0,599 [0,551; 0,648] | 0,599 | 59% | 977 | 0,0166 | 7,2 |
| gpt_mini | Refine | 96 | 0,728 [0,684; 0,770] | 0,728 | 26% | 1.752 | 0,0406 | 19,0 |
| gpt_mini | Map-Reduce | 2 | 0,839 [0,783; 0,895] | 0,839 | 0% | 1.112 | 0,0004 | 6,3 |
| gpt_mini | Map-Rerank | 2 | 0,701 [0,618; 0,784] | 0,701 | 100% | 1.148 | 0,0005 | 6,1 |
| gpt_mini | Step-Down | 2 | 0,911 [0,897; 0,924] | 0,911 | 0% | 4.735 | 0,0017 | 9,6 |
| gpt_mini | Reciprocal | 2 | 0,643 [0,620; 0,667] | 0,643 | 0% | 4.678 | 0,0016 | 8,3 |

Friedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:

| LLM | Perguntas | χ² | p | Pares com p_Holm < 0,05 |
|---|---|---|---|---|
| gpt_mini | 2 | -- | -- | -- |

### medqa

Métrica principal: Acurácia.

| LLM | Método | n | Acurácia [IC 95%] | Cosseno | Recusas | Tokens (média) | Custo total US$ | Tempo s (mediana) |
|---|---|---|---|---|---|---|---|---|
| gpt_mini | Sem RAG | 2 | 0,500 [0,000; 1,000] | 0,508 | 0% | 463 | 0,0002 | 3,1 |
| gpt_mini | Stuff | 2 | 0,500 [0,000; 1,000] | 0,472 | 0% | 1.246 | 0,0005 | 4,3 |
| gpt_mini | Refine | 2 | 0,500 [0,000; 1,000] | 0,523 | 0% | 3.174 | 0,0014 | 9,4 |
| gpt_mini | Map-Reduce | 2 | 0,500 [0,000; 1,000] | 0,503 | 0% | 3.306 | 0,0013 | 9,2 |
| gpt_mini | Map-Rerank | 2 | 0,500 [0,000; 1,000] | 0,470 | 0% | 2.569 | 0,0009 | 6,7 |
| gpt_mini | Step-Down | 2 | 1,000 [1,000; 1,000] | 0,541 | 0% | 5.828 | 0,0022 | 12,6 |
| gpt_mini | Reciprocal | 2 | 0,500 [0,000; 1,000] | 0,481 | 0% | 4.994 | 0,0018 | 10,0 |

Friedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:

| LLM | Perguntas | χ² | p | Pares com p_Holm < 0,05 |
|---|---|---|---|---|
| gpt_mini | 2 | -- | -- | -- |

### faquad

Métrica principal: Cobertura de tokens.

| LLM | Método | n | Cobertura de tokens [IC 95%] | Cosseno | Recusas | Tokens (média) | Custo total US$ | Tempo s (mediana) |
|---|---|---|---|---|---|---|---|---|
| gpt_mini | Sem RAG | 2 | 0,483 [0,167; 0,800] | 0,491 | 0% | 190 | 0,0002 | 3,6 |
| gpt_mini | Stuff | 2 | 1,000 [1,000; 1,000] | 0,629 | 0% | 682 | 0,0002 | 1,6 |
| gpt_mini | Refine | 2 | 1,000 [1,000; 1,000] | 0,539 | 0% | 2.166 | 0,0014 | 14,6 |
| gpt_mini | Map-Reduce | 2 | 1,000 [1,000; 1,000] | 0,629 | 0% | 1.062 | 0,0004 | 4,8 |
| gpt_mini | Map-Rerank | 2 | 0,600 [0,200; 1,000] | 0,606 | 50% | 1.116 | 0,0005 | 4,5 |
| gpt_mini | Step-Down | 2 | 0,821 [0,808; 0,833] | 0,563 | 0% | 3.834 | 0,0015 | 10,1 |
| gpt_mini | Reciprocal | 2 | 0,917 [0,833; 1,000] | 0,699 | 0% | 3.650 | 0,0013 | 7,4 |

Friedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:

| LLM | Perguntas | χ² | p | Pares com p_Holm < 0,05 |
|---|---|---|---|---|
| gpt_mini | 2 | -- | -- | -- |

## 3. Efeito do RAG (método − sem RAG, na métrica principal)

### docugami_10q (Similaridade de cosseno)

| LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down |
|---|---|---|---|---|---|---|
| gpt_luna | 0,808 | -0,031* | -0,018 |
| gpt_mini | 0,568 | +0,031 | +0,165* | +0,196 | +0,058 | +0,268 | +0,001 |
| gpt_sol | 0,778 | -0,042 | +0,019 | -0,012 | -0,100* | +0,029 |

\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).

### medqa (Acurácia)

| LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|
| gpt_mini | 0,500 | +0,000 | +0,000 | +0,000 | +0,000 | +0,500 | +0,000 |

\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).

### faquad (Cobertura de tokens)

| LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|
| gpt_mini | 0,483 | +0,517 | +0,517 | +0,517 | +0,117 | +0,337 | +0,433 |

\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).

## 4. Generalização (7 modelos, as mesmas 25 perguntas por dataset)

### docugami_10q (Similaridade de cosseno, média nas 25 perguntas)

| LLM | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| gpt_mini | 0,607 | 0,748 | 0,839 | 0,701 | 0,911 | 0,643 |
### medqa (Acurácia, média nas 25 perguntas)

| LLM | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| gpt_mini | 0,500 | 0,500 | 0,500 | 0,500 | 1,000 | 0,500 |
### faquad (Cobertura de tokens, média nas 25 perguntas)

| LLM | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| gpt_mini | 1,000 | 1,000 | 1,000 | 0,600 | 0,821 | 0,917 |
## 5. Variação do modelo

Ainda sem repetições: rode `rodar_benchmarks.py --repeticoes`.

## 6. Comparação com Şakar & Emekci (2025), Tabela 1 (mediana da similaridade de cosseno)

### docugami_10q

| Método | Paper-âncora | gpt_luna | gpt_mini | gpt_sol |
|---|---|---|---|---|
| Reciprocal | 0,971 | -- | 0,643 | -- |
| Stuff | 0,895 | 0,784 | 0,421 | 0,747 |
| Step-Down | 0,890 | -- | 0,911 | 0,850 |
| Map-Reduce | 0,873 | -- | 0,839 | 0,770 |
| Refine | 0,836 | 0,807 | 0,819 | 0,806 |
| Map-Rerank | 0,801 | -- | 0,701 | 0,656 |

### medqa

| Método | Paper-âncora | gpt_mini |
|---|---|---|
| Step-Down | 0,678 | 0,541 |
| Reciprocal | 0,673 | 0,481 |
| Stuff | 0,641 | 0,472 |
| Refine | 0,609 | 0,523 |
| Map-Reduce | 0,569 | 0,503 |
| Map-Rerank | 0,148 | 0,470 |

O paper-âncora agrega vários LLMs, embeddings e bancos vetoriais, e não informa o embedding da similaridade: a comparação é indicativa.

## 7. RAGAS (juiz gpt-4o-mini; métricas complementares)

Ainda não calculado: rode `avaliar_ragas_benchmarks.py respostas` e `... contexto`.

## 8. Bancos vetoriais: ChromaDB (HNSW, aproximado) × FAISS (IndexFlatL2, exato)

| Dataset (vetores) | Index. Chroma (s) | Index. FAISS (s) | Consulta Chroma (ms) | Consulta FAISS (ms) | Mem. Chroma (MB) | Mem. FAISS (MB) | Top-4 idêntico | Sobreposição top-4 | Hit@4 Chroma × FAISS |
|---|---|---|---|---|---|---|---|---|---|
| faquad (318) | 0,55 | 0,004 | 3,43 | 0,06 | 32,1 | 1,6 | 100% | 100% | 0,930 × 0,930 |

Top-4 idêntico: fração das consultas em que os dois bancos entregam os mesmos 4 fragmentos ao LLM (nelas, a resposta do RAG seria a mesma com o FAISS).
