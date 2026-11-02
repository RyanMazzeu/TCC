# Análise da rodada dos benchmarks

5025 execuções nos logs `benchmarks_*.jsonl`.

## 1. Recuperação (sem LLM, todas as perguntas)

| Dataset | Perguntas | Hit@4 | Recall@4 | MRR@10 | Hit@10 |
|---|---|---|---|---|---|
| docugami_10q | 193 | 0,819 | 0,754 | 0,625 | 0,969 |
| faquad | 900 | 0,906 | 0,906 | 0,812 | 0,967 |

O MedQA não tem trecho certo marcado: a busca dele é avaliada pela Context Precision/Recall (seção 7).

## 2. Camada principal (5 modelos econômicos, 100 perguntas por dataset)

### docugami_10q

Métrica principal: Valores do gabarito na resposta.

Só entram as perguntas com algum valor numérico no gabarito (as outras são qualitativas). Recusas parciais ("I don't have verified figures") contam como recusa.

| LLM | Método | n | Valores do gabarito na resposta [IC 95%] | Cosseno | Recusas | Tokens (média) | Custo total US$ | Tempo s (mediana) |
|---|---|---|---|---|---|---|---|---|
| gpt_luna | Sem RAG | 66 | 0,080 [0,030; 0,143] | 0,808 | 23% | 140 | 0,0048 | 6,1 |
| gpt_luna | Stuff | 66 | 0,149 [0,079; 0,229] | 0,777 | 36% | 997 | 0,0126 | 15,0 |
| gpt_luna | Refine | 66 | 0,143 [0,079; 0,217] | 0,783 | 4% | 1.705 | 0,0295 | 18,0 |
| gpt_luna | Map-Reduce | 66 | 0,202 [0,124; 0,289] | 0,812 | 2% | 1.465 | 0,0214 | 16,1 |
| gpt_luna | Map-Rerank | 66 | 0,068 [0,027; 0,121] | 0,693 | 22% | 1.451 | 0,0230 | 16,6 |
| gpt_luna | Step-Down | 66 | 0,225 [0,142; 0,317] | 0,825 | 10% | 5.293 | 0,0702 | 31,4 |
| gpt_luna | Reciprocal | 66 | 0,163 [0,091; 0,244] | 0,651 | 24% | 5.072 | 0,0641 | 16,7 |
| gpt_mini | Sem RAG | 66 | 0,000 [0,000; 0,000] | 0,568 | 84% | 108 | 0,0040 | 4,8 |
| gpt_mini | Stuff | 66 | 0,104 [0,044; 0,174] | 0,599 | 59% | 977 | 0,0166 | 7,2 |
| gpt_mini | Refine | 66 | 0,157 [0,085; 0,235] | 0,733 | 25% | 1.771 | 0,0430 | 19,2 |
| gpt_mini | Map-Reduce | 66 | 0,093 [0,030; 0,165] | 0,692 | 38% | 1.358 | 0,0245 | 15,3 |
| gpt_mini | Map-Rerank | 66 | 0,038 [0,004; 0,087] | 0,714 | 63% | 1.382 | 0,0270 | 11,7 |
| gpt_mini | Step-Down | 66 | 0,163 [0,087; 0,246] | 0,830 | 5% | 5.198 | 0,0938 | 27,0 |
| gpt_mini | Reciprocal | 66 | 0,106 [0,045; 0,182] | 0,671 | 29% | 5.015 | 0,0871 | 13,7 |

Friedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:

| LLM | Perguntas | χ² | p | Pares com p_Holm < 0,05 |
|---|---|---|---|---|
| gpt_luna | 66 | 29,24 | < 0,001 | Map-Reduce×Map-Rerank, Map-Rerank×Step-Down |
| gpt_mini | 66 | 26,79 | < 0,001 | Refine×Map-Rerank, Map-Rerank×Step-Down |

### medqa

Métrica principal: Acurácia.

| LLM | Método | n | Acurácia [IC 95%] | Cosseno | Recusas | Tokens (média) | Custo total US$ | Tempo s (mediana) |
|---|---|---|---|---|---|---|---|---|
| gpt_luna | Sem RAG | 100 | 0,880 [0,810; 0,940] | 0,613 | 0% | 295 | 0,0048 | 2,3 |
| gpt_luna | Stuff | 100 | 0,890 [0,820; 0,950] | 0,617 | 0% | 1.084 | 0,0127 | 4,7 |
| gpt_luna | Refine | 100 | 0,890 [0,820; 0,950] | 0,603 | 0% | 2.236 | 0,0302 | 9,4 |
| gpt_luna | Map-Reduce | 100 | 0,880 [0,810; 0,940] | 0,619 | 0% | 2.352 | 0,0290 | 8,7 |
| gpt_luna | Map-Rerank | 100 | 0,910 [0,850; 0,960] | 0,603 | 0% | 2.112 | 0,0284 | 9,8 |
| gpt_luna | Step-Down | 100 | 0,910 [0,850; 0,960] | 0,625 | 0% | 5.395 | 0,0686 | 19,5 |
| gpt_luna | Reciprocal | 100 | 0,900 [0,840; 0,950] | 0,612 | 0% | 5.077 | 0,0643 | 15,9 |
| gpt_mini | Sem RAG | 100 | 0,710 [0,620; 0,800] | 0,594 | 0% | 351 | 0,0099 | 4,5 |
| gpt_mini | Stuff | 100 | 0,680 [0,590; 0,770] | 0,590 | 0% | 1.134 | 0,0213 | 6,2 |
| gpt_mini | Refine | 100 | 0,680 [0,590; 0,770] | 0,578 | 0% | 2.620 | 0,0576 | 12,6 |
| gpt_mini | Map-Reduce | 100 | 0,730 [0,640; 0,810] | 0,604 | 0% | 2.621 | 0,0521 | 10,2 |
| gpt_mini | Map-Rerank | 100 | 0,690 [0,600; 0,780] | 0,617 | 0% | 2.123 | 0,0403 | 7,4 |
| gpt_mini | Step-Down | 100 | 0,650 [0,560; 0,740] | 0,586 | 0% | 5.761 | 0,1122 | 17,8 |
| gpt_mini | Reciprocal | 100 | 0,610 [0,520; 0,700] | 0,599 | 0% | 4.983 | 0,0893 | 14,6 |

Friedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:

| LLM | Perguntas | χ² | p | Pares com p_Holm < 0,05 |
|---|---|---|---|---|
| gpt_luna | 100 | 2,04 | 0,844 | -- |
| gpt_mini | 100 | 12,08 | 0,034 | -- |

### faquad

Métrica principal: Cobertura de tokens.

| LLM | Método | n | Cobertura de tokens [IC 95%] | Cosseno | Recusas | Tokens (média) | Custo total US$ | Tempo s (mediana) |
|---|---|---|---|---|---|---|---|---|
| gpt_luna | Sem RAG | 100 | 0,346 [0,284; 0,411] | 0,518 | 20% | 81 | 0,0022 | 1,8 |
| gpt_luna | Stuff | 100 | 0,850 [0,797; 0,899] | 0,650 | 4% | 673 | 0,0079 | 2,1 |
| gpt_luna | Refine | 100 | 0,783 [0,723; 0,840] | 0,556 | 3% | 1.266 | 0,0226 | 6,6 |
| gpt_luna | Map-Reduce | 100 | 0,796 [0,738; 0,851] | 0,615 | 2% | 1.045 | 0,0140 | 5,5 |
| gpt_luna | Map-Rerank | 100 | 0,767 [0,699; 0,832] | 0,699 | 0% | 1.110 | 0,0165 | 5,4 |
| gpt_luna | Step-Down | 100 | 0,794 [0,740; 0,845] | 0,574 | 1% | 3.816 | 0,0503 | 10,8 |
| gpt_luna | Reciprocal | 100 | 0,696 [0,626; 0,764] | 0,682 | 0% | 3.717 | 0,0475 | 7,7 |
| gpt_mini | Sem RAG | 100 | 0,411 [0,351; 0,473] | 0,488 | 15% | 122 | 0,0051 | 3,0 |
| gpt_mini | Stuff | 100 | 0,861 [0,797; 0,918] | 0,629 | 9% | 673 | 0,0114 | 2,3 |
| gpt_mini | Refine | 100 | 0,922 [0,881; 0,957] | 0,530 | 1% | 1.753 | 0,0518 | 12,0 |
| gpt_mini | Map-Reduce | 100 | 0,851 [0,796; 0,900] | 0,598 | 2% | 1.077 | 0,0209 | 4,7 |
| gpt_mini | Map-Rerank | 100 | 0,819 [0,755; 0,879] | 0,697 | 7% | 1.102 | 0,0220 | 3,8 |
| gpt_mini | Step-Down | 100 | 0,789 [0,735; 0,839] | 0,551 | 1% | 3.888 | 0,0742 | 9,5 |
| gpt_mini | Reciprocal | 100 | 0,585 [0,508; 0,662] | 0,624 | 3% | 3.729 | 0,0673 | 7,6 |

Friedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:

| LLM | Perguntas | χ² | p | Pares com p_Holm < 0,05 |
|---|---|---|---|---|
| gpt_luna | 100 | 29,86 | < 0,001 | Stuff×Map-Reduce, Stuff×Reciprocal, Map-Reduce×Reciprocal, Step-Down×Reciprocal |
| gpt_mini | 100 | 128,57 | < 0,001 | Stuff×Step-Down, Stuff×Reciprocal, Refine×Map-Reduce, Refine×Map-Rerank, Refine×Step-Down, Refine×Reciprocal, Map-Reduce×Step-Down, Map-Reduce×Reciprocal, Map-Rerank×Reciprocal, Step-Down×Reciprocal |

## 3. Efeito do RAG (método − sem RAG, na métrica principal)

### docugami_10q (Valores do gabarito na resposta)

| LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|
| gpt_luna | 0,080 | +0,069 | +0,063 | +0,122 | -0,012 | +0,145 | +0,082 |
| gpt_mini | 0,000 | +0,104* | +0,157* | +0,093* | +0,038 | +0,163* | +0,106* |
| gpt_sol | 0,089 | +0,054 | +0,188 | +0,060 | -0,036 | +0,197 | +0,024 |

\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).

### medqa (Acurácia)

| LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|
| gpt_luna | 0,880 | +0,010 | +0,010 | +0,000 | +0,030 | +0,030 | +0,020 |
| gpt_mini | 0,710 | -0,030 | -0,030 | +0,020 | -0,020 | -0,060 | -0,100 |
| gpt_sol | 0,960 | +0,000 | +0,000 | +0,000 | +0,000 | +0,000 | +0,000 |

\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).

### faquad (Cobertura de tokens)

| LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|
| gpt_luna | 0,346 | +0,504* | +0,437* | +0,450* | +0,421* | +0,449* | +0,350* |
| gpt_mini | 0,411 | +0,450* | +0,511* | +0,440* | +0,408* | +0,378* | +0,174* |
| gpt_sol | 0,482 | +0,335* | +0,246* | +0,240* | +0,201* | +0,253* | +0,269* |

\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).

## 4. Generalização (7 modelos, as mesmas 25 perguntas por dataset)

### docugami_10q (Valores do gabarito na resposta, média nas 14 perguntas com valor no gabarito)

| LLM | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| gpt_luna | 0,176 | 0,251 | 0,245 | 0,054 | 0,328 | 0,161 |
| gpt_mini | 0,089 | 0,236 | 0,018 | 0,018 | 0,107 | 0,071 |
| gpt_sol | 0,143 | 0,277 | 0,149 | 0,054 | 0,286 | 0,113 |

W de Kendall (concordância da ordem dos métodos entre os 3 LLMs): **0,857** (1 = mesma ordem em todos; 0 = nenhuma concordância).

### medqa (Acurácia, média nas 25 perguntas)

| LLM | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| gpt_luna | 0,920 | 0,880 | 0,880 | 0,920 | 0,920 | 0,920 |
| gpt_mini | 0,680 | 0,680 | 0,760 | 0,720 | 0,760 | 0,680 |
| gpt_sol | 0,960 | 0,960 | 0,960 | 0,960 | 0,960 | 0,960 |

W de Kendall (concordância da ordem dos métodos entre os 3 LLMs): **0,152** (1 = mesma ordem em todos; 0 = nenhuma concordância).

### faquad (Cobertura de tokens, média nas 25 perguntas)

| LLM | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| gpt_luna | 0,862 | 0,756 | 0,759 | 0,733 | 0,756 | 0,617 |
| gpt_mini | 0,793 | 0,876 | 0,779 | 0,756 | 0,752 | 0,566 |
| gpt_sol | 0,817 | 0,727 | 0,722 | 0,683 | 0,735 | 0,751 |

W de Kendall (concordância da ordem dos métodos entre os 3 LLMs): **0,492** (1 = mesma ordem em todos; 0 = nenhuma concordância).

## 5. Variação do modelo (desvio-padrão entre as 3 execuções do Stuff, por pergunta)

| Dataset | LLM | Perguntas | Desvio médio na métrica principal |
|---|---|---|---|
| docugami_10q | gpt_luna | 14 | 0,009 |
| docugami_10q | gpt_sol | 14 | 0,053 |
| faquad | gpt_luna | 25 | 0,051 |
| faquad | gpt_sol | 25 | 0,023 |
| medqa | gpt_luna | 25 | 0,000 |
| medqa | gpt_sol | 25 | 0,000 |

Compare com as diferenças entre métodos da seção 2: diferenças menores que esse desvio não se distinguem da variação do próprio modelo.

## 6. Comparação com Şakar & Emekci (2025), Tabela 1 (mediana da similaridade de cosseno)

### docugami_10q

| Método | Paper-âncora | gpt_luna | gpt_mini | gpt_sol |
|---|---|---|---|---|
| Reciprocal | 0,971 | 0,666 | 0,686 | 0,595 |
| Stuff | 0,895 | 0,784 | 0,421 | 0,747 |
| Step-Down | 0,890 | 0,838 | 0,840 | 0,818 |
| Map-Reduce | 0,873 | 0,837 | 0,785 | 0,770 |
| Refine | 0,836 | 0,788 | 0,819 | 0,806 |
| Map-Rerank | 0,801 | 0,696 | 0,705 | 0,656 |

### medqa

| Método | Paper-âncora | gpt_luna | gpt_mini | gpt_sol |
|---|---|---|---|---|
| Step-Down | 0,678 | 0,635 | 0,617 | 0,637 |
| Reciprocal | 0,673 | 0,623 | 0,616 | 0,643 |
| Stuff | 0,641 | 0,620 | 0,617 | 0,637 |
| Refine | 0,609 | 0,614 | 0,615 | 0,651 |
| Map-Reduce | 0,569 | 0,620 | 0,640 | 0,640 |
| Map-Rerank | 0,148 | 0,612 | 0,642 | 0,618 |

O paper-âncora agrega vários LLMs, embeddings e bancos vetoriais, e não informa o embedding da similaridade: a comparação é indicativa.

## 7. RAGAS (juiz gpt-4o-mini; métricas complementares)

### Faithfulness (média)

| Dataset | LLM | Stuff | Refine | Map-Reduce | Map-Rerank |
|---|---|---|---|---|---|
| docugami_10q | gpt_luna | 0,869 | 0,792 | 0,559 | 0,719 |
| docugami_10q | gpt_mini | 0,461 | 0,678 | 0,377 | 0,666 |
| docugami_10q | gpt_sol | 0,777 | 0,726 | 0,686 | 0,766 |
| faquad | gpt_luna | 0,905 | 0,845 | 0,867 | 0,878 |
| faquad | gpt_mini | 0,894 | 0,906 | 0,759 | 0,905 |
| faquad | gpt_sol | 0,907 | 0,840 | 0,779 | 0,800 |
| medqa | gpt_luna | 0,186 | 0,297 | 0,172 | 0,171 |
| medqa | gpt_mini | 0,232 | 0,282 | 0,199 | 0,236 |
| medqa | gpt_sol | 0,204 | 0,257 | 0,199 | 0,221 |

### Answer Relevancy (média)

| Dataset | LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|---|
| docugami_10q | gpt_luna | 0,575 | 0,339 | 0,457 | 0,609 | 0,256 | 0,521 | 0,315 |
| docugami_10q | gpt_mini | 0,130 | 0,351 | 0,608 | 0,457 | 0,268 | 0,547 | 0,364 |
| docugami_10q | gpt_sol | 0,297 | 0,215 | 0,356 | 0,346 | 0,178 | 0,459 | 0,331 |
| faquad | gpt_luna | 0,600 | 0,664 | 0,677 | 0,804 | 0,396 | 0,730 | 0,435 |
| faquad | gpt_mini | 0,684 | 0,755 | 0,786 | 0,844 | 0,569 | 0,788 | 0,548 |
| faquad | gpt_sol | 0,724 | 0,589 | 0,614 | 0,712 | 0,278 | 0,765 | 0,454 |
| medqa | gpt_luna | 0,528 | 0,527 | 0,530 | 0,525 | 0,514 | 0,517 | 0,508 |
| medqa | gpt_mini | 0,551 | 0,564 | 0,550 | 0,561 | 0,522 | 0,561 | 0,516 |
| medqa | gpt_sol | 0,542 | 0,548 | 0,533 | 0,524 | 0,534 | 0,526 | 0,465 |

O gpt-4o-mini é juiz e também um dos modelos testados (viés de autopreferência; Panickssery et al., 2024): as notas dele não entram na comparação entre LLMs.

### Context Precision e Context Recall (uma vez por pergunta)

| Dataset | Perguntas | Context Precision | Context Recall | Spearman com o Hit@4 |
|---|---|---|---|---|
| docugami_10q | 100 | 0,549 | 0,449 | 0,107 (p=0,291) |
| medqa | 100 | 0,439 | 0,523 | -- |
| faquad | 100 | 0,872 | 0,940 | 0,341 (p=< 0,001) |

## 8. Bancos vetoriais: ChromaDB (HNSW, aproximado) × FAISS (IndexFlatL2, exato)

| Dataset (vetores) | Index. Chroma (s) | Index. FAISS (s) | Consulta Chroma (ms) | Consulta FAISS (ms) | Mem. Chroma (MB) | Mem. FAISS (MB) | Top-4 idêntico | Sobreposição top-4 | Hit@4 Chroma × FAISS |
|---|---|---|---|---|---|---|---|---|---|
| faquad (318) | 0,63 | 0,001 | 3,78 | 0,05 | 29,3 | 1,6 | 100% | 100% | 0,930 × 0,930 |
| docugami_10q (4.463) | 5,77 | 0,017 | 2,75 | 1,20 | 184,7 | 17,4 | 100% | 100% | 0,849 × 0,849 |
| medqa (121.553) | 260,83 | 0,304 | 5,00 | 36,47 | 801,5 | 474,8 | 82% | 94% | -- |

Top-4 idêntico: fração das consultas em que os dois bancos entregam os mesmos 4 fragmentos ao LLM (nelas, a resposta do RAG seria a mesma com o FAISS). Medido num índice Chroma recriado só pro comparativo; o HNSW não é determinístico, então a tabela abaixo compara com o índice dos experimentos.

### Efeito na resposta: Stuff do gpt-4o-mini com o top-4 do FAISS (100 perguntas)

| Dataset | Métrica | Top-4 igual ao dos experimentos | Perguntas refeitas | Chroma | FAISS | Nas refeitas (Chroma × FAISS) | Só um acerta (Chroma × FAISS) | p |
|---|---|---|---|---|---|---|---|---|
| docugami_10q | Valores do gabarito na resposta | 99% | 1 | 0,104 | 0,104 | 1,000 × 1,000 | -- | -- |
| medqa | Acurácia | 84% | 16 | 0,680 | 0,690 | 0,750 × 0,812 | 0 × 1 | 1,000 |
| faquad | Cobertura de tokens | 100% | 0 | 0,861 | 0,861 | -- | -- | -- |

Só as perguntas com top-4 diferente são respondidas de novo com os fragmentos do FAISS; nas outras, o LLM recebe a mesma entrada. p: McNemar exato (acurácia) ou Wilcoxon pareado, nas perguntas refeitas. O gpt-4o-mini roda com temperatura 0, mas a API não garante saída idêntica: parte da diferença pode ser variação do próprio modelo.
