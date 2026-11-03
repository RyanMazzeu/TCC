# Análise da rodada dos benchmarks

12150 execuções nos logs `benchmarks_*.jsonl`.

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
| claude_haiku | Sem RAG | 66 | 0,009 [0,000; 0,023] | 0,799 | 87% | 386 | 0,0161 | 7,5 |
| claude_haiku | Stuff | 66 | 0,189 [0,111; 0,274] | 0,787 | 59% | 1.578 | 0,0250 | 5,3 |
| claude_haiku | Refine | 66 | 0,158 [0,087; 0,237] | 0,788 | 46% | 3.386 | 0,0736 | 10,0 |
| claude_haiku | Map-Reduce | 66 | 0,143 [0,073; 0,222] | 0,816 | 14% | 2.380 | 0,0409 | 9,1 |
| claude_haiku | Map-Rerank | 66 | 0,045 [0,008; 0,095] | 0,703 | 60% | 2.323 | 0,0430 | 7,9 |
| claude_haiku | Step-Down | 66 | 0,275 [0,188; 0,370] | 0,832 | 31% | 9.137 | 0,1528 | 18,0 |
| claude_haiku | Reciprocal | 66 | 0,148 [0,076; 0,229] | 0,663 | 42% | 7.755 | 0,1107 | 14,1 |
| deepseek | Sem RAG | 66 | 0,067 [0,021; 0,124] | 0,823 | 64% | 194 | 0,0300 | 8,5 |
| deepseek | Stuff | 66 | 0,127 [0,057; 0,208] | 0,702 | 68% | 950 | 0,0901 | 5,8 |
| deepseek | Refine | 66 | 0,078 [0,027; 0,138] | 0,717 | 62% | 1.494 | 0,1529 | 13,3 |
| deepseek | Map-Reduce | 66 | 0,100 [0,039; 0,172] | 0,667 | 71% | 1.309 | 0,1255 | 13,2 |
| deepseek | Map-Rerank | 66 | 0,066 [0,024; 0,117] | 0,653 | 59% | 1.375 | 0,1387 | 13,2 |
| deepseek | Step-Down | 66 | 0,150 [0,084; 0,226] | 0,811 | 31% | 5.127 | 0,5012 | 30,2 |
| deepseek | Reciprocal | 66 | 0,056 [0,016; 0,108] | 0,615 | 52% | 4.890 | 0,4709 | 25,9 |
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
| qwen | Sem RAG | 66 | 0,026 [0,007; 0,052] | 0,812 | 61% | 315 | 0,0073 | 9,5 |
| qwen | Stuff | 66 | 0,109 [0,045; 0,185] | 0,648 | 66% | 1.585 | 0,0310 | 4,9 |
| qwen | Refine | 66 | 0,073 [0,031; 0,124] | 0,771 | 28% | 2.398 | 0,0457 | 12,3 |
| qwen | Map-Reduce | 66 | 0,075 [0,023; 0,138] | 0,769 | 79% | 1.570 | 0,0275 | 7,6 |
| qwen | Map-Rerank | 66 | 0,054 [0,013; 0,106] | 0,676 | 72% | 1.635 | 0,0293 | 7,7 |
| qwen | Step-Down | 66 | 0,143 [0,074; 0,220] | 0,807 | 12% | 6.239 | 0,1106 | 13,5 |
| qwen | Reciprocal | 66 | 0,064 [0,015; 0,125] | 0,622 | 37% | 6.245 | 0,1125 | 18,4 |

Friedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:

| LLM | Perguntas | χ² | p | Pares com p_Holm < 0,05 |
|---|---|---|---|---|
| claude_haiku | 66 | 43,42 | < 0,001 | Stuff×Map-Rerank, Map-Reduce×Step-Down, Map-Rerank×Step-Down |
| deepseek | 66 | 15,62 | 0,008 | -- |
| gpt_luna | 66 | 29,24 | < 0,001 | Map-Reduce×Map-Rerank, Map-Rerank×Step-Down |
| gpt_mini | 66 | 26,79 | < 0,001 | Refine×Map-Rerank, Map-Rerank×Step-Down |
| qwen | 66 | 13,50 | 0,019 | -- |

### medqa

Métrica principal: Acurácia.

| LLM | Método | n | Acurácia [IC 95%] | Cosseno | Recusas | Tokens (média) | Custo total US$ | Tempo s (mediana) |
|---|---|---|---|---|---|---|---|---|
| claude_haiku | Sem RAG | 100 | 0,830 [0,750; 0,900] | 0,607 | 0% | 614 | 0,0152 | 8,2 |
| claude_haiku | Stuff | 100 | 0,800 [0,720; 0,870] | 0,604 | 1% | 1.892 | 0,0275 | 13,3 |
| claude_haiku | Refine | 100 | 0,810 [0,730; 0,880] | 0,596 | 0% | 4.343 | 0,0740 | 13,8 |
| claude_haiku | Map-Reduce | 100 | 0,850 [0,780; 0,920] | 0,616 | 8% | 4.239 | 0,0643 | 13,1 |
| claude_haiku | Map-Rerank | 100 | 0,920 [0,860; 0,970] | 0,606 | 0% | 3.805 | 0,0674 | 10,4 |
| claude_haiku | Step-Down | 100 | 0,890 [0,830; 0,950] | 0,615 | 0% | 10.063 | 0,1511 | 16,9 |
| claude_haiku | Reciprocal | 100 | 0,880 [0,810; 0,940] | 0,596 | 1% | 8.741 | 0,1244 | 13,9 |
| deepseek | Sem RAG | 100 | 0,800 [0,720; 0,870] | 0,611 | 0% | 358 | 0,0427 | 6,9 |
| deepseek | Stuff | 100 | 0,810 [0,730; 0,880] | 0,601 | 1% | 1.137 | 0,1141 | 8,2 |
| deepseek | Refine | 100 | 0,880 [0,810; 0,940] | 0,656 | 0% | 2.476 | 0,2543 | 17,0 |
| deepseek | Map-Reduce | 100 | 0,810 [0,730; 0,880] | 0,611 | 0% | 2.702 | 0,2756 | 17,9 |
| deepseek | Map-Rerank | 100 | 0,880 [0,820; 0,940] | 0,820 | 0% | 2.014 | 0,1928 | 10,4 |
| deepseek | Step-Down | 100 | 0,910 [0,850; 0,960] | 0,619 | 0% | 5.764 | 0,5767 | 29,3 |
| deepseek | Reciprocal | 100 | 0,830 [0,750; 0,900] | 0,771 | 0% | 4.862 | 0,4637 | 21,1 |
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
| qwen | Sem RAG | 100 | 0,720 [0,630; 0,810] | 0,541 | 0% | 1.040 | 0,0239 | 12,9 |
| qwen | Stuff | 100 | 0,740 [0,650; 0,820] | 0,546 | 0% | 1.589 | 0,0309 | 13,6 |
| qwen | Refine | 100 | 0,790 [0,710; 0,870] | 0,533 | 1% | 5.245 | 0,1028 | 13,6 |
| qwen | Map-Reduce | 100 | 0,800 [0,720; 0,870] | 0,588 | 3% | 4.420 | 0,0846 | 11,4 |
| qwen | Map-Rerank | 100 | 0,750 [0,660; 0,830] | 0,582 | 0% | 2.873 | 0,0552 | 10,3 |
| qwen | Step-Down | 100 | 0,830 [0,750; 0,900] | 0,587 | 0% | 8.806 | 0,1655 | 29,2 |
| qwen | Reciprocal | 100 | 0,760 [0,670; 0,840] | 0,578 | 0% | 6.018 | 0,1095 | 22,0 |

Friedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:

| LLM | Perguntas | χ² | p | Pares com p_Holm < 0,05 |
|---|---|---|---|---|
| claude_haiku | 100 | 16,54 | 0,005 | Stuff×Map-Rerank, Stuff×Step-Down |
| deepseek | 100 | 15,76 | 0,008 | -- |
| gpt_luna | 100 | 2,04 | 0,844 | -- |
| gpt_mini | 100 | 12,08 | 0,034 | -- |
| qwen | 100 | 7,84 | 0,165 | -- |

### faquad

Métrica principal: Cobertura de tokens.

| LLM | Método | n | Cobertura de tokens [IC 95%] | Cosseno | Recusas | Tokens (média) | Custo total US$ | Tempo s (mediana) |
|---|---|---|---|---|---|---|---|---|
| claude_haiku | Sem RAG | 100 | 0,496 [0,436; 0,554] | 0,486 | 43% | 334 | 0,0137 | 3,4 |
| claude_haiku | Stuff | 100 | 0,921 [0,875; 0,960] | 0,596 | 6% | 1.299 | 0,0175 | 2,2 |
| claude_haiku | Refine | 100 | 0,885 [0,833; 0,931] | 0,546 | 6% | 3.137 | 0,0688 | 7,8 |
| claude_haiku | Map-Reduce | 100 | 0,838 [0,779; 0,891] | 0,588 | 2% | 2.048 | 0,0307 | 6,2 |
| claude_haiku | Map-Rerank | 100 | 0,711 [0,632; 0,787] | 0,583 | 5% | 2.134 | 0,0365 | 5,8 |
| claude_haiku | Step-Down | 100 | 0,915 [0,878; 0,947] | 0,554 | 1% | 8.164 | 0,1314 | 15,4 |
| claude_haiku | Reciprocal | 100 | 0,775 [0,707; 0,838] | 0,605 | 0% | 7.128 | 0,1005 | 9,6 |
| deepseek | Sem RAG | 100 | 0,485 [0,415; 0,556] | 0,493 | 22% | 192 | 0,0299 | 8,1 |
| deepseek | Stuff | 100 | 0,866 [0,806; 0,920] | 0,675 | 5% | 792 | 0,0742 | 4,9 |
| deepseek | Refine | 100 | 0,858 [0,798; 0,911] | 0,623 | 5% | 1.408 | 0,1473 | 13,4 |
| deepseek | Map-Reduce | 100 | 0,859 [0,800; 0,911] | 0,651 | 7% | 1.200 | 0,1162 | 12,7 |
| deepseek | Map-Rerank | 100 | 0,795 [0,724; 0,861] | 0,688 | 9% | 1.285 | 0,1303 | 11,0 |
| deepseek | Step-Down | 100 | 0,866 [0,817; 0,909] | 0,567 | 1% | 4.622 | 0,4548 | 22,5 |
| deepseek | Reciprocal | 100 | 0,754 [0,686; 0,817] | 0,649 | 3% | 4.461 | 0,4322 | 17,0 |
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
| qwen | Sem RAG | 100 | 0,502 [0,444; 0,560] | 0,484 | 28% | 251 | 0,0058 | 9,9 |
| qwen | Stuff | 100 | 0,875 [0,813; 0,930] | 0,660 | 0% | 719 | 0,0125 | 3,0 |
| qwen | Refine | 100 | 0,838 [0,777; 0,894] | 0,530 | 1% | 1.748 | 0,0337 | 11,8 |
| qwen | Map-Reduce | 100 | 0,853 [0,797; 0,904] | 0,603 | 7% | 1.150 | 0,0204 | 6,1 |
| qwen | Map-Rerank | 100 | 0,795 [0,723; 0,861] | 0,670 | 9% | 1.198 | 0,0215 | 5,3 |
| qwen | Step-Down | 100 | 0,820 [0,766; 0,870] | 0,545 | 1% | 4.182 | 0,0747 | 12,1 |
| qwen | Reciprocal | 100 | 0,651 [0,577; 0,720] | 0,555 | 4% | 3.939 | 0,0696 | 7,8 |

Friedman entre os 6 métodos (perguntas como blocos) e pares significativos após Holm:

| LLM | Perguntas | χ² | p | Pares com p_Holm < 0,05 |
|---|---|---|---|---|
| claude_haiku | 100 | 50,84 | < 0,001 | Stuff×Map-Reduce, Stuff×Map-Rerank, Stuff×Reciprocal, Refine×Map-Rerank, Refine×Reciprocal, Map-Reduce×Map-Rerank, Map-Reduce×Step-Down, Map-Rerank×Step-Down, Step-Down×Reciprocal |
| deepseek | 100 | 22,00 | < 0,001 | Stuff×Reciprocal, Refine×Reciprocal, Map-Reduce×Reciprocal, Step-Down×Reciprocal |
| gpt_luna | 100 | 29,86 | < 0,001 | Stuff×Map-Reduce, Stuff×Reciprocal, Map-Reduce×Reciprocal, Step-Down×Reciprocal |
| gpt_mini | 100 | 128,57 | < 0,001 | Stuff×Step-Down, Stuff×Reciprocal, Refine×Map-Reduce, Refine×Map-Rerank, Refine×Step-Down, Refine×Reciprocal, Map-Reduce×Step-Down, Map-Reduce×Reciprocal, Map-Rerank×Reciprocal, Step-Down×Reciprocal |
| qwen | 100 | 66,12 | < 0,001 | Stuff×Map-Rerank, Stuff×Reciprocal, Refine×Reciprocal, Map-Reduce×Reciprocal, Map-Rerank×Reciprocal, Step-Down×Reciprocal |

## 3. Efeito do RAG (método − sem RAG, na métrica principal)

### docugami_10q (Valores do gabarito na resposta)

| LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|
| claude_haiku | 0,009 | +0,180* | +0,150* | +0,135* | +0,037 | +0,266* | +0,139* |
| claude_sonnet | 0,048 | +0,051 | +0,200 | +0,155 | -0,048 | +0,209 | +0,048 |
| deepseek | 0,067 | +0,060 | +0,010 | +0,033 | -0,002 | +0,082 | -0,012 |
| gpt_luna | 0,080 | +0,069 | +0,063 | +0,122 | -0,012 | +0,145 | +0,082 |
| gpt_mini | 0,000 | +0,104* | +0,157* | +0,093* | +0,038 | +0,163* | +0,106* |
| gpt_sol | 0,089 | +0,054 | +0,188 | +0,060 | -0,036 | +0,197 | +0,024 |
| qwen | 0,026 | +0,083 | +0,047 | +0,049 | +0,028 | +0,116* | +0,038 |

\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).

### medqa (Acurácia)

| LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|
| claude_haiku | 0,830 | -0,030 | -0,020 | +0,020 | +0,090* | +0,060 | +0,050 |
| claude_sonnet | 0,960 | -0,040 | +0,000 | +0,000 | +0,000 | +0,000 | +0,000 |
| deepseek | 0,800 | +0,010 | +0,080 | +0,010 | +0,080 | +0,110* | +0,030 |
| gpt_luna | 0,880 | +0,010 | +0,010 | +0,000 | +0,030 | +0,030 | +0,020 |
| gpt_mini | 0,710 | -0,030 | -0,030 | +0,020 | -0,020 | -0,060 | -0,100 |
| gpt_sol | 0,960 | +0,000 | +0,000 | +0,000 | +0,000 | +0,000 | +0,000 |
| qwen | 0,720 | +0,020 | +0,070 | +0,080 | +0,030 | +0,110 | +0,040 |

\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).

### faquad (Cobertura de tokens)

| LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|
| claude_haiku | 0,496 | +0,425* | +0,390* | +0,342* | +0,216* | +0,420* | +0,279* |
| claude_sonnet | 0,501 | +0,407* | +0,419* | +0,285* | +0,060 | +0,400* | +0,060 |
| deepseek | 0,485 | +0,381* | +0,372* | +0,374* | +0,310* | +0,381* | +0,269* |
| gpt_luna | 0,346 | +0,504* | +0,437* | +0,450* | +0,421* | +0,449* | +0,350* |
| gpt_mini | 0,411 | +0,450* | +0,511* | +0,440* | +0,408* | +0,378* | +0,174* |
| gpt_sol | 0,482 | +0,335* | +0,246* | +0,240* | +0,201* | +0,253* | +0,269* |
| qwen | 0,502 | +0,372* | +0,336* | +0,350* | +0,292* | +0,318* | +0,149* |

\* diferença com p_Holm < 0,05 (Wilcoxon pareado contra o sem RAG).

## 4. Generalização (7 modelos, as mesmas 25 perguntas por dataset)

### docugami_10q (Valores do gabarito na resposta, média nas 14 perguntas com valor no gabarito)

| LLM | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| claude_haiku | 0,215 | 0,140 | 0,185 | 0,089 | 0,219 | 0,232 |
| claude_sonnet | 0,099 | 0,248 | 0,203 | 0,000 | 0,256 | 0,095 |
| deepseek | 0,071 | 0,125 | 0,078 | 0,054 | 0,090 | 0,006 |
| gpt_luna | 0,176 | 0,251 | 0,245 | 0,054 | 0,328 | 0,161 |
| gpt_mini | 0,089 | 0,236 | 0,018 | 0,018 | 0,107 | 0,071 |
| gpt_sol | 0,143 | 0,277 | 0,149 | 0,054 | 0,286 | 0,113 |
| qwen | 0,143 | 0,099 | 0,143 | 0,036 | 0,185 | 0,000 |

W de Kendall (concordância da ordem dos métodos entre os 7 LLMs): **0,639** (p = < 0,001; 1 = mesma ordem em todos; 0 = nenhuma concordância).

### medqa (Acurácia, média nas 25 perguntas)

| LLM | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| claude_haiku | 0,800 | 0,880 | 0,840 | 0,960 | 0,880 | 0,920 |
| claude_sonnet | 0,920 | 0,960 | 0,960 | 0,960 | 0,960 | 0,960 |
| deepseek | 0,880 | 0,920 | 0,880 | 0,920 | 1,000 | 0,960 |
| gpt_luna | 0,920 | 0,880 | 0,880 | 0,920 | 0,920 | 0,920 |
| gpt_mini | 0,680 | 0,680 | 0,760 | 0,720 | 0,760 | 0,680 |
| gpt_sol | 0,960 | 0,960 | 0,960 | 0,960 | 0,960 | 0,960 |
| qwen | 0,800 | 0,800 | 0,880 | 0,800 | 0,960 | 0,880 |

W de Kendall (concordância da ordem dos métodos entre os 7 LLMs): **0,236** (p = 0,142; 1 = mesma ordem em todos; 0 = nenhuma concordância).

### faquad (Cobertura de tokens, média nas 25 perguntas)

| LLM | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| claude_haiku | 0,914 | 0,883 | 0,785 | 0,715 | 0,913 | 0,681 |
| claude_sonnet | 0,908 | 0,919 | 0,786 | 0,561 | 0,901 | 0,560 |
| deepseek | 0,856 | 0,750 | 0,844 | 0,707 | 0,861 | 0,687 |
| gpt_luna | 0,862 | 0,756 | 0,759 | 0,733 | 0,756 | 0,617 |
| gpt_mini | 0,793 | 0,876 | 0,779 | 0,756 | 0,752 | 0,566 |
| gpt_sol | 0,817 | 0,727 | 0,722 | 0,683 | 0,735 | 0,751 |
| qwen | 0,841 | 0,746 | 0,803 | 0,761 | 0,815 | 0,683 |

W de Kendall (concordância da ordem dos métodos entre os 7 LLMs): **0,606** (p = < 0,001; 1 = mesma ordem em todos; 0 = nenhuma concordância).

## 5. Variação do modelo (desvio-padrão entre as 3 execuções do Stuff, por pergunta)

| Dataset | LLM | Perguntas | Desvio médio na métrica principal |
|---|---|---|---|
| docugami_10q | claude_haiku | 14 | 0,063 |
| docugami_10q | claude_sonnet | 14 | 0,014 |
| docugami_10q | gpt_luna | 14 | 0,009 |
| docugami_10q | gpt_sol | 14 | 0,053 |
| faquad | claude_haiku | 25 | 0,010 |
| faquad | claude_sonnet | 25 | 0,005 |
| faquad | gpt_luna | 25 | 0,051 |
| faquad | gpt_sol | 25 | 0,023 |
| medqa | claude_haiku | 25 | 0,000 |
| medqa | claude_sonnet | 25 | 0,023 |
| medqa | gpt_luna | 25 | 0,000 |
| medqa | gpt_sol | 25 | 0,000 |

Compare com as diferenças entre métodos da seção 2: diferenças menores que esse desvio não se distinguem da variação do próprio modelo.

## 6. Comparação com Şakar & Emekci (2025), Tabela 1 (mediana da similaridade de cosseno)

### docugami_10q

| Método | Paper-âncora | claude_haiku | claude_sonnet | deepseek | gpt_luna | gpt_mini | gpt_sol | qwen |
|---|---|---|---|---|---|---|---|---|
| Reciprocal | 0,971 | 0,667 | 0,570 | 0,638 | 0,666 | 0,686 | 0,595 | 0,639 |
| Stuff | 0,895 | 0,792 | 0,776 | 0,713 | 0,784 | 0,421 | 0,747 | 0,704 |
| Step-Down | 0,890 | 0,839 | 0,831 | 0,828 | 0,838 | 0,840 | 0,818 | 0,823 |
| Map-Reduce | 0,873 | 0,809 | 0,837 | 0,708 | 0,837 | 0,785 | 0,770 | 0,746 |
| Refine | 0,836 | 0,792 | 0,796 | 0,716 | 0,788 | 0,819 | 0,806 | 0,773 |
| Map-Rerank | 0,801 | 0,710 | 0,625 | 0,655 | 0,696 | 0,705 | 0,656 | 0,671 |

### medqa

| Método | Paper-âncora | claude_haiku | claude_sonnet | deepseek | gpt_luna | gpt_mini | gpt_sol | qwen |
|---|---|---|---|---|---|---|---|---|
| Step-Down | 0,678 | 0,635 | 0,572 | 0,636 | 0,635 | 0,617 | 0,637 | 0,594 |
| Reciprocal | 0,673 | 0,612 | 0,562 | 0,871 | 0,623 | 0,616 | 0,643 | 0,581 |
| Stuff | 0,641 | 0,626 | 0,555 | 0,611 | 0,620 | 0,617 | 0,637 | 0,543 |
| Refine | 0,609 | 0,605 | 0,585 | 0,641 | 0,614 | 0,615 | 0,651 | 0,542 |
| Map-Reduce | 0,569 | 0,627 | 0,570 | 0,631 | 0,620 | 0,640 | 0,640 | 0,601 |
| Map-Rerank | 0,148 | 0,614 | 0,569 | 0,891 | 0,612 | 0,642 | 0,618 | 0,575 |

O paper-âncora agrega vários LLMs, embeddings e bancos vetoriais, e não informa o embedding da similaridade: a comparação é indicativa.

### Com o bge-small-en-v1.5, o embedding do artigo-base (mediana de todas as respostas, 7 LLMs)

| Método | docugami_10q: artigo | docugami_10q: este trabalho | medqa: artigo | medqa: este trabalho |
|---|---|---|---|---|
| Stuff | 0,895 | 0,839 | 0,641 | 0,757 |
| Refine | 0,836 | 0,865 | 0,609 | 0,760 |
| Map-Reduce | 0,873 | 0,841 | 0,569 | 0,769 |
| Map-Rerank | 0,801 | 0,705 | 0,148 | 0,773 |
| Step-Down | 0,890 | 0,881 | 0,678 | 0,769 |
| Reciprocal | 0,971 | 0,684 | 0,673 | 0,776 |

Calculada por `similaridade_bge_small.py`, com o mesmo cálculo da similaridade dos experimentos.

## 7. RAGAS (juiz gpt-4o-mini; métricas complementares)

### Faithfulness (média)

| Dataset | LLM | Stuff | Refine | Map-Reduce | Map-Rerank |
|---|---|---|---|---|---|
| docugami_10q | claude_haiku | 0,907 | 0,813 | 0,669 | 0,645 |
| docugami_10q | claude_sonnet | 0,893 | 0,883 | 0,648 | 0,713 |
| docugami_10q | deepseek | 0,766 | 0,716 | 0,362 | 0,704 |
| docugami_10q | gpt_luna | 0,869 | 0,792 | 0,559 | 0,719 |
| docugami_10q | gpt_mini | 0,461 | 0,678 | 0,377 | 0,666 |
| docugami_10q | gpt_sol | 0,777 | 0,726 | 0,686 | 0,766 |
| docugami_10q | qwen | 0,684 | 0,828 | 0,313 | 0,631 |
| faquad | claude_haiku | 0,927 | 0,846 | 0,831 | 0,737 |
| faquad | claude_sonnet | 0,912 | 0,886 | 0,819 | 0,747 |
| faquad | deepseek | 0,906 | 0,886 | 0,822 | 0,872 |
| faquad | gpt_luna | 0,905 | 0,845 | 0,867 | 0,878 |
| faquad | gpt_mini | 0,894 | 0,906 | 0,759 | 0,905 |
| faquad | gpt_sol | 0,907 | 0,840 | 0,779 | 0,800 |
| faquad | qwen | 0,883 | 0,791 | 0,842 | 0,900 |
| medqa | claude_haiku | 0,190 | 0,408 | 0,183 | 0,138 |
| medqa | claude_sonnet | 0,190 | 0,403 | 0,193 | 0,210 |
| medqa | deepseek | 0,222 | 0,257 | 0,162 | 0,123 |
| medqa | gpt_luna | 0,186 | 0,297 | 0,172 | 0,171 |
| medqa | gpt_mini | 0,232 | 0,282 | 0,199 | 0,236 |
| medqa | gpt_sol | 0,204 | 0,257 | 0,199 | 0,221 |
| medqa | qwen | 0,289 | 0,401 | 0,211 | 0,221 |

### Answer Relevancy (média)

| Dataset | LLM | Sem RAG | Stuff | Refine | Map-Reduce | Map-Rerank | Step-Down | Reciprocal |
|---|---|---|---|---|---|---|---|---|
| docugami_10q | claude_haiku | 0,008 | 0,237 | 0,247 | 0,226 | 0,169 | 0,265 | 0,254 |
| docugami_10q | claude_sonnet | 0,035 | 0,191 | 0,295 | 0,481 | 0,250 | 0,359 | 0,237 |
| docugami_10q | deepseek | 0,277 | 0,237 | 0,226 | 0,288 | 0,245 | 0,362 | 0,182 |
| docugami_10q | gpt_luna | 0,575 | 0,339 | 0,457 | 0,609 | 0,256 | 0,521 | 0,315 |
| docugami_10q | gpt_mini | 0,130 | 0,351 | 0,608 | 0,457 | 0,268 | 0,547 | 0,364 |
| docugami_10q | gpt_sol | 0,297 | 0,215 | 0,356 | 0,346 | 0,178 | 0,459 | 0,331 |
| docugami_10q | qwen | 0,349 | 0,230 | 0,274 | 0,167 | 0,160 | 0,403 | 0,250 |
| faquad | claude_haiku | 0,331 | 0,747 | 0,685 | 0,742 | 0,507 | 0,558 | 0,583 |
| faquad | claude_sonnet | 0,240 | 0,693 | 0,623 | 0,667 | 0,312 | 0,604 | 0,343 |
| faquad | deepseek | 0,638 | 0,622 | 0,713 | 0,699 | 0,455 | 0,635 | 0,428 |
| faquad | gpt_luna | 0,600 | 0,664 | 0,677 | 0,804 | 0,396 | 0,730 | 0,435 |
| faquad | gpt_mini | 0,684 | 0,755 | 0,786 | 0,844 | 0,569 | 0,788 | 0,548 |
| faquad | gpt_sol | 0,724 | 0,589 | 0,614 | 0,712 | 0,278 | 0,765 | 0,454 |
| faquad | qwen | 0,580 | 0,672 | 0,581 | 0,818 | 0,525 | 0,671 | 0,569 |
| medqa | claude_haiku | 0,543 | 0,549 | 0,502 | 0,542 | 0,528 | 0,543 | 0,514 |
| medqa | claude_sonnet | 0,523 | 0,541 | 0,479 | 0,527 | 0,502 | 0,530 | 0,506 |
| medqa | deepseek | 0,522 | 0,545 | 0,488 | 0,538 | 0,441 | 0,547 | 0,467 |
| medqa | gpt_luna | 0,528 | 0,527 | 0,530 | 0,525 | 0,514 | 0,517 | 0,508 |
| medqa | gpt_mini | 0,551 | 0,564 | 0,550 | 0,561 | 0,522 | 0,561 | 0,516 |
| medqa | gpt_sol | 0,542 | 0,548 | 0,533 | 0,524 | 0,534 | 0,526 | 0,465 |
| medqa | qwen | 0,507 | 0,527 | 0,476 | 0,533 | 0,491 | 0,536 | 0,492 |

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

## 9. Efeito do porte (FrugalGPT): econômico com o melhor método × intermediário com o Stuff

| Dataset | Família | Econômico + melhor método (camada principal) | Econômico | Intermediário + Stuff | p | Custo do intermediário / econômico | Melhor(es) nas 25 (econômico) | Melhor(es) nas 25 (intermediário) |
|---|---|---|---|---|---|---|---|---|
| docugami_10q | OpenAI | gpt_luna + Step-Down | 0,328 | 0,143 | 0,041 | 3,4× | Step-Down | Step-Down |
| docugami_10q | Anthropic | claude_haiku + Step-Down | 0,219 | 0,099 | 0,109 | 3,3× | Reciprocal | Step-Down |
| medqa | OpenAI | gpt_luna + Map-Rerank | 0,920 | 0,960 | 1,000 | 8,7× | Map-Rerank, Step-Down, Reciprocal, Stuff | Map-Reduce, Map-Rerank, Step-Down, Reciprocal, Refine, Stuff |
| medqa | Anthropic | claude_haiku + Map-Rerank | 0,960 | 0,920 | 1,000 | 7,4× | Map-Rerank | Map-Reduce, Map-Rerank, Step-Down, Reciprocal, Refine |
| faquad | OpenAI | gpt_luna + Stuff | 0,862 | 0,817 | 0,463 | 19,5× | Stuff | Stuff |
| faquad | Anthropic | claude_haiku + Stuff | 0,914 | 0,908 | 0,686 | 18,3× | Stuff | Refine |

Médias nas perguntas da camada de generalização (25; no Docugami, as com valor no gabarito), pareadas. p: McNemar exato (acurácia) ou Wilcoxon pareado. O melhor método do econômico vem das 100 perguntas da camada principal, que incluem essas 25.
