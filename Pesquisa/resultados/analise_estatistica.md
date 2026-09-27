# Análise estatística — resultados finais

Gerado por `scripts/analisar_resultados.py` (não editar à mão; rodar de novo o script).

## 1. Comparativo dos 6 métodos (v3: 27 perguntas × 6 métodos = 162 execuções)

### 1.1 Similaridade de cosseno média

| Domínio (n) | Stuff | Refine | Map-Reduce | Map-Rerank | Query Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| Revalida (12) | 0,555 | 0,687 | 0,684 | 0,714 | 0,698 | 0,586 |
| Grendene (7) | 0,602 | 0,557 | 0,557 | 0,746 | 0,582 | 0,780 |
| SBC (8) | 0,825 | 0,701 | 0,789 | 0,786 | 0,746 | 0,719 |
| **Geral (27)** | 0,647 | 0,657 | 0,682 | 0,744 | 0,682 | 0,676 |
| DP | 0,212 | 0,143 | 0,169 | 0,183 | 0,159 | 0,188 |
| IC 95% | [0,57; 0,73] | [0,60; 0,71] | [0,62; 0,75] | [0,67; 0,81] | [0,62; 0,74] | [0,61; 0,75] |

**Friedman** (27 perguntas como blocos, 6 métodos): χ² = 7,21, p = 0,205.

Pares com diferença significativa após Holm (α = 0,05): nenhum.

Os 5 menores p brutos: Refine vs Map-Rerank: Δ = -0,086, p = 0,017, p_Holm = 0,261; Stuff vs Map-Rerank: Δ = -0,097, p = 0,026, p_Holm = 0,367; Map-Rerank vs Reciprocal: Δ = 0,068, p = 0,101, p_Holm = 1,000; Map-Rerank vs Query Step-Down: Δ = 0,061, p = 0,141, p_Holm = 1,000; Map-Reduce vs Map-Rerank: Δ = -0,061, p = 0,185, p_Holm = 1,000.

### 1.2 Custo por pergunta

| Método | Tokens (média ± DP) | × Stuff | Tempo total (s) |
|---|---|---|---|
| Stuff | 1.313 ± 254 | 1,0 | 3,40 ± 1,70 |
| Refine | 2.453 ± 437 | 1,9 | 10,32 ± 4,03 |
| Map-Reduce | 1.987 ± 315 | 1,5 | 7,30 ± 2,87 |
| Map-Rerank | 1.931 ± 253 | 1,5 | 5,83 ± 1,77 |
| Query Step-Down | 6.773 ± 1.222 | 5,2 | 12,44 ± 3,49 |
| Reciprocal | 6.592 ± 1.268 | 5,0 | 9,06 ± 1,70 |

### 1.3 Recusas (o modelo declarou não ter a resposta)

| Método | Revalida | Grendene | SBC | Total | Sim. média sem recusas |
|---|---|---|---|---|---|
| Stuff | 6 | 2 | 0 | 8 | 0,746 |
| Refine | 1 | 1 | 0 | 2 | 0,674 |
| Map-Reduce | 0 | 2 | 0 | 2 | 0,703 |
| Map-Rerank | 0 | 1 | 0 | 1 | 0,758 |
| Query Step-Down | 0 | 0 | 0 | 0 | 0,682 |
| Reciprocal | 4 | 0 | 0 | 4 | 0,708 |

Similaridade média das respostas-recusa: 0,436 (n = 17); das demais: 0,710.

## 2. RAGAS sobre o v3 (juiz gpt-4o-mini; 162 de 162 linhas avaliadas)

| Método | Answer Rel. | Faithfulness | Context Prec. | Context Rec. | Cosseno | n válidos (AR/F/CP/CR) |
|---|---|---|---|---|---|---|
| Stuff | 0,545 | 0,582 | 0,830 | 0,857 | 0,647 | 27/27/27/27 |
| Refine | 0,706 | 0,749 | 0,868 | 0,849 | 0,657 | 27/27/27/27 |
| Map-Reduce | 0,745 | 0,635 | 0,876 | 0,844 | 0,682 | 27/27/27/27 |
| Map-Rerank | 0,427 | 0,631 | 0,874 | 0,844 | 0,744 | 27/27/27/27 |
| Query Step-Down | 0,782 | n/a | n/a | n/a | 0,682 | 27/0/0/0 |
| Reciprocal | 0,285 | n/a | n/a | n/a | 0,676 | 27/0/0/0 |

Falhas de avaliação (None, não imputado): {'answer_relevancy': 0, 'faithfulness': 0, 'context_precision': 0, 'context_recall': 0}.

Faithfulness sem recusas: Stuff 0,827, Refine 0,808, Map-Reduce 0,666, Map-Rerank 0,656.

**Friedman – Faithfulness** (27 perguntas completas, 4 métodos): χ² = 2,65, p = 0,449. Pares significativos (Holm): nenhum.

**Friedman – Answer Relevancy** (27 perguntas completas, 6 métodos): χ² = 55,85, p = < 0,001. Pares significativos (Holm): Stuff vs Map-Reduce (Δ = -0,199, p_Holm = 0,032), Refine vs Map-Rerank (Δ = 0,279, p_Holm = 0,003), Refine vs Reciprocal (Δ = 0,420, p_Holm = 0,001), Map-Reduce vs Map-Rerank (Δ = 0,318, p_Holm = < 0,001), Map-Reduce vs Reciprocal (Δ = 0,459, p_Holm = < 0,001), Map-Rerank vs Query Step-Down (Δ = -0,355, p_Holm = < 0,001), Query Step-Down vs Reciprocal (Δ = 0,496, p_Holm = < 0,001).

**Correlação de Spearman, linha a linha, entre similaridade de cosseno e cada métrica RAGAS:**

- Answer Relevancy: ρ = 0,20 (p = 0,009, n = 162)
- Faithfulness: ρ = 0,35 (p = < 0,001, n = 108)
- Context Precision: ρ = 0,16 (p = 0,105, n = 108)
- Context Recall: ρ = 0,33 (p = < 0,001, n = 108)

**Ruído do juiz em entradas idênticas:** em 27 de 27 perguntas os 4 métodos com contexto receberam exatamente os mesmos fragmentos, então Context Precision e Context Recall deveriam coincidir entre eles.
- Context Precision: mesmo valor nos 4 métodos em 21 de 27 perguntas; amplitude média entre os 4 = 0,069; amplitude máxima = 1,000.
- Context Recall: mesmo valor nos 4 métodos em 24 de 27 perguntas; amplitude média entre os 4 = 0,032; amplitude máxima = 0,333.

**Recusas × Context Recall** (métodos com contexto): 13 recusas; em 10 delas o Context Recall era ≥ 0,5 (o juiz considerou que o contexto recuperado cobria ao menos metade do gabarito). Context Recall médio nas recusas: 0,744.

| Método | Recusas | Context Recall médio nelas |
|---|---|---|
| Stuff | 8 | 0,708 |
| Refine | 2 | 1,000 |
| Map-Reduce | 2 | 0,500 |
| Map-Rerank | 1 | 1,000 |

## 3. Validação do juiz: gpt-4o-mini × Command A (Cohere) e reteste do gpt-4o-mini

Amostra estratificada: 54 linhas avaliadas pelo Command A (3 por domínio × método, seed 42); 54 também já avaliadas pelo 4o-mini na rodada principal; 54 com reteste do 4o-mini.

- Answer Relevancy: média 4o-mini = 0,555, média Command A = 0,605 (n = 54)
- Faithfulness: média 4o-mini = 0,686, média Command A = 0,723 (n = 36)
- Context Precision: média 4o-mini = 0,931, média Command A = 0,750 (n = 36)
- Context Recall: média 4o-mini = 0,831, média Command A = 0,767 (n = 36)

| Métrica | n reteste | ρ reteste | Dif. abs. reteste | n inter | ρ inter | Dif. abs. inter | Viés (mini − A) | p viés | ρ inter sem recusas |
|---|---|---|---|---|---|---|---|---|---|
| Answer Relevancy | 54 | 0,99 | 0,012 | 54 | 0,93 | 0,067 | -0,050 | < 0,001 | 0,90 |
| Faithfulness | 36 | 0,76 | 0,085 | 36 | 0,57 | 0,179 | -0,038 | 0,616 | 0,59 |
| Context Precision | 36 | 0,78 | 0,029 | 36 | 0,03 | 0,232 | 0,181 | 0,004 | 0,11 |
| Context Recall | 36 | 0,84 | 0,033 | 36 | 0,87 | 0,065 | 0,065 | 0,068 | 0,91 |

**Ranking de métodos por Answer Relevancy na amostra** — 4o-mini: Refine > Query Step-Down > Map-Reduce > Stuff > Map-Rerank > Reciprocal; Command A: Refine > Query Step-Down > Map-Reduce > Stuff > Map-Rerank > Reciprocal; τ de Kendall = 1,00 (p = 0,003).

**Ranking de métodos por Faithfulness na amostra** — 4o-mini: Refine > Map-Reduce > Stuff > Map-Rerank; Command A: Refine > Stuff > Map-Rerank > Map-Reduce; τ de Kendall = 0,33 (p = 0,750).

**Faithfulness nas recusas da amostra** (n = 7): 4o-mini média 0,083, Command A média 0,333.

## 4. Compressão contextual (método Stuff, 27 perguntas)

### 4.1 Limiares fixos (0,3 / 0,5 / 0,7)

| Domínio | Limiar | Chunks mantidos | Tokens | Δ tokens | Similaridade | Δ similaridade |
|---|---|---|---|---|---|---|
| Revalida | sem filtro | 4,00 | 1.159 | -- | 0,592 | -- |
| Revalida | 0,3 | 4,00 | 1.158 | -0,1% | 0,592 | -0,1% |
| Revalida | 0,5 | 4,00 | 1.162 | 0,2% | 0,591 | -0,3% |
| Revalida | 0,7 | 1,08 | 458 | -60,5% | 0,434 | -26,7% |
| Grendene | sem filtro | 4,00 | 1.578 | -- | 0,602 | -- |
| Grendene | 0,3 | 4,00 | 1.578 | 0,0% | 0,602 | 0,0% |
| Grendene | 0,5 | 4,00 | 1.578 | 0,0% | 0,602 | 0,0% |
| Grendene | 0,7 | 1,00 | 425 | -73,1% | 0,582 | -3,3% |
| SBC | sem filtro | 4,00 | 1.326 | -- | 0,829 | -- |
| SBC | 0,3 | 4,00 | 1.326 | 0,0% | 0,825 | -0,6% |
| SBC | 0,5 | 4,00 | 1.326 | 0,0% | 0,829 | 0,0% |
| SBC | 0,7 | 1,12 | 477 | -64,0% | 0,722 | -12,9% |

Geral, pareado por pergunta contra sem filtro (Wilcoxon): 0,3: n = 27, Δ tokens = -0,0%, Δ similaridade = -0,2%, p = 0,225; 0,5: n = 27, Δ tokens = 0,1%, Δ similaridade = -0,1%, p = 0,593; 0,7: n = 27, Δ tokens = -65,4%, Δ similaridade = -16,1%, p = 0,004.

### 4.2 Limiares calibrados por domínio (quantis 25/50/75 da similaridade do top-4)

| Domínio | Limiar | Chunks mantidos | Tokens | Δ tokens | Similaridade | Δ similaridade |
|---|---|---|---|---|---|---|
| Revalida | sem filtro | 4,00 | 1.158 | -- | 0,591 | -- |
| Revalida | P25 (0,592) | 3,17 | 948 | -18,2% | 0,570 | -3,6% |
| Revalida | P50 (0,626) | 2,33 | 781 | -32,6% | 0,505 | -14,6% |
| Revalida | P75 (0,669) | 1,75 | 631 | -45,5% | 0,471 | -20,4% |
| Grendene | sem filtro | 4,00 | 1.578 | -- | 0,602 | -- |
| Grendene | P25 (0,625) | 3,00 | 1.230 | -22,0% | 0,602 | 0,0% |
| Grendene | P50 (0,639) | 2,14 | 870 | -44,9% | 0,602 | 0,0% |
| Grendene | P75 (0,661) | 1,57 | 662 | -58,0% | 0,599 | -0,5% |
| SBC | sem filtro | 4,00 | 1.326 | -- | 0,829 | -- |
| SBC | P25 (0,579) | 3,12 | 1.065 | -19,6% | 0,827 | -0,2% |
| SBC | P50 (0,629) | 2,12 | 776 | -41,5% | 0,830 | 0,1% |
| SBC | P75 (0,671) | 1,50 | 583 | -56,0% | 0,806 | -2,9% |

Geral, pareado por pergunta contra sem filtro (Wilcoxon): P25: n = 27, Δ tokens = -19,8%, Δ similaridade = -1,5%, p = 0,866; P50: n = 27, Δ tokens = -39,0%, Δ similaridade = -5,7%, p = 0,139; P75: n = 27, Δ tokens = -52,5%, Δ similaridade = -9,3%, p = 0,093.

Limiares calibrados: Revalida: p25 = 0,592, p50 = 0,626, p75 = 0,669 (sim. do top-4 entre 0,539 e 0,718); Grendene: p25 = 0,625, p50 = 0,639, p75 = 0,661 (sim. do top-4 entre 0,539 e 0,729); SBC: p25 = 0,579, p50 = 0,629, p75 = 0,671 (sim. do top-4 entre 0,525 e 0,750).

## 5. Bancos vetoriais

| Domínio (chunks) | Index. Chroma (ms) | Index. FAISS (ms) | Consulta Chroma (ms) | Consulta FAISS (ms) | Mem. Chroma (MB) | Mem. FAISS (MB) | Concord. top-k |
|---|---|---|---|---|---|---|---|
| Revalida (430) | 424,8 | 0,8 | 3,059 | 0,049 | 38,68 | 2,05 | 100% |
| Grendene (199) | 170,8 | 0,1 | 5,924 | 0,022 | 2,77 | 0,02 | 100% |
| SBC (42) | 95,1 | 0,1 | 2,818 | 0,016 | 3,25 | 0,00 | 100% |

