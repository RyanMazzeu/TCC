# Plano de pesquisa — RAG com Docugami, MedQA e FaQuAD

## Situação (30/10/2026)

- **Feito:** passos 1 a 4 da seção 7 (código, datasets, embeddings no Colab, índices, piloto, validação do juiz: AUROC 0,701 contra 0,567 do RAGAS original).
- **Rodando sozinho** (`scripts/continuar_openai.sh`, log em `scripts/continuar_openai.log`): tudo o que depende só da chave da OpenAI — camada principal do gpt-4o-mini e do GPT-6 Luna, generalização do GPT-6 Sol, repetições do Stuff, RAGAS, comparativo ChromaDB × FAISS e a análise (`resultados/analise_benchmarks.md`). Se for interrompido, rodar o mesmo script de novo retoma de onde parou.
- **Falta, de vocês:** a chave `OPENROUTER_API_KEY` no `.env` (Qwen, DeepSeek, Claude Haiku e Claude Sonnet) e combinar com o professor a troca do Opus 4.8 e do GPT-6 Astra pelos intermediários.
- **Depois da chave:** rodar os 4 modelos do OpenRouter (passos 5 a 8 só com eles), refazer a análise e reescrever o artigo.
- **Limitação a declarar:** os modelos rodaram em paralelo no mesmo notebook, então os tempos de resposta medidos incluem disputa de CPU (o embedding da pergunta e da similaridade é local). Tokens e custo não são afetados; o tempo de indexação e consulta dos bancos vetoriais é medido depois, sem nada rodando junto.

## Contexto

O professor apontou três problemas na pesquisa atual:
1. Só um LLM foi usado (gpt-4o-mini).
2. A escolha dos modelos não foi justificada.
3. O gabarito foi feito pelos próprios autores, em vez de vir de datasets prontos.

A decisão agora é refazer os testes com três datasets prontos — **Docugami SEC 10-Q, MedQA e FaQuAD** — e vários LLMs.

**Opinião sobre a escolha:** é uma boa combinação.
- Docugami e MedQA são dois dos três datasets do artigo-base [Şakar & Emekci, 2025], então dá para comparar direto com a Tabela 1 dele.
- O FaQuAD traz o português, com perguntas e respostas escritas por pessoas.
- O MedQA tem uma vantagem extra: são questões reais de prova (USMLE), com a alternativa certa oficial. Isso dá a métrica mais objetiva da pesquisa: acurácia.
- O Llama2 sai, porque o gabarito dele é todo do GPT-4. O Pirá também sai, porque os documentos são traduzidos.


**Pergunta de pesquisa:** como os 6 métodos de RAG se comparam em qualidade da resposta, qualidade da recuperação e custo (tokens, tempo, US$), e essa comparação muda conforme o LLM e o domínio (financeiro, médico, institucional em PT-BR)?

## Escopo

| Tema do plano aprovado no TCC01 | Nesta pesquisa |
|---|---|
| Métodos de RAG | **Dentro**: é o eixo central |
| LLMs | **Dentro**: 7 modelos (a crítica do professor) |
| Bancos vetoriais | **Dentro, refeito com os vetores dos novos datasets** (seção "Bancos vetoriais" abaixo) |
| Embeddings | **Fora**: fixo no BGE-M3, que é multilíngue e serve aos 3 datasets |
| Compressão contextual | **Fora**: os resultados foram feitos nos datasets descartados e saem do artigo |
| Hardware (CPU/RAM) | **Fora**: com os LLMs via API, a máquina local não mede o custo do método |

## Bancos vetoriais

| Uso | Banco | Índice | Busca |
|---|---|---|---|
| Todos os experimentos de RAG (6 métodos, 7 LLMs) | ChromaDB 1.5.9, persistente em disco | HNSW (padrão do Chroma) | **Aproximada** |
| Comparativo de infraestrutura | ChromaDB × FAISS 1.14.3 | HNSW × `IndexFlatL2` | Aproximada × **exata** |
| Pinecone | Fora (serviço pago) | | |

**Comparativo ChromaDB × FAISS refeito com os vetores dos novos datasets.** O comparativo antigo foi feito com os corpora descartados, de ~670 fragmentos no total, em que a diferença entre os bancos quase não aparece. Agora ele usa os mesmos vetores BGE-M3 dos experimentos: FaQuAD (318 fragmentos), Docugami (4.463) e MedQA (~121 mil). Isso dá três tamanhos de base, de centenas a centenas de milhares de vetores.
- **Mede:** tempo de indexação, tempo médio de consulta, memória, e a **concordância do top-4** entre o HNSW do Chroma (aproximado) e o FAISS exato. Essa concordância mostra quanto a busca aproximada erra em relação à exata em cada tamanho de base.
- **Recall@k com o FAISS** no Docugami e no FaQuAD, ao lado do Recall@k do Chroma. É o que responde, sem LLM, se a busca aproximada perde o trecho certo e, portanto, se os resultados de RAG valeriam também com o FAISS.
- **Se a concordância do top-4 no MedQA ficar abaixo de 90%:** rodar o Stuff com o FAISS no MedQA com o gpt-4o-mini (25 perguntas, centavos), pra medir o efeito na acurácia.
- **Custo:** zero, fora o caso acima. Roda localmente, sem API, reaproveitando os embeddings já calculados. As consultas usam as perguntas da amostra de cada dataset.
- **Por que não testar todo o RAG nos dois bancos:** o banco só devolve os trechos. Se os dois devolvem os mesmos trechos, o LLM recebe a mesma entrada e a resposta não muda. A concordância do top-4 e o Recall@k mostram isso diretamente.
- **Correção no artigo:** o texto atual diz que "a busca é exata nos dois casos". Isso está errado para o ChromaDB, que usa HNSW por padrão. A metodologia precisa dizer que os experimentos de RAG usam busca aproximada, e o comparativo mede o efeito disso.

Os resultados com Revalida, Grendene, SBC e o gabarito do CFM também saem do artigo. A seção "Diferenças em relação ao plano de trabalho" do artigo precisa listar esses cortes e o motivo.


## 1. Datasets

| Dataset | Texto base (vai para o RAG) | Perguntas e gabarito | Trecho certo marcado? | Idioma |
|---|---|---|---|---|
| Docugami SEC 10-Q (LlamaIndex) | 20 relatórios 10-Q, PDF → texto | 195 perguntas humanas; respostas do GPT-4 com revisão humana | Sim (qual relatório) | EN |
| MedQA-USMLE, teste (Jin et al., 2021) | 18 livros-texto do MedQA (`MedRAG/textbooks`, ~126 mil trechos) | 1.273 questões de 4 alternativas com a alternativa certa oficial | Não | EN |
| FaQuAD (Sayama et al., BRACIS 2019) | 249 parágrafos de 39 documentos | 900 perguntas e respostas humanas | Sim (o parágrafo) | PT-BR |

**Amostra: desenho misto em duas camadas.**

| Camada | Modelos | Perguntas | Para que serve |
|---|---|---|---|
| Principal | Os 5 baratos: Qwen3.5-9B, DeepSeek V4 Pro, Claude Haiku 5.5, GPT-6 Luna, gpt-4o-mini | 100 por dataset (300) | Comparação entre os métodos com força estatística |
| Generalização | Os 7 modelos (os 5 baratos + Claude Sonnet 5 e GPT-6 Sol) | 25 por dataset (75), um subconjunto das 100 | Verificar se a ordem dos métodos se mantém nos modelos mais fortes |

- **Por que assim:** quase todo o custo está nos 2 intermediários (~US$ 7 de ~US$ 8,5 nas 75 perguntas). Os baratos permitem 4 vezes mais perguntas por ~US$ 4 a mais. Com 25 perguntas, o IC de 95% da similaridade fica em ±0,04, maior que boa parte das diferenças entre métodos já observadas (0,02–0,08). Com 100, cai para ~±0,02.
- **Sorteio:** com semente fixa (`SEED_AMOSTRA=42`). As 25 perguntas da camada de generalização são as 25 primeiras das 100 sorteadas, então todos os 7 modelos compartilham essas 75 perguntas e a comparação entre eles é pareada.
- **Limitação a declarar:** os intermediários têm só 25 perguntas por dataset, então neles só as diferenças grandes entre métodos aparecem. As conclusões sobre qual método é melhor vêm da camada principal.
- A recuperação (Recall@k) não custa nada e usa todas as perguntas.

**Fragmentação:** igual para todos e igual ao artigo-base: 1.000 caracteres com 100 de sobreposição.
- No MedQA, os livros são remontados por título e refragmentados assim.

## 2. Métodos de RAG (sem mudança)

Stuff, Refine, Map-Reduce, Map-Rerank, Query Step-Down e Reciprocal RAG. Já estão em `scripts/pipeline_base.py`, com k=4.

**Grupo de controle: sem RAG.** Cada LLM também responde a pergunta sem nenhum trecho recuperado (1 chamada). Sem isso não dá para afirmar que o RAG ajudou, principalmente no MedQA, em que o modelo pode acertar de memória por ser uma prova conhecida. É o mesmo controle do MIRAGE [Xiong et al., 2024].

## 3. LLMs e justificativa

Todos os modelos rodam nos 6 métodos e no controle sem RAG. As faixas são de **preço**, não de tamanho: o DeepSeek V4 Pro custa como um modelo econômico, mas é um dos maiores modelos abertos.

| Modelo | Empresa | Pesos | Tamanho | US$/1M (entrada/saída) | Faixa de preço | Camada |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | Alibaba | Abertos (Apache-2.0) | 9,7 bi | 0,17 / 0,25 | Econômica | Principal + generalização |
| DeepSeek V4 Pro | DeepSeek | Abertos (MIT) | 1,6 tri (MoE) | 0,21 / 0,42 | Econômica | Principal + generalização |
| Claude Haiku 5.5 | Anthropic | Fechados | não divulgado | 0,10 / 0,50 | Econômica | Principal + generalização |
| GPT-6 Luna | OpenAI | Fechados | não divulgado | 0,10 / 0,50 | Econômica | Principal + generalização |
| gpt-4o-mini (ponte com o artigo-base) | OpenAI | Fechados | não divulgado | 0,15 / 0,60 | Econômica | Principal + generalização |
| Claude Sonnet 5 | Anthropic | Fechados | não divulgado | 2 / 10 | Intermediária | Só generalização |
| GPT-6 Sol | OpenAI | Fechados | não divulgado | 2 / 10 | Intermediária | Só generalização |

São 7 modelos, de 4 empresas, 2 deles de pesos abertos, com tamanhos de 9,7 bilhões a 1,6 trilhão de parâmetros, lançados de 2024 a 2026. Nas duas famílias fechadas há um modelo de cada faixa de preço, o que permite analisar o efeito do porte.

**Por que intermediários, e não os modelos de topo (Claude Opus 4.8 e GPT-6 Astra):**
- **Custo:** os de topo custam de 2,5 a 5 vezes mais por token. Nos 6 métodos, o Astra sozinho custaria ~US$ 21 e o Opus ~US$ 9 nas 75 perguntas, contra ~US$ 3,6 de cada intermediário.
- **Desempenho próximo:** o Sonnet 5 fica perto do Opus 4.8 no GPQA (88,9% contra 92,4%) e o supera em contexto longo, o mais próximo de RAG (82,0% contra 77,7% no AA-LCR). O GPT-6 Sol fica a ~5 pontos do Astra no placar geral do BenchLM (78,7 contra 84,2).
- **Condições iguais:** o Astra não permite desligar o raciocínio. Os intermediários permitem, então rodam como os outros modelos.
- **Atenção:** o professor pediu "Claude 4.8" e "GPT parrudo" pelo nome. A troca precisa ser combinada com ele, com esse argumento.

**Por que esses intermediários:**
- **Sonnet 5, e não o Sonnet 5.5:** o 5.5 não permite desligar o raciocínio.
- **GPT-6 Sol, e não o GPT-6.1 Sol:** o 6.1 também não permite.
- **gpt-4o-mini:** um dos 3 LLMs do artigo-base, para comparar direto com os números dele.

**Configuração:** raciocínio desligado onde é possível e temperatura 0 onde a API aceita. Claude e GPT-6 não aceitam temperatura (a API devolve erro).

**Repetições:** como 4 dos 7 modelos (Haiku, Sonnet 5, Luna e Sol) não aceitam temperatura 0, as respostas variam entre execuções. O Stuff é rodado 3 vezes com esses modelos, para medir quanto da diferença entre métodos é só variação do próprio modelo.

**Justificativa para o texto:**

- **Variar o LLM:** Şakar & Emekci (2025), RAGBench (Friel et al., 2024) e MIRAGE (Xiong et al., 2024) mostram que o LLM muda o resultado.
- **Deixar claro o motivo da escolha dos modelos da pesquisa**: Lembrando que a ideia de selecionar os modelos diferentes foi para generalização dos resultados da pesquisa, escolhendo modelos diversos, um gpt-4o (por conta da pesquisa base), um gpt parrudo, um opensource, um deepseek, um claude, etc. Logo, tem que ficar claro o motivo da escolha de cada modelo.
- **Usar modelos econômicos:** FrugalGPT (Chen, Zaharia & Zou, 2023).
- **Incluir um modelo aberto:** reprodutibilidade (Palmer, Smith & Spirling, 2024; Chen, Zaharia & Zou, 2024, sobre a mudança de comportamento do GPT-4).

## 4. Métricas

| Métrica | Docugami | MedQA | FaQuAD | Juiz LLM? |
|---|---|---|---|---|
| Similaridade de cosseno com o gabarito (métrica do artigo-base) | ✓ | ✓ (contra o texto da alternativa certa) | ✓ | Não |
| Acurácia (letra certa) | — | ✓ | — | Não |
| F1 e Exact Match (métrica oficial SQuAD/FaQuAD) | — | — | ✓ | Não |
| Hit@4, Recall@4, MRR (recuperação) | ✓ | — | ✓ | Não |
| Taxa de recusa | ✓ | ✓ | ✓ | Não |
| Tokens, tempo, custo em US$ | ✓ | ✓ | ✓ | Não |
| RAGAS Faithfulness | ✓ | ✓ | ✓ | Sim |
| RAGAS Answer Relevancy | ✓ | ✓ (com ressalva) | ✓ | Sim |
| RAGAS Context Precision e Context Recall | ✓ | ✓ | ✓ | Sim |

**Métricas do RAGAS [Es et al., 2024]:** juiz gpt-4o-mini, embedding `text-embedding-3-small`, com o script [avaliar_ragas.py](TCC/Pesquisa/scripts/avaliar_ragas.py) já existente. Entram como métricas complementares: as conclusões principais vêm das métricas comparadas direto com o gabarito.

**Limitação do juiz:** o gpt-4o-mini também é um dos 7 modelos testados, então pode dar notas melhores às próprias respostas (viés de autopreferência; Panickssery et al., 2024). Isso vai declarado, e as notas do RAGAS do gpt-4o-mini não entram em conclusões sobre qual LLM é melhor. A validação do juiz no RAGBench (AUROC 0,687) foi apagada pelo `git reset` e precisa ser refeita (~US$ 0,40).

| Métrica | O que mede | Compara os métodos? | Como é calculada |
|---|---|---|---|
| Faithfulness | Se a resposta inventa algo que não está nos trechos (alucinação) | Sim | Por resposta. Só nos 4 métodos com conjunto único de trechos (Stuff, Refine, Map-Reduce, Map-Rerank). Juiz validado no RAGBench: AUROC 0,687 |
| Answer Relevancy | Se a resposta foca no que foi perguntado | Sim | Por resposta, nos 6 métodos. Ressalvas: não mede acerto; no MedQA informa pouco (a resposta é letra + justificativa); não validada no RAGBench |
| Context Precision e Context Recall | Se os trechos trazidos servem para responder (avaliam a busca) | Não | **Uma vez por pergunta**, não por método e LLM: os 4 métodos com conjunto único recebem os mesmos trechos (conferido nos logs antigos: 27/27 perguntas). Diferenças entre métodos nessas métricas são ruído do juiz |

**Usos da Context Precision/Recall:**
- **MedQA:** é a única avaliação da busca, porque o dataset não marca o trecho certo e não dá Recall@k.
- **Docugami e FaQuAD:** validar o juiz, comparando com o Recall@k calculado pelo gabarito (correlação de Spearman). Se as duas concordarem, isso sustenta o uso no MedQA.

**Correção no texto antigo:** se o artigo atual interpreta diferenças de Context Precision/Recall entre métodos, isso precisa sair, porque era ruído do juiz.

**Comparação com o artigo-base:** além do BGE-M3, recalcular a similaridade com o `bge-small-en-v1.5`, o embedding do artigo, nos datasets em inglês. É local e não custa nada.

## 5. Análise estatística (reaproveita `scripts/analisar_resultados.py`)

- **Friedman entre os métodos:** dentro de cada dataset × LLM, com Wilcoxon par a par e correção de Holm (`friedman_com_posthoc`).
- **Intervalos de confiança:** 95% por bootstrap (`ic_bootstrap`).
- **Comparação com o artigo-base:** medianas por método contra a Tabela 1 dele, no Docugami e no MedQA.
- **Qual método é melhor:** na camada principal (5 modelos, 300 perguntas).
- **Generalização:** na camada de 75 perguntas, concordância da ordem dos métodos entre os 7 modelos (coeficiente W de Kendall).
- **Efeito do RAG:** cada método contra o controle sem RAG, por LLM e dataset.
- **Efeito do porte:** nas famílias OpenAI (Luna × Sol) e Anthropic (Haiku × Sonnet 5), o melhor método muda? Um modelo econômico com o melhor método alcança o intermediário com o Stuff [FrugalGPT, Chen, Zaharia & Zou, 2023]?
- **Variação do modelo:** desvio entre as 3 repetições do Stuff, comparado com as diferenças entre métodos.

## 6. Implementação

1. **`scripts/preparar_benchmarks.py`:** adicionar `preparar_medqa()`.
   - Baixar o teste do MedQA (4 opções) e os livros do `MedRAG/textbooks`.
   - Gerar `dados/processados/medqa.json`: pergunta = enunciado + alternativas A–D; `resposta_esperada` = texto da alternativa certa; `letra_correta`.
   - Gerar o corpus por livro.
2. **`scripts/benchmarks.py`:**
   - `DATASETS = ["docugami_10q", "medqa", "faquad"]`
   - `IDIOMA["medqa"] = "en"`
3. **`scripts/pipeline_base.py`:** sem novo método de RAG.
   - **Prompts no idioma de cada dataset:** inglês para Docugami e MedQA, português para o FaQuAD. A versão em inglês é a tradução exata da portuguesa. Motivo: com prompt em português, o GPT-6 Luna e o gpt-4o-mini responderam em português ~80% das perguntas em inglês, e a similaridade caía por causa do idioma (essa correção também foi apagada pelo `git reset`).
   - **Prompts do MedQA:** os prompts atuais mandam dizer "não sei" quando a resposta não está nos trechos. Os livros raramente trazem a resposta exata de uma questão de prova, então a acurácia mediria recusa, não conhecimento. No MedQA, o modelo usa os trechos como apoio e **sempre escolhe uma alternativa**, como no MIRAGE [Xiong et al., 2024]. O controle sem RAG usa o mesmo prompt, sem os trechos.
   - **Formato da resposta no MedQA:** a letra da alternativa, seguida de uma justificativa curta, igual nos 6 métodos.
   - **Limitação a declarar:** no Step-Down e no Reciprocal, as perguntas alternativas são geradas a partir do caso clínico com as alternativas junto, o que é diferente dos outros datasets.
4. **Novo `scripts/metricas_resposta.py`:**
   - F1/EM no formato SQuAD (normalização PT: minúsculas, sem pontuação e sem artigos), contra todas as `respostas_aceitas`.
   - Extração da letra (A–D) e acurácia no MedQA.
5. **`scripts/avaliar_recuperacao.py`:** pular o MedQA, que não tem trecho certo.
6. **`scripts/analisar_benchmarks.py`:**
   - Novas colunas: acurácia, F1/EM e custo em US$ (tabela de preços por modelo).
   - Referências do artigo-base para o MedQA na Tabela 1.
7. **`scripts/avaliar_ragas.py`:**
   - Rodar Faithfulness e Answer Relevancy por resposta, sobre os novos logs.
   - Calcular Context Precision e Context Recall uma vez por pergunta (sobre os trechos do Stuff) e replicar o valor para os outros métodos na análise, em vez de pagar o juiz várias vezes pelos mesmos trechos.
   - No MedQA, a referência das métricas é o texto da alternativa certa.
8. **Embeddings e índices:** `scripts/construir_indices.py` (pronto). Os embeddings são calculados uma vez, guardados em blocos em `vectorstores_benchmarks/_cache_embeddings/` e reaproveitados em todos os testes. Em CPU, o Docugami leva ~1,7 h e o MedQA ~27 h (medido), então os dois são calculados no Colab com GPU: `scripts/exportar_fragmentos.py` exporta os fragmentos, `colab/colab_embeddings.py` calcula os vetores no Colab, e um manifesto (número de fragmentos + hash dos textos) garante que os vetores correspondem exatamente aos fragmentos daqui. Testado de ponta a ponta numa amostra.
9. **Comparativo de bancos vetoriais:** adaptar `scripts/_gerar_vetores_para_benchmark.py` (passo 1: Chroma) e `scripts/_medir_faiss_puro.py` (passo 2: FAISS, em processo separado) para ler os vetores do cache de embeddings dos 3 datasets, em vez de reencodar os corpora antigos.

## 7. Execução e custo estimado

Estimativas a partir do consumo medido nas rodadas anteriores (~17 mil tokens de entrada e ~1,4 mil de saída por pergunta nos 6 métodos), com os preços atuais do OpenRouter e da OpenAI.

| Etapa | Custo |
|---|---|
| 5 modelos baratos × 6 métodos × 300 perguntas | ~US$ 5 |
| Claude Sonnet 5 e GPT-6 Sol × 6 métodos × 75 perguntas | ~US$ 7,2 |
| **Geração** | **~US$ 12** |
| Controle sem RAG (mesmas perguntas de cada camada) | ~US$ 1 |
| Repetições do Stuff (2 a mais, nos 4 modelos sem temperatura, nas 75 perguntas) | ~US$ 1 |
| RAGAS (Faithfulness + Answer Relevancy por resposta; Context Precision/Recall 1 vez por pergunta) | ~US$ 5–6 |
| **Total** | **~US$ 19–20** |

Já gasto até agora com a OpenAI: ~US$ 4,6. Se o orçamento apertar, cortar perguntas dos intermediários dói menos do que cortar modelos: tirar modelos reabre a crítica do professor.

**Ordem:**
1. ~~Recuperar o código apagado pelo `git reset --hard` de 25/10 e baixar os datasets de novo.~~ Feito.
2. **Embeddings no Google Colab (vocês, ~20–30 min no total):**
   1. Entrar em colab.research.google.com > Novo notebook.
   2. Menu Ambiente de execução > Alterar o tipo de ambiente de execução > **GPU T4** > Salvar.
   3. Painel Arquivos (ícone de pasta, à esquerda) > Fazer upload dos 3 arquivos da pasta `Pesquisa/colab/`: `colab_embeddings.py`, `fragmentos_docugami_10q.jsonl.gz` e `fragmentos_medqa.jsonl.gz`.
   4. Numa célula, rodar `!pip install -q FlagEmbedding==1.4.0`; noutra, `!python colab_embeddings.py`.
   5. Quando aparecer "Baixe o arquivo embeddings_colab.zip", baixar esse zip pelo painel Arquivos (três pontinhos > Fazer download). São ~250 MB.
   6. Descompactar o zip dentro de `Pesquisa/vectorstores_benchmarks/_cache_embeddings/` (devem aparecer as pastas `bench_docugami_10q_bge-m3_c1000` e `bench_medqa_bge-m3_c1000`).
   7. Rodar `python construir_indices.py` em `Pesquisa/scripts` (ou pedir "siga o plano", que eu rodo). Ele confere o manifesto e cria os índices em poucos minutos.
3. Piloto de 2 perguntas por dataset com o gpt-4o-mini (~US$ 0,01).
4. Validação do juiz no RAGBench (`validar_juiz_ragbench.py`, ~US$ 0,40).
5. Camada principal: os 5 modelos baratos nas 300 perguntas (`rodar_benchmarks.py --camada principal`).
6. Camada de generalização: Sonnet 5 e Sol nas 75 perguntas (`--camada generalizacao`), depois de conferir os resultados dos baratos.
7. Repetições do Stuff (`--repeticoes`).
8. RAGAS (`avaliar_ragas_benchmarks.py respostas` e `contexto`).
9. Comparativo ChromaDB × FAISS com os vetores novos, com o Recall@k dos dois bancos (sem custo).
10. Análise (`analisar_benchmarks.py`) e reescrita do artigo (seção 9).

## 8. Cronograma

Estimativa de duração de cada etapa:

| Etapa | Duração estimada |
|---|---|
| Recuperar o código e preparar os 3 datasets (feito); embeddings no Colab | ~30 min |
| Piloto e ajustes | 1 dia |
| Camada principal (5 modelos, 300 perguntas) | 2–3 dias de execução |
| Camada de generalização, controle sem RAG e repetições | 1–2 dias |
| RAGAS e análise estatística | 2 dias |
| Reescrita da metodologia, dos resultados e da discussão | 1 semana |
| Revisão com o orientador e apresentação | conforme o prazo |

Total até o rascunho completo: ~3 semanas.

## 9. Entregáveis

- Logs JSONL por dataset × LLM em `resultados/metricas/`.
- `resultados/analise_benchmarks.md`, com as tabelas e figuras.
- Reescrita de `Artigo_TCC02/sections/03-metodologia.tex`: datasets, LLMs com justificativa e métricas.
- Reescrita de `04-resultados_e_discussao.tex`, sem os resultados de compressão e dos datasets antigos.
- Atualização da seção "Diferenças em relação ao plano de trabalho", com os cortes de escopo.
- Comparativo ChromaDB × FAISS refeito com os 3 tamanhos de base, e a correção da frase sobre "busca exata".
- Novas referências no `referencias.bib`: Jin 2021, Sayama 2019, Docugami/LlamaIndex, FrugalGPT, Friel 2024, Xiong 2024, Palmer 2024, Panickssery 2024 (já está no .bib).

## Verificação

- **Preparação:** `python preparar_benchmarks.py` gera o `medqa.json` com 1.273 perguntas e o `faquad.json` com 900.
- **Teste das letras:** testar `metricas_resposta.py` com casos conhecidos, por exemplo uma resposta "B" com gabarito "B" deve dar acurácia 1, e conferir o F1 de exemplos do SQuAD.
- **Recuperação:** `avaliar_recuperacao.py` dá Recall@4 para Docugami e FaQuAD.
- **Bancos vetoriais:** `comparativo_vectorstores.json` com os 3 datasets e a concordância do top-4 entre o Chroma (HNSW) e o FAISS exato.
- **Piloto:** `rodar_benchmarks.py --llms gpt_mini --perguntas 2` sem erro, respostas em inglês no Docugami e no MedQA, letra extraída em todas as respostas do MedQA e nenhuma recusa nele.
- **RAGAS:** as 4 métricas preenchidas sem falhas; Context Precision/Recall com um valor por pergunta; correlação com o Recall@k reportada para Docugami e FaQuAD.
- **Controle sem RAG e repetições:** presentes no log para todos os modelos de cada camada.
- **Análise:** `analisar_benchmarks.py` gera as tabelas das duas camadas (300 perguntas nos 5 baratos; 75 perguntas, as mesmas, nos 7 modelos).
