# Relatório do piloto experimental — TCC RAG (Ryan & Afonso)

Data: 2026-09-06

## 1. Mudança de direção e por quê

Levantamento de originalidade mostrou que o desenho original do plano (6 métodos de RAG × 3
bancos vetoriais × 3 embeddings × datasets SEC 10Q / Llama2 ArXiv / MedQA) reproduz quase
integralmente **Şakar & Emekci (2025), "Maximizing RAG efficiency: A comparative analysis of RAG
methods"** (*Natural Language Processing*, Cambridge, v.31 — já citado como artigo-âncora na nossa
revisão bibliográfica). É um paper real, publicado, já citado por terceiros — não um achado obscuro.

Não encontramos réplica publicada desse paper nem TCC brasileiro com o mesmo desenho. Decisão
tomada: manter a estrutura comparativa do paper-âncora, mas trocar os 3 domínios/datasets originais
(em inglês) por **equivalentes brasileiros em português**, mesmo tipo de domínio:

| Domínio (paper-âncora) | Equivalente PT-BR proposto | Status |
|---|---|---|
| Médico — MedQA (USMLE) | **Revalida (INEP)** | ✅ piloto rodado (este relatório) |
| Financeiro — SEC 10-Q | **Grendene S.A., ITR 30/09/2025 (CVM/B3)** | ✅ piloto rodado (este relatório) |
| Técnico-científico — Llama2 Paper | **Medeiros & Oliveira (2025), paper IFES/SBC sobre RAG em português** | ✅ piloto rodado (este relatório) |

Os 3 domínios do desenho original agora têm um equivalente brasileiro em português com resultado
experimental real.

Isso precisa ser validado com o orientador antes de considerarmos fechado — é mudança na redação
do problema de pesquisa, não só decisão técnica.

## 2. O que foi de fato executado

### 2.1 Validação de engenharia (não é resultado científico)

Antes de qualquer dataset real, o pipeline (`scripts/pipeline_base.py`) foi validado ponta a ponta
(chunking → embedding → ChromaDB → recuperação → geração via OpenAI `gpt-4o-mini` → similaridade
de cosseno → log em JSONL) com um texto de exemplo trivial. Serviu só para confirmar que a
integração com a API paga funciona.

### 2.2 Piloto real: Revalida INEP 2024/2 — Questão 3

- **Pergunta:** prova discursiva oficial do INEP — [2024_2_PV_discursiva_regular.pdf](https://download.inep.gov.br/revalida/provas_e_gabaritos/2024_2_PV_discursiva_regular.pdf), Questão 3 (caso de adolescente de 16 anos buscando teste de gravidez e contraceptivo).
- **Gabarito:** padrão de resposta oficial definitivo — [2024_2_PEP_discursiva_versao_final.pdf](https://download.inep.gov.br/revalida/provas_e_gabaritos/2024_2_PEP_discursiva_versao_final.pdf).
- **Corpus de recuperação:** os dois documentos citados na própria referência bibliográfica do
  gabarito (não foi escolha nossa, e sim da banca do Revalida) — Código de Ética Médica (CFM,
  Resolução 2.217/2018) e o guia da Sociedade Brasileira de Pediatria "Consulta do adolescente:
  abordagem clínica, orientações éticas e legais". 109 chunks gerados dos 2 documentos.
- **Configuração:** método **Stuff**, **ChromaDB** (persistente local), embedding **BGE-M3**
  (multilíngue — BGE-small do paper-âncora só cobre inglês), geração via **OpenAI gpt-4o-mini**,
  chunk_size=1000/overlap=100 (mesmo parâmetro do paper-âncora, mantém comparabilidade futura).

**Resultados (n=3, os três sub-itens a/b/c da questão):**

| Sub-item | Similaridade cosseno | Tokens totais | T. recuperação (s) | T. geração (s) |
|---|---|---|---|---|
| a | 0,7505 | 1259 | 1,48 | 3,66 |
| b | 0,8830 | 1173 | 1,32 | 1,69 |
| c | 0,7575 | 1208 | 1,54 | 1,56 |
| **Média** | **0,797** | **1213** | **1,45** | **2,30** |

Dados brutos completos (contexto recuperado, resposta gerada, CPU/RAM): `resultados/metricas/piloto_revalida_stuff.jsonl`.

### 2.3 Leitura qualitativa

Nos três sub-itens, a **conclusão substantiva** da resposta gerada bateu com o gabarito oficial:

- Item a: modelo respondeu corretamente que não precisa de responsável e não há quebra de sigilo.
- Item c: modelo respondeu corretamente que pode participar das reuniões, sem autorização.

As similaridades mais baixas (a e c, ~0,75) parecem penalizar **verbosidade/fraseado**, não erro de
conteúdo — o padrão de resposta oficial é propositalmente telegráfico (pontuado por critério de
correção da banca), enquanto o modelo gera explicação mais longa. Isso é a limitação de "avaliação
semântica limitada" que o próprio `plano_de_pesquisa.md` (seção 15) já antecipava — vale registrar
como achado, não como bug.

### 2.4 Piloto real: Grendene S.A. — ITR 30/09/2025 (paralelo ao SEC 10-Q)

- **Documento:** [ITR consolidado, 30/09/2025](https://s3.sa-east-1.amazonaws.com/static.grendene.aatb.com.br/IFRS_ITR/2458_GRENDENESET25.ITR.pdf) — relatório trimestral oficial da Grendene S.A. (B3, Novo Mercado), com balanço, DFC e notas explicativas (72 páginas, texto narrativo real, não só tabela).
- **Diferença importante em relação ao Revalida:** este documento não vem com perguntas/gabarito prontos. As 3 perguntas (`dados/processados/grendene_itr_3t2025.json`) foram escritas por nós, cada uma ancorada em um trecho literal do documento — não inventamos nenhum número.
- **Configuração:** mesma do piloto do Revalida (Stuff, ChromaDB, BGE-M3, gpt-4o-mini, chunk 1000/100). 199 chunks gerados de um único documento.

| Pergunta | Similaridade | Conteúdo correto? |
|---|---|---|
| q1 — taxa final do USD em 30/09/2025 | 0,4386 | ✅ sim (5,3186, bate exatamente) |
| q2 — regra de consolidação de controladas | 0,9415 | ✅ sim |
| q3 — saldo de "caixa e bancos" consolidado | 0,4755 | ⚠️ ver nota abaixo |

**Achado 1 — similaridade baixa não implica conteúdo errado (reforça 2.3):** a q1 tem resposta
numérica **exatamente correta** (5,3186) mas similaridade de só 0,44, porque a resposta esperada é um
número isolado e a resposta gerada é uma frase completa. Mesmo problema do Revalida, agora mais
extremo — evidência de que, pra dado numérico, cosseno sobre texto livre sozinho não é métrica
suficiente (precisa complementar com extração/comparação numérica direta, não só embedding).

**Achado 2 — ambiguidade de escopo numérico em documento financeiro:** a q3 perguntou pelo saldo de
"caixa e bancos" (resposta correta: 13.092 mil reais, uma sub-linha específica da nota). O modelo
respondeu 92.011 mil reais — que é real e verificável (conferimos em 3 pontos do documento: balanço
patrimonial, demonstração de fluxo de caixa e nota de instrumentos financeiros), mas é o total de
**"Caixa e Equivalentes de Caixa"** (13.092 de caixa e bancos + 78.919 de aplicações de curto prazo).
O retriever trouxe a cifra mais proeminente no documento, não a sub-linha estreita perguntada. Não
tratamos isso como "erro do modelo" nem corrigimos a pergunta depois do fato — registramos como
achado real: perguntas numéricas em documentos financeiros densos precisam de escopo muito preciso,
porque quase todo valor tem uma variante "mais agregada" competindo por relevância no retrieval.

Dados brutos: `resultados/metricas/piloto_grendene_stuff.jsonl`.

### 2.5 Piloto real: paper IFES/SBC sobre RAG em português (paralelo ao Llama2 Paper)

- **Documento:** Medeiros, L. S. F.; Oliveira, H. T. A. — [*"Comparação de Modelos de Embeddings e LLMs para Geração Aumentada por Recuperação em Português"*](https://sol.sbc.org.br/index.php/semish/article/download/36829/36615/), SEMISH/CSBC, IFES (Instituto Federal do Espírito Santo). 12 páginas.
- **Por que esse documento:** achado durante a própria pesquisa de originalidade (seção 1) — é um artigo científico brasileiro, em português, sobre exatamente o mesmo tema do nosso TCC (compara embeddings + LLMs em RAG usando 3 corpora em PT-BR: Pirá, FairytaleQA PT-BR e SQuAD2 PT-BR). Serve tanto como documento de teste (paralelo ao Llama2 Paper) quanto como **referência bibliográfica direta pro nosso próprio TCC** — recomendo adicionar ao `referencias.bib` (ver seção 4).
- **Configuração:** mesma dos outros pilotos (Stuff, ChromaDB, BGE-M3, gpt-4o-mini, chunk 1000/100). 42 chunks de um único documento.

| Pergunta | Similaridade | Conteúdo correto? |
|---|---|---|
| q1 — quais as 3 bases de dados usadas | 0,8103 | ✅ sim |
| q2 — top-n definido como melhor equilíbrio | 0,9588 | ✅ sim |
| q3 — melhor embedding e melhor LLM nas conclusões | 0,9049 | ✅ sim |

Sem ambiguidades como no piloto financeiro — as 3 respostas bateram com o texto-fonte de forma
direta, com similaridade consistentemente alta. Dados brutos: `resultados/metricas/piloto_sbc_stuff.jsonl`.

## 3. Comparativo real entre os 6 métodos de RAG do paper-âncora

Reaproveitando os 3 documentos e as 9 perguntas já validadas (seção 2), rodamos as mesmas perguntas
com os **6 métodos de RAG do escopo original**, todos implementados em `pipeline_base.py`:

- **Stuff** — 1 chamada, todos os contextos de uma vez.
- **Refine** — percorre os `k=4` contextos em sequência, refinando a resposta (k chamadas).
- **Map-Reduce** — gera uma resposta parcial por contexto, depois combina todas numa final (k+1).
- **Map-Rerank** — gera resposta + nota de confiança por contexto, mantém só a de maior confiança (k).
- **Query Step-Down** — gera 4 perguntas alternativas a partir da pergunta original (1 chamada),
  responde cada uma isoladamente com Stuff, e combina as 4 respostas numa final (1+4+1=6 chamadas).
- **Reciprocal RAG** — mesmo primeiro passo do Step-Down, mas mantém só a resposta de maior
  confiança entre as 4 alternativas, sem combinar (1+4=5 chamadas).

As definições de Query Step-Down e Reciprocal RAG seguem a leitura literal da seção 2.5 do
paper-âncora (`Referências/Maximizando-Eficiência-RAG/`). Uma decisão de implementação que o paper
não deixa 100% explícita: se a etapa de combinação do Step-Down usa uma chamada de LLM extra — nós
optamos que sim, documentado no código (`gerar_resposta_query_step_down`).

> Nota: esta seção já teve quatro versões anteriores (Stuff vs. Refine só; +Map-Reduce e Map-Rerank;
> os 6 métodos com só 3 perguntas por domínio; depois com o Revalida expandido mas Grendene/SBC
> ainda em 3 perguntas cada), cada uma com uma conclusão diferente e só parcialmente correta.
> Mantemos esse histórico de correções explícito em vez de apagar o rastro. **Esta versão usa os 3
> domínios já equilibrados: Revalida (12), Grendene (7) e SBC (8) — 27 perguntas, 162 execuções.**

**Similaridade média por domínio e método:**

| Domínio (n perguntas) | Stuff | Refine | Map-Reduce | Map-Rerank | Query Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| Revalida — médico (12) | 0,555 | 0,687 | 0,684 | 0,714 | 0,698 | 0,586 |
| Grendene — financeiro (7) | 0,602 | 0,557 | 0,557 | 0,746 | 0,582 | 0,780 |
| SBC — técnico-científico (8) | 0,825 | 0,701 | 0,789 | 0,786 | 0,746 | 0,719 |
| **Média geral** | **0,647** | **0,657** | **0,682** | **0,744** | **0,682** | **0,676** |
| **Desvio padrão (geral)** | 0,212 | 0,143 | 0,169 | 0,183 | 0,159 | 0,188 |

**Custo médio por método (todos os domínios):**

| Método | Tokens médios | Tempo de geração médio (s) |
|---|---|---|
| Stuff | 1313 | 1,45 |
| Map-Rerank | 1931 | 4,13 |
| Map-Reduce | 1987 | 4,70 |
| Refine | 2453 | 6,47 |
| Reciprocal | 6592 | 8,16 |
| Query Step-Down | 6773 | 10,11 |

**O achado principal mudou de novo, e ficou mais claro — mesma transparência de antes:** com
Grendene e SBC em só 3 perguntas, Stuff parecia disputar a liderança em qualidade. Com as amostras
equilibradas (27 perguntas), **Stuff caiu para a última posição em similaridade média (0,647)** —
o que segurava sua média alta antes era o desempenho forte no SBC (0,891 com n=3), que se revelou
não representativo ao ampliar para 8 perguntas nesse domínio (0,825, ainda o melhor domínio pra
Stuff, mas puxando menos a média geral). Isso reforça um ponto metodológico: com n pequeno por
domínio, a média geral é facilmente dominada pelo domínio onde por acaso um método foi melhor —
exatamente o motivo de termos ampliado a amostra em vez de aceitar a conclusão anterior como final.

O que se sustenta e ganha clareza com a amostra maior:

- **Map-Rerank é o método com melhor similaridade média (0,744)**, e por boa margem — a distância
  para o segundo colocado (Map-Reduce/Query Step-Down, 0,682) é maior do que qualquer diferença
  observada nas rodadas anteriores. É o achado mais robusto do estudo até agora.
- **Reciprocal (0,676) e Query Step-Down (0,682) não são mais claramente piores que os métodos
  simples** — ficam em posição intermediária, empatados com Map-Reduce e acima de Stuff e Refine em
  qualidade. Mas continuam custando **3-5x mais tokens e 2-7x mais tempo** que qualquer outro
  método, sem alcançar o desempenho de Map-Rerank. A divergência do paper-âncora (lá, Reciprocal é
  o método de maior acurácia) permanece, mas de forma mais matizada: não é que Reciprocal seja
  "pior" aqui — é que **não compensa o custo extra**, porque Map-Rerank entrega mais qualidade por
  uma fração do preço.
- **Refine deixa de ser o pior método** (0,657, à frente de Stuff) — mas continua **dominado**: tanto
  Map-Reduce quanto Map-Rerank entregam mais qualidade E custam menos que Refine. Não há cenário,
  nesta amostra, em que Refine seja a escolha certa.
- **Stuff é o mais barato e mais rápido, mas agora com a pior qualidade média** — o clássico
  trade-off "eficiência vs. acurácia" apareceu de forma mais nítida com mais dados, não menos.

**Achados que se mantêm firmes:**

- **Stuff continua o piso de eficiência** (mais barato e mais rápido em todos os domínios) — só que
  agora fica claro que isso tem um preço real em qualidade, não é "quase de graça" como parecia com
  amostra pequena.
- **Métodos baseados em confiança (Map-Rerank e Reciprocal) tendem a respostas mais curtas e
  diretas**, favorecendo a métrica de cosseno em perguntas numéricas — a mesma pergunta do dólar
  (Grendene q1) teve similaridade **1,0000** em Map-Rerank e Reciprocal, de novo, reforçando o
  achado da seção 2.4 sobre verbosidade penalizar a métrica.
- **O desvio padrão é alto em todos os métodos** (0,14 a 0,21) — a variação entre perguntas
  individuais é grande, maior até que a diferença entre métodos em vários casos. Isso é esperado
  dado o tamanho da amostra (27 perguntas) e reforça que estas médias são um indicativo direcional,
  não uma conclusão estatisticamente robusta no sentido de um teste de hipótese formal.

Dados brutos: `resultados/metricas/comparativo_6_metodos_v3.jsonl` (as rodadas anteriores —
`comparativo_stuff_refine.jsonl`, `comparativo_4_metodos.jsonl`, `comparativo_6_metodos.jsonl`,
`comparativo_6_metodos_v2.jsonl` — ficam como registro histórico de amostras menores/desequilibradas).

## 4. Expansão do domínio médico + comparativo de bancos vetoriais (ChromaDB vs FAISS)

### 4.1 Revalida: de 3 para 12 perguntas

Adicionamos as Questões 1, 4 e 5 do mesmo caderno do Revalida 2024/2 (a Questão 2 foi
deliberadamente **descartada**: sua única referência bibliográfica é o manual oficial do curso
ATLS, pago e protegido por direitos autorais — não temos licença pra usá-lo). Itens que dependiam
de imagem (raio-X da Q1c) também foram descartados. Resultado: **12 perguntas** no domínio médico
(antes eram 3), cobrindo TEP/lúpus, sigilo com adolescente, trabalho de parto e doença
meningocócica pediátrica.

Cada nova questão trouxe seu próprio documento de referência, buscado do mesmo jeito seguro da
Questão 3 (usando as fontes que o próprio gabarito cita, nunca "achando" nós mesmos qual protocolo
clínico seria relevante):

| Questão | Documento-fonte | Origem |
|---|---|---|
| Q1 (TEP/lúpus) | Diretriz Conjunta sobre Tromboembolismo Venoso – 2022 (SBC/CBR/SBACV) | SciELO (open access) |
| Q4 (parto) | Artigo Febrasgo sobre recomendações OMS 2018 para parto | citado literalmente no gabarito |
| Q5 (meningite) | Guia de Vigilância em Saúde — capítulo Doença Meningocócica (Ministério da Saúde) | gov.br |

Os 5 documentos do domínio médico (CFM + SBP + esses 3 novos) agora formam **um único corpus
combinado** de 430 chunks — cada pergunta precisa que o retriever ache o documento certo entre
vários, não um índice isolado por pergunta. Isso é mais realista e testa precisão de recuperação de
verdade. Ainda não rodamos os 6 métodos de RAG sobre esse conjunto expandido (só validamos que a
recuperação funciona, seção 4.2) — é o próximo passo.

### 4.2 ChromaDB vs FAISS: infraestrutura, não qualidade

Como os dois bancos buscam sobre os mesmos vetores, comparar qualidade de resposta entre eles seria
redundante — o que varia é velocidade e memória (exatamente as métricas de infraestrutura do plano
de pesquisa). Medimos isso sem nenhuma chamada a LLM, nos 3 domínios (corpus médico já expandido):

| Domínio | Indexação Chroma | Indexação FAISS | Consulta Chroma | Consulta FAISS | Memória Chroma | Memória FAISS | Concordância top-k |
|---|---|---|---|---|---|---|---|
| Revalida (430 chunks) | 0,425s | 0,0008s | 3,06ms | 0,049ms | 38,7 MB | 2,1 MB | 100% |
| Grendene (199 chunks) | 0,171s | 0,0001s | 5,92ms | 0,022ms | 2,8 MB | 0,02 MB | 100% |
| SBC (42 chunks) | 0,095s | 0,0001s | 2,82ms | 0,016ms | 3,3 MB | 0,0 MB | 100% |

**Achados:**

- **Concordância de 100%** entre os dois bancos em todas as perguntas de todos os domínios — os
  chunks recuperados são exatamente os mesmos, como esperado (mesma distância L2, mesmos vetores).
  Confirma que vale a pena comparar bancos vetoriais só por infraestrutura, não por qualidade.
- **FAISS é dramaticamente mais rápido**: ~200-500x mais rápido pra indexar, ~60-190x mais rápido
  por consulta, e usa uma fração da memória do ChromaDB.
- **Isso não quer dizer "sempre usar FAISS"**: o `IndexFlatL2` do FAISS usado aqui é uma estrutura
  pura em memória, sem persistência em disco nem armazenamento de metadados — o ChromaDB entrega
  isso pronto (é para onde vai boa parte do overhead medido). A comparação justa não é "qual é mais
  rápido" isolado, é "vale a pena pagar o overhead do Chroma pela conveniência de persistência e
  filtros de metadado" — pergunta que fica pro capítulo de discussão do TCC.

**Nota técnica que quase nos custou 1h+ de máquina:** rodar FAISS e o modelo de embedding (PyTorch)
no mesmo processo travou duas vezes (disputa pelo runtime OpenMP) — resolvido separando em dois
processos: um gera e salva os embeddings em `.npy` (`_gerar_vetores_para_benchmark.py`), outro só
importa FAISS e lê os `.npy` do disco (`_medir_faiss_puro.py`), nunca no mesmo processo que o
PyTorch. Também corrigimos, nessa mesma rodada, um bug em `construir_indice`/`recuperar` que
reencodava os textos duas vezes sem necessidade — os dois agora aceitam vetores já calculados.

Dados brutos: `resultados/metricas/comparativo_vectorstores.json`.

## 5. Compressão contextual (o eixo do plano que ainda faltava)

Implementamos o filtro de compressão contextual como um limiar de similaridade de cosseno sobre os
`k=4` chunks recuperados: descarta os chunks abaixo do limiar antes de gerar a resposta (nunca
descarta todos — se sobrar zero, mantém ao menos o mais similar). Testamos 4 limiares (0,0 = sem
filtro / 0,3 / 0,5 / 0,7), usando o método **Stuff** (mais barato, suficiente pra medir o trade-off
em si, sem precisar repetir isso nos outros 5 métodos), nos 18 pares pergunta-resposta dos 3
domínios.

**Achado de calibração, honesto:** os limiares 0,3 e 0,5 **não tiveram nenhum efeito** — em
nenhuma das 18 perguntas, em nenhum domínio, algum chunk do top-4 ficou abaixo de 0,5 de
similaridade. Só o limiar "alto" (0,7) de fato cortou alguma coisa. Isso não invalida o
experimento — é em si um achado: **para esses corpora e esse embedding (BGE-M3), a similaridade
dos chunks recuperados por uma busca top-4 já vem naturalmente alta**, então limiares "moderados"
copiados de outro contexto (ex.: os 80%/90% do paper-âncora, calculados com embeddings diferentes)
não são diretamente transferíveis — cada combinação embedding+corpus precisa da própria calibração
empírica, não dá pra assumir um valor "razoável" de antemão.

**Com o limiar alto (0,7), o trade-off apareceu com clareza:**

| Domínio | Redução média de tokens | Variação média de similaridade |
|---|---|---|
| Revalida (12 perguntas) | -64% | -29% (0,591 → 0,419) |
| Grendene (3 perguntas) | -75% | -7% (0,619 → 0,573) |
| SBC (3 perguntas) | -68% | -24% (0,891 → 0,682) |

Em média, **~69% de economia de tokens custou ~20% de queda na similaridade** — na maioria das
perguntas o corte foi de 4 chunks para 1 só. Em algumas perguntas específicas a queda foi drástica
(ex.: Revalida Q3-b caiu de 0,879 pra 0,433; SBC q2 caiu de 0,959 pra 0,354), sinal de que o único
chunk mantido às vezes não era o que continha a informação relevante. Isso bate com o que o
paper-âncora já alertava sobre "limiares punitivos" — aqui, com dado nosso, não emprestado.

Dados brutos: `resultados/metricas/comparativo_compressao_contextual.jsonl`.

## 6. Avaliação com RAGAS — a similaridade de cosseno não conta a história toda

Até aqui, a única métrica de qualidade era similaridade de cosseno entre resposta gerada e
resposta esperada — a mesma limitação que o `plano_de_pesquisa.md` (seção 15) já previa, e que o
próprio paper-âncora e o paper do IFES/SBC contornam usando **RAGAS**. Implementamos 4 métricas
RAGAS (`ragas` 0.4.3, API "collections", juiz `gpt-4o-mini`) sobre as 108 linhas do comparativo de
6 métodos na versão **v2** (`comparativo_6_metodos_v2.jsonl` — não o v3 da seção 3, que tem 162
linhas e a amostra equilibrada; o RAGAS sobre o v3 está em andamento, ver seção 8): **Answer Relevancy** (a resposta é relevante e não evasiva?), **Faithfulness**
(a resposta é factualmente sustentada pelo contexto recuperado?), **Context Precision** e
**Context Recall** (o contexto recuperado é útil e suficiente?). As duas últimas só valem pros 4
métodos com contexto real (Stuff/Refine/Map-Reduce/Map-Rerank) — Query Step-Down e Reciprocal
logam perguntas alternativas nesse campo, não trechos recuperados, então ficam de fora dessas duas
métricas especificamente (não inventamos valor pra isso).

| Método | Answer Relevancy | Faithfulness | Context Precision | Context Recall | (referência) Similaridade cosseno |
|---|---|---|---|---|---|
| Stuff | 0,536 | 0,556 | 0,844 | 0,841 | 0,701 |
| Refine | 0,681 | **0,764** | 0,858 | 0,859 | 0,638 (pior) |
| Map-Reduce | **0,802** | 0,541 | 0,845 | 0,804 | 0,687 |
| Map-Rerank | 0,454 | 0,583 | 0,858 | 0,822 | **0,777** (melhor) |
| Query Step-Down | 0,756 | n/a | n/a | n/a | 0,672 |
| Reciprocal | **0,323** (pior) | n/a | n/a | n/a | 0,713 |

**Achado principal: cosseno e RAGAS discordam, na direção oposta, exatamente no par que mais
importava.** O **Refine** — o método com pior similaridade de cosseno em toda a seção 3 — é
disparado o método **mais fiel ao contexto recuperado** (0,764, muito acima dos outros três). O
**Map-Rerank** — o método com melhor similaridade de cosseno — não é o pior em fidelidade (isso é
Map-Reduce, 0,541), mas também não se destaca (0,583, 2º lugar): ele vence por bater com a
*fraseologia* da resposta esperada, não por ser o mais ancorado no documento. Isso quer dizer que
**"melhor por similaridade de cosseno" e "mais confiável/fundamentado" são coisas diferentes, e um
método pode ganhar em uma métrica e não se destacar na outra** — não dá pra escolher "o melhor
método de RAG" olhando só pra uma métrica de qualidade, tem que declarar qual noção de qualidade
importa pro caso de uso.

**Achados secundários:**
- **Map-Reduce** tem a melhor relevância de resposta (0,802) — suas respostas respondem diretamente
  à pergunta, mesmo não sendo as mais fiéis ao contexto nem as mais parecidas com o gabarito.
- **Reciprocal** tem a pior relevância de resposta (0,323) de todos os 6 métodos, mesmo com
  similaridade de cosseno mediana (0,713) — sinal de que a resposta escolhida (maior confiança
  entre 4 alternativas) às vezes deriva do que a pergunta original pedia.
- **Context Precision e Context Recall variam pouco entre os 4 métodos com contexto** (0,84-0,86 e
  0,80-0,86) — faz sentido, os quatro usam a mesma recuperação (mesmo `k=4`, mesma pergunta); a
  diferença de qualidade entre métodos vem de como cada um *usa* o contexto na geração, não de
  recuperar contextos diferentes.
- **Taxa de erro:** 5 de 72 avaliações de fidelidade e 1 de contexto falharam (estouro de limite de
  tokens do juiz LLM em respostas muito longas, principalmente do Refine) — ficaram como `None`
  registrado, não um valor inventado. ~7% de falha, vale monitorar se crescer.

Dados brutos: `resultados/metricas/comparativo_6_metodos_v2_ragas.jsonl` (mesmo conteúdo de
`comparativo_6_metodos_v2.jsonl`, com um campo `ragas` adicionado por linha).

## 7. Arquivos gerados nesta etapa

```
Pesquisa/dados/brutos/revalida_2024_2_prova_discursiva.{pdf,txt}
Pesquisa/dados/brutos/revalida_2024_2_padrao_resposta.{pdf,txt}
Pesquisa/dados/brutos/codigo_etica_medica_cfm.{pdf,txt}
Pesquisa/dados/brutos/sbp_consulta_adolescente.{pdf,txt}
Pesquisa/dados/brutos/grendene_itr_3t2025.{pdf,txt}
Pesquisa/dados/brutos/sbc_embeddings_llms.{pdf,txt}
Pesquisa/dados/brutos/revalida_ref_tev_scielo.{pdf,txt}
Pesquisa/dados/brutos/revalida_ref_parto_febrasgo.{html,txt}
Pesquisa/dados/brutos/revalida_ref_meningococica_ms.{pdf,txt}
Pesquisa/dados/processados/revalida_2024_2_questao3.json      <- histórico (só Q3, 3 perguntas)
Pesquisa/dados/processados/revalida_2024_2_completo.json      <- atual (Q1+Q3+Q4+Q5, 12 perguntas)
Pesquisa/dados/processados/grendene_itr_3t2025.json    <- 7 perguntas (era 3)
Pesquisa/dados/processados/sbc_embeddings_llms.json    <- 8 perguntas (era 3)
Pesquisa/scripts/pipeline_base.py                      <- pipeline base: 6 métodos de RAG, Chroma/FAISS, backend Ollama/OpenAI e embedding EN/multilíngue configuráveis por .env
Pesquisa/scripts/piloto_revalida.py                    <- runner do piloto médico em português (histórico, só Q3)
Pesquisa/scripts/piloto_grendene.py                    <- runner do piloto financeiro em português
Pesquisa/scripts/piloto_sbc_paper.py                   <- runner do piloto técnico-científico em português
Pesquisa/scripts/comparar_metodos.py                   <- comparativo dos 6 métodos, 27 perguntas equilibradas (12+7+8)
Pesquisa/scripts/testar_compressao.py                  <- teste de compressão contextual (4 limiares, método Stuff), retoma de onde parou
Pesquisa/scripts/avaliar_ragas.py                      <- avaliação RAGAS (4 métricas) sobre um log já rodado, retoma de onde parou
Pesquisa/scripts/_gerar_vetores_para_benchmark.py      <- passo 1/2 do comparativo Chroma vs FAISS (embeddings + Chroma)
Pesquisa/scripts/_medir_faiss_puro.py                  <- passo 2/2 (só FAISS, processo separado)
Pesquisa/resultados/metricas/piloto_revalida_stuff.jsonl
Pesquisa/resultados/metricas/piloto_grendene_stuff.jsonl
Pesquisa/resultados/metricas/piloto_sbc_stuff.jsonl
Pesquisa/resultados/metricas/comparativo_stuff_refine.jsonl   <- histórico (rodada com só 2 métodos)
Pesquisa/resultados/metricas/comparativo_4_metodos.jsonl      <- histórico (rodada com 4 métodos, Revalida com 3 perguntas)
Pesquisa/resultados/metricas/comparativo_6_metodos.jsonl      <- histórico (6 métodos, Revalida ainda com 3 perguntas)
Pesquisa/resultados/metricas/comparativo_6_metodos_v2.jsonl   <- histórico (6 métodos, Revalida 12 perguntas, mas Grendene/SBC ainda em 3)
Pesquisa/resultados/metricas/comparativo_6_metodos_v3.jsonl   <- atual (6 métodos, 27 perguntas equilibradas: 12+7+8)
Pesquisa/resultados/metricas/comparativo_vectorstores.json    <- ChromaDB vs FAISS (infraestrutura)
Pesquisa/resultados/metricas/comparativo_compressao_contextual.jsonl    <- rodada de 18 perguntas (seção 5)
Pesquisa/resultados/metricas/comparativo_compressao_contextual_v2.jsonl <- atual (27 perguntas equilibradas, 108 execuções)
Pesquisa/resultados/metricas/comparativo_6_metodos_v2_ragas.jsonl       <- v2 + campo `ragas` (seção 6)
Pesquisa/resultados/metricas/comparativo_6_metodos_v3_ragas.jsonl       <- v3 + campo `ragas` (em andamento)
```

## 8. Pendências em aberto

**Decisões já tomadas:**
- Mudança de direção (domínios PT-BR) **validada com o orientador**.
- Descartado o Llama2 Paper (arquivos removidos) — foco 100% nos 3 domínios PT-BR.
- Revalida expandido de 3 para 12 perguntas (Q1+Q3+Q4+Q5; Q2/ATLS ficou de fora por direito autoral).
- Grendene expandido de 3 para 7 perguntas; SBC de 3 para 8 — amostra agora equilibrada entre os
  3 domínios (27 perguntas, 162 execuções no comparativo dos 6 métodos).
- Compressão contextual implementada e testada (seção 5).
- ChromaDB vs FAISS comparado por infraestrutura (seção 4.2).
- RAGAS **passou a ser coberto** (seção 6) — revisa a decisão anterior de não cobrir, já que a
  seção 6 mostrou que só cosseno esconde diferenças reais entre os métodos.
- Pinecone e embeddings proprietários (Cohere/OpenAI): **decisão consciente de não cobrir** — o
  escopo fica com BGE-M3 + ChromaDB/FAISS.

**Trabalho técnico — situação em 27/09/2026:**
- ✅ RAGAS sobre o v3 completo (162/162, sem falhas) — seção 9. A seção 6 fica como histórico (v2).
- ✅ Refine × Stuff no Revalida: explicado pelas recusas do Stuff ("Não sei." em 6 das 12 perguntas
  do Revalida), na maioria com o contexto contendo a resposta segundo o Context Recall — seção 9.
- ✅ Compressão na amostra equilibrada (27 perguntas), com limiares fixos e calibrados — seção 9.
  A seção 5 fica como histórico (18 perguntas).
- ✅ Gráficos e análise estatística (`scripts/analisar_resultados.py` → `resultados/tabelas/`,
  `resultados/graficos/`, `resultados/analise_estatistica.md`).
- ✅ Redação: artigo completo em `Artigo_TCC02/` (rascunho para revisão).
- ⏳ Não feito: repetir os 6 métodos ponta a ponta com FAISS (a recuperação é idêntica, seção 4.2;
  só mudaria o tempo de recuperação, que já é desprezível diante da geração).
- ⏳ Não feito: validar as métricas automáticas contra avaliação humana (limitação declarada no artigo).

## 9. Rodada final (27/09/2026)

Todos os números abaixo saem de `scripts/analisar_resultados.py`; o resumo completo, com todos os
testes, está em `resultados/analise_estatistica.md`, e a redação em `Artigo_TCC02/`.

**Estatística do comparativo (v3, 27 perguntas × 6 métodos).** Friedman sobre a similaridade de
cosseno: χ² = 7,21, p = 0,205 — nenhum par sobrevive à correção de Holm. O "Map-Rerank é o melhor"
da seção 3 é uma tendência (0,744, IC 95% [0,67; 0,81]), não uma diferença estatisticamente
estabelecida. Em custo, as diferenças são grandes: Query Step-Down e Reciprocal usam ~5× os tokens
do Stuff.

**Recusas.** Stuff: 8 de 27 respostas foram "Não sei." (6 no Revalida). Sem as recusas, a
similaridade do Stuff vai de 0,647 para 0,746 (Map-Rerank sem recusas: 0,758). Nas 13 recusas dos
métodos com contexto, o Context Recall médio foi 0,744 (10 com CR ≥ 0,5): na maioria das vezes, o
contexto tinha a resposta e o modelo recusou. As recusas também são instáveis: repetindo o Stuff três
vezes (temperatura 0), `revalida_q3_b` recusou numa execução e respondeu certo nas outras duas.

**RAGAS (juiz gpt-4o-mini, sem falhas).**

| Método | Answer Rel. | Faithfulness | Faithf. sem recusas | Cosseno |
|---|---|---|---|---|
| Stuff | 0,545 | 0,582 | 0,827 | 0,647 |
| Refine | 0,706 | 0,749 | 0,808 | 0,657 |
| Map-Reduce | 0,745 | 0,635 | 0,666 | 0,682 |
| Map-Rerank | 0,427 | 0,631 | 0,656 | 0,744 |
| Query Step-Down | 0,782 | n/a | n/a | 0,682 |
| Reciprocal | 0,285 | n/a | n/a | 0,676 |

Answer Relevancy é a única métrica com diferença significativa (Friedman p < 0,001), e ordena os
métodos quase ao contrário do cosseno: respostas curtas ("5,3186") ganham no cosseno e perdem na
Answer Relevancy. Correlação cosseno × RAGAS, linha a linha: ρ entre 0,16 e 0,35.

**Validação do juiz (amostra estratificada de 54, seed 42).** Reteste do próprio gpt-4o-mini e juiz
independente Cohere Command A (`command-a-03-2025`, chave Trial — 20 chamadas/min).

| Métrica | ρ reteste | ρ inter-juiz | Viés (4o-mini − Command A) |
|---|---|---|---|
| Answer Relevancy | 0,99 | 0,93 (mesma ordem de métodos, τ = 1) | −0,050 (p < 0,001) |
| Faithfulness | 0,76 | 0,57 | −0,038 (p = 0,616) |
| Context Precision | 0,78 | 0,03 | +0,181 (p = 0,004) |
| Context Recall | 0,84 | 0,87 | +0,065 (p = 0,068) |

Sem sinal de autopreferência (o 4o-mini é igual ou mais rigoroso com as próprias respostas).
Answer Relevancy e Context Recall são robustos; Faithfulness é moderada; Context Precision não é
confiável. Reteste embutido nos dados: os 4 métodos com contexto recebem os mesmos fragmentos, e
mesmo assim a Context Precision coincidiu nos 4 em só 21 de 27 perguntas.

**Compressão contextual (Stuff, 27 perguntas).** Limiares fixos: 0,3 e 0,5 não cortam nada; 0,7
corta 65,4% dos tokens com −16,1% de similaridade (p = 0,004). Limiares calibrados pelos quantis
25/50/75 da similaridade do top-4 de cada domínio (`resultados/metricas/limiares_calibrados.json`):

| Domínio | P50: Δ tokens / Δ similaridade | P75: Δ tokens / Δ similaridade |
|---|---|---|
| Revalida | −32,6% / −14,6% | −45,5% / −20,4% |
| Grendene | −44,9% / 0,0% | −58,0% / −0,5% |
| SBC | −41,5% / +0,1% | −56,0% / −2,9% |

Calibração dentro da amostra (mesmas perguntas) — os ganhos são um limite superior.

**Problemas de engenharia resolvidos nesta rodada** (vale saber pra próximas rodadas):
- A telemetria do RAGAS fazia uma chamada de rede bloqueante por avaliação (~9 s vs ~0,6 s por
  chamada ao juiz). Desligada via `RAGAS_DO_NOT_TRACK=true` em `avaliar_ragas.py`.
- Paralelismo estourou o limite de 200 mil tokens/min da OpenAI e o de 20 chamadas/min da chave
  Trial da Cohere. `avaliar_ragas.py` agora tem limitador de requisições (`--por-minuto`) e refaz só
  as métricas que falharam em linhas já gravadas.
- A lentidão do BGE-M3 é da CPU (sem GPU; o FlagEmbedding já usa float32 na CPU). Indexar os 430
  trechos do Revalida leva ~10 min com a máquina ocupada.
