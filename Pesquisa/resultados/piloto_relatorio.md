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

> Nota: esta seção já teve duas versões anteriores (Stuff vs. Refine só; depois +Map-Reduce e
> Map-Rerank), cada uma com uma conclusão diferente e só parcialmente correta. Mantemos esse
> histórico de correções explícito em vez de apagar o rastro — é assim que a leitura muda conforme
> mais dados entram.

**Similaridade média por domínio e método:**

| Domínio | Stuff | Refine | Map-Reduce | Map-Rerank | Query Step-Down | Reciprocal |
|---|---|---|---|---|---|---|
| Revalida (médico) | 0,800 | 0,760 | 0,780 | 0,782 | 0,777 | 0,656 |
| Grendene (financeiro) | 0,619 | 0,529 | 0,571 | 0,815 | 0,569 | 0,810 |
| SBC (técnico-científico) | 0,891 | 0,694 | 0,793 | 0,784 | 0,782 | 0,736 |
| **Média geral** | **0,770** | **0,661** | **0,714** | **0,794** | **0,709** | **0,734** |

**Custo médio por método (todos os domínios):**

| Método | Tokens médios | Tempo de geração médio (s) |
|---|---|---|
| Stuff | 1339 | 1,54 |
| Map-Rerank | 1922 | 3,51 |
| Map-Reduce | 2028 | 4,40 |
| Refine | 2437 | 5,57 |
| Reciprocal | 6699 | 6,52 |
| Query Step-Down | 6835 | 8,08 |

**Achado principal — diverge do paper-âncora, e registramos isso abertamente:** no paper original,
Reciprocal RAG é o método de **maior** acurácia (91% de similaridade mediana). No nosso piloto,
**Query Step-Down e Reciprocal RAG foram ~5x mais caros que o Stuff em tokens (e os mais lentos),
sem entregar a melhor qualidade** — ficaram atrás de Map-Rerank e do próprio Stuff em similaridade
média. Não sabemos ainda a causa exata; hipóteses possíveis, nenhuma confirmada:

1. Diferença de implementação: nossa leitura da etapa de combinação/geração de perguntas
   alternativas pode não ser idêntica à dos autores (o paper não detalha o suficiente pra
   reproduzir com 100% de fidelidade — ver nota acima).
2. Nossos documentos são muito menores e mais focados (1 documento, poucos chunks) que os do paper
   original — gerar 4 perguntas alternativas pode ajudar mais em corpora grandes e diversos do que
   em um documento único e curto, onde a pergunta original já recupera o que importa.
3. Amostra pequena (3 perguntas por domínio) — pode ser ruído que desaparece com mais perguntas.

Isso é um achado genuinamente interessante pro TCC (questionar se um resultado da literatura se
mantém em outro cenário é uma contribuição válida), mas precisa ficar marcado como **divergência
observada, não conclusão**, até investigarmos mais.

**Achados que se mantêm firmes com mais métodos testados:**

- **Refine continua sendo dominado** — nunca é o melhor em qualidade nem em custo, em nenhum domínio.
- **Map-Rerank tem a melhor similaridade média (0,794)** entre todos os 6, a um custo moderado
  (2º mais barato) — o melhor trade-off custo/qualidade encontrado até agora.
- **Stuff continua sendo o piso de eficiência** (mais barato e mais rápido), com qualidade próxima
  ao topo.
- **Métodos baseados em confiança (Map-Rerank e Reciprocal) tendem a respostas mais curtas e
  diretas**, o que favorece a métrica de cosseno em perguntas numéricas — a mesma pergunta do dólar
  (Grendene q1) teve similaridade **1,0000** tanto no Map-Rerank quanto no Reciprocal, reforçando o
  achado da seção 2.4 sobre verbosidade penalizar a métrica.

Ainda é leitura preliminar (3 perguntas por domínio). Dados brutos:
`resultados/metricas/comparativo_6_metodos.jsonl` (as rodadas anteriores, `comparativo_stuff_refine.jsonl`
e `comparativo_4_metodos.jsonl`, ficam como registro histórico).

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

## 5. Arquivos gerados nesta etapa

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
Pesquisa/dados/processados/grendene_itr_3t2025.json
Pesquisa/dados/processados/sbc_embeddings_llms.json
Pesquisa/scripts/pipeline_base.py                      <- pipeline base: 6 métodos de RAG, Chroma/FAISS, backend Ollama/OpenAI e embedding EN/multilíngue configuráveis por .env
Pesquisa/scripts/piloto_revalida.py                    <- runner do piloto médico em português (histórico, só Q3)
Pesquisa/scripts/piloto_grendene.py                    <- runner do piloto financeiro em português
Pesquisa/scripts/piloto_sbc_paper.py                   <- runner do piloto técnico-científico em português
Pesquisa/scripts/comparar_metodos.py                   <- comparativo dos 6 métodos, 3 domínios (ainda não rodado sobre o Revalida expandido)
Pesquisa/scripts/_gerar_vetores_para_benchmark.py      <- passo 1/2 do comparativo Chroma vs FAISS (embeddings + Chroma)
Pesquisa/scripts/_medir_faiss_puro.py                  <- passo 2/2 (só FAISS, processo separado)
Pesquisa/resultados/metricas/piloto_revalida_stuff.jsonl
Pesquisa/resultados/metricas/piloto_grendene_stuff.jsonl
Pesquisa/resultados/metricas/piloto_sbc_stuff.jsonl
Pesquisa/resultados/metricas/comparativo_stuff_refine.jsonl   <- histórico (rodada com só 2 métodos)
Pesquisa/resultados/metricas/comparativo_4_metodos.jsonl      <- histórico (rodada com 4 métodos)
Pesquisa/resultados/metricas/comparativo_6_metodos.jsonl      <- 6 métodos, mas ainda com Revalida em só 3 perguntas
Pesquisa/resultados/metricas/comparativo_vectorstores.json    <- ChromaDB vs FAISS (infraestrutura)
```

## 6. Pendências em aberto

**Decisão já tomada:** descartado o Llama2 Paper (arquivos removidos) — foco 100% nos 3 domínios PT-BR.

**Decisão que precisa de vocês/orientador:**
- Adicionar Medeiros & Oliveira (2025) — o paper do piloto técnico-científico — ao `referencias.bib`:
  é diretamente relevante como trabalho relacionado, não só como corpus de teste.

**Trabalho técnico (sem decisão pendente, só execução):**
- Rodar as outras 4 questões discursivas do mesmo caderno do Revalida — cada uma precisa de um
  corpus/protocolo clínico próprio (ex: TEP em lúpus, ATLS/trauma, parto OMS, doença meningocócica
  pediátrica). Sourcing desses é mais delicado (julgamento clínico), então cada um deve ser conferido
  com cuidado antes de eu buscar sozinho.
- Repetir esse mesmo experimento com os métodos **Refine** e **Reciprocal RAG** (o RRF do
  `Busca_Hibrida_RRF.ipynb` já cobre boa parte do caminho para Reciprocal).
- Repetir com **FAISS** no lugar do ChromaDB, mesmo corpus/perguntas, para o primeiro comparativo de
  banco vetorial.
