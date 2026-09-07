# Análise geral — onde o TCC está e o que falta

Data: 2026-09-06. Este documento junta o anteprojeto (`Plano_Trabalho/`), o plano de pesquisa
(`Pesquisa/plano_de_pesquisa.md`) e os resultados reais (`Pesquisa/resultados/piloto_relatorio.md`)
num raio-x único: o que já existe, o que precisa virar texto de TCC, e o que ainda falta.

## 1. Inventário — o que existe hoje

| Peça | Onde está | Estado |
|---|---|---|
| Anteprojeto (TCC01) | `Plano_Trabalho/ECO_plano_de_trabalho_TCC01/` | **Já enviado e aprovado pela banca** (junho/2026) |
| Plano de pesquisa detalhado | `Pesquisa/plano_de_pesquisa.md` | Escrito, descreve o desenho original |
| Revisão bibliográfica | `Plano_Trabalho/.../03-revisao.tex` + `referencias.bib` | 12 referências, texto substancial já escrito |
| Pipeline experimental | `Pesquisa/scripts/pipeline_base.py` | Funcional: 6 métodos de RAG, ChromaDB + FAISS, embedding EN/PT configurável, backend Ollama/OpenAI |
| Dados experimentais reais | `Pesquisa/resultados/metricas/*.jsonl` | **180 execuções reais** logadas (108 no comparativo de 6 métodos + 72 no teste de compressão), mais o benchmark de infraestrutura Chroma/FAISS |
| Relatório de resultados | `Pesquisa/resultados/piloto_relatorio.md` | Narrativa completa e honesta de tudo que foi rodado |
| Texto final da monografia | — | **Não existe ainda** (só o anteprojeto, que é uma proposta, não o TCC em si) |

## 2. A divergência que precisa de decisão do orientador (o item mais crítico)

Reli o anteprojeto inteiro agora (`01-introducao.tex`, `02-motivacao.tex`, `03-revisao.tex`,
`05-materiais_e_metodos.tex`, `06-resultados.tex`). Ele foi **formalmente enviado e aprovado pela
banca do TCC01** descrevendo, sem nenhuma ambiguidade, o desenho **original**:

- Datasets em inglês: SEC 10Q, Llama2 ArXiv, MedQA.
- Bancos vetoriais: ChromaDB, **FAISS e Pinecone**.
- Embeddings: Bge-en-small, **Text-embedding-3-large, Cohere-en-v3**.
- Métricas: similaridade de cosseno **e RAGAS**.

O que foi de fato executado (documentado no `piloto_relatorio.md`) é **diferente** em pontos
centrais:

- Datasets: Revalida (INEP), Grendene (ITR/B3), paper IFES/SBC — todos em português, nenhum dos
  três do anteprojeto.
- Bancos vetoriais: só ChromaDB e FAISS — **Pinecone nunca foi testado**.
- Embeddings: só BGE-M3 — **Text-embedding-3-large e Cohere-en-v3 nunca foram testados**.
- Métricas: só similaridade de cosseno — **RAGAS nunca foi implementado**.

Isso **não invalida o trabalho** — a mudança de direção tem justificativa forte e documentada
(risco de sobreposição com Şakar & Emekci 2025, ver seção 1 do `piloto_relatorio.md`) — mas é uma
divergência real entre o que a banca aprovou e o que está sendo entregue. **Isso precisa ser
formalmente comunicado e justificado ao orientador antes de qualquer coisa ser considerada
"fechada".** Não é birocracia: se o texto final do TCC simplesmente aparecer com um desenho
diferente do anteprojeto sem explicação, isso é o tipo de coisa que gera pergunta difícil na
defesa. A boa notícia é que a justificativa já está escrita e é sólida — só falta a conversa.

## 3. Peça por peça: o que aproveitar, o que reescrever, o que falta

### 3.1 Introdução e motivação (`01-introducao.tex`, `02-motivacao.tex`)

**Aproveitável quase integralmente.** Os textos falam de RAG, tradeoffs de hardware/tokens/
acurácia, e citam Şakar & Emekci como base metodológica — tudo isso continua verdadeiro. Só
precisam de um parágrafo novo explicando a adaptação pra datasets brasileiros (o "porquê"
já está pronto no `piloto_relatorio.md`, seção 1 — é praticamente copiar e adaptar o tom).

### 3.2 Revisão bibliográfica (`03-revisao.tex`)

**Aproveitável, com uma adição importante.** O texto é substancial e bem estruturado (6 subseções,
12 referências). Falta incluir **Medeiros & Oliveira (2025)** — o paper do IFES/SBC que vocês usaram
como corpus de teste e que é diretamente um trabalho relacionado (compara embeddings e LLMs em RAG
para português). Isso já está anotado como pendência no `piloto_relatorio.md`. Vale também um
parágrafo mencionando a ausência de estudos com dados brasileiros em português — é exatamente a
lacuna que a pesquisa de vocês agora preenche, e fortalece a seção "Lacunas identificadas".

### 3.3 Materiais e métodos (`05-materiais_e_metodos.tex`)

**Precisa ser reescrito.** Hoje descreve Pinecone, embeddings proprietários e os 3 datasets em
inglês — nada disso reflete o que foi executado. A boa notícia: o `piloto_relatorio.md` já tem,
seção a seção, tudo que precisa entrar aqui (datasets, fontes, configuração do pipeline, parâmetros
de chunking, bancos vetoriais testados, limitações assumidas como a exclusão da Questão 2 do
Revalida por direito autoral). É mais trabalho de **tradução pra prosa acadêmica** do que de
descoberta nova.

### 3.4 Resultados (`06-resultados.tex`)

**Maior lacuna de forma (não de substância).** O texto promete matrizes de desempenho, gráficos de
barras agrupadas, tabelas cruzando limiares de compressão com embeddings, e métricas RAGAS. O que
existe:

- ✅ Substância real: 180 execuções, 6 métodos, 3 domínios, comparação de bancos vetoriais,
  compressão contextual testada e com achado claro.
- ❌ **Nenhum gráfico foi gerado ainda** — tudo está em tabelas markdown/JSONL, não em figuras.
- ❌ **Nenhuma análise estatística formal** — o anteprojeto promete desvio padrão e frequências;
  hoje só temos médias simples. Fácil de calcular (os dados já existem), mas não foi feito.
- ❌ **RAGAS nunca foi usado.**
- ❌ Amostra pequena por domínio (3-12 perguntas) — o anteprojeto fala em escala de milhares de
  iterações (herdada do paper-âncora); mesmo reconhecendo o corte de escopo deliberado, vale deixar
  isso explícito no texto como limitação assumida, não escondida.

### 3.5 Discussão e conclusão

**Não existe ainda em lugar nenhum** — nem no anteprojeto (que é só proposta), nem nos relatórios
de pesquisa. É preciso escrever do zero. Boa notícia: os "achados" já documentados no
`piloto_relatorio.md` (Map-Rerank como melhor custo-benefício, divergência do Reciprocal RAG frente
ao paper-âncora, o problema de verbosidade na métrica de cosseno, a calibração de limiares de
compressão) já são o conteúdo bruto de uma discussão substantiva — falta só a redação.

## 4. Veredito honesto: dá pra entregar um TCC com o que existe?

**Sim, a substância já é real e defensável.** 180 execuções reais, com dados oficiais (INEP, CVM/B3,
paper científico), rigor de sourcing (nunca inventamos gabarito), achados que divergem
honestamente da literatura em vez de confirmá-la por conveniência — isso é ciência de verdade, não
só um exercício de programação.

**Mas ainda não está "redondo"** porque falta:

1. Validar a mudança de direção com o orientador (bloqueante — sem isso, o resto é retrabalho em
   risco).
2. Reescrever "Materiais e métodos" e escrever "Resultados"/"Discussão"/"Conclusão" no texto real
   do TCC (hoje só existe como relatório de pesquisa em markdown, não como monografia).
3. Gerar gráficos de verdade (nenhum existe ainda).
4. Rodar uma análise estatística mínima (desvio padrão, pelo menos) sobre os dados que já existem.
5. Decidir o que fazer com as 3 lacunas do anteprojeto que não foram cobertas (Pinecone, embeddings
   proprietários, RAGAS) — cobrir cada uma, ainda que num teste pequeno, ou justificar
   explicitamente por que foram deixadas de fora.

## 5. Plano de ação priorizado

**Bloqueante, fazer primeiro:**
- Marcar conversa com o orientador (Carlos Henrique Valério de Moraes) pra validar a mudança de
  desenho. Levar o `piloto_relatorio.md` como base — já está pronto pra isso.

**Rápido e de alto impacto (dá pra fazer sem esperar o orientador):**
- Gerar os gráficos a partir dos JSONL já existentes (matplotlib/seaborn, como o próprio plano já
  previa).
- Calcular desvio padrão e outras estatísticas simples sobre os dados já coletados.
- Adicionar Medeiros & Oliveira (2025) ao `referencias.bib`.

**Decisão de escopo (conversar com o orientador antes, mas vale já ter a opinião formada):**
- RAGAS: vale pelo menos uma rodada pequena, já que foi prometido explicitamente no anteprojeto.
- Pinecone e embeddings proprietários (Cohere/OpenAI embeddings): testar um mínimo, ou assumir e
  justificar a omissão por escrito (custo/dependência de API paga é uma justificativa legítima,
  mas precisa estar no texto, não só na nossa cabeça).

**Redação (depois do sinal verde do orientador):**
- Adaptar Introdução/Motivação (pequeno ajuste).
- Reescrever Materiais e Métodos com base no `piloto_relatorio.md`.
- Escrever Resultados, Discussão e Conclusão do zero, usando os achados já documentados.
