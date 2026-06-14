# Plano de Pesquisa

## Título provisório

Maximizando a eficiência de RAG: uma análise comparativa de métodos de recuperação, bancos vetoriais, embeddings e compressão contextual.

## 1. Delimitação da pesquisa

Esta pesquisa investigará como diferentes configurações de pipelines de Geração Aumentada de Recuperação (Retrieval-Augmented Generation - RAG) afetam simultaneamente a qualidade semântica das respostas, o consumo de tokens e o uso de infraestrutura computacional.

O estudo será delimitado à comparação experimental entre métodos de RAG, bancos de dados vetoriais, modelos de embedding e filtros de compressão contextual. O foco não será propor um novo modelo de linguagem, mas avaliar combinações práticas de componentes já existentes para identificar quais configurações oferecem melhor equilíbrio entre acurácia, custo e desempenho.

## 2. Problema de pesquisa

Sistemas RAG reduzem alucinações ao recuperar informações externas e incorporá-las ao contexto do modelo gerativo. Entretanto, a recuperação e o envio de grandes volumes de texto para o modelo aumentam o consumo de tokens, elevam a latência e demandam mais CPU e memória RAM.

Diante disso, a pesquisa buscará responder à seguinte pergunta:

> Como diferentes métodos de RAG, bancos vetoriais, modelos de embedding e filtros de compressão contextual influenciam a acurácia semântica, o consumo de tokens e o uso de hardware em pipelines de geração aumentada por recuperação?

## 3. Hipóteses

A pesquisa será guiada pelas seguintes hipóteses:

- Métodos de RAG que reduzem ou reorganizam o contexto antes da geração tendem a diminuir o consumo de tokens, mas podem afetar a qualidade semântica das respostas.
- Modelos de embedding mais robustos tendem a melhorar a recuperação de contexto relevante, porém podem aumentar custo, tempo de processamento ou dependência de serviços externos.
- Bancos vetoriais diferentes apresentam variações mensuráveis de latência, consumo de memória e desempenho de recuperação, especialmente em bases textuais de domínios distintos.
- Filtros de compressão contextual com limiares bem ajustados podem reduzir redundância e custo sem perda significativa de acurácia.
- Não haverá uma configuração universalmente superior; a melhor escolha dependerá da prioridade do cenário: menor custo, menor latência, menor uso de hardware ou maior acurácia.

## 4. Objetivo geral

Analisar experimentalmente o impacto de diferentes arquiteturas e métodos de RAG sobre a eficiência de hardware, o consumo de tokens e a acurácia semântica, a fim de propor diretrizes técnicas para otimização de pipelines de inteligência artificial aplicados a bases de dados complexas.

## 5. Objetivos específicos

- Mapear o funcionamento dos métodos Stuff, Refine, Map Reduce, Map Re-rank, Query Step-Down e Reciprocal RAG.
- Comparar ChromaDB, FAISS e Pinecone quanto à latência, uso de CPU, consumo de memória RAM e comportamento na recuperação vetorial.
- Avaliar os modelos de embedding BGE-small, Text-embedding-v3-large e Cohere Embed v3 em tarefas de recuperação semântica.
- Medir o impacto de filtros de compressão contextual baseados em similaridade e redundância.
- Executar uma varredura experimental controlada combinando métodos de RAG, bancos vetoriais, embeddings, filtros e bases de dados.
- Organizar os resultados em matrizes comparativas e propor recomendações práticas para escolha de componentes RAG.

## 6. Tipo de pesquisa

A pesquisa será aplicada, quantitativa e experimental.

Ela será aplicada porque busca produzir recomendações técnicas úteis para o desenvolvimento de sistemas RAG reais. Será quantitativa porque os resultados serão analisados por métricas numéricas, como latência, uso de CPU, uso de memória, quantidade de tokens e similaridade semântica. Será experimental porque as variáveis do pipeline serão controladas e combinadas em cenários de teste planejados.

## 7. Variáveis da pesquisa

### 7.1 Variáveis independentes

As variáveis independentes serão os componentes alterados durante os experimentos:

- Método de RAG: Stuff, Refine, Map Reduce, Map Re-rank, Query Step-Down e Reciprocal RAG.
- Banco de dados vetorial: ChromaDB, FAISS e Pinecone.
- Modelo de embedding: BGE-small, Text-embedding-v3-large e Cohere Embed v3.
- Filtro de compressão contextual: sem filtro, limiar baixo, limiar médio e limiar alto.
- Base de dados textual: SEC 10Q, Llama2 ArXiv e MedQA.

### 7.2 Variáveis dependentes

As variáveis dependentes serão os resultados medidos em cada execução:

- Acurácia semântica da resposta.
- Similaridade de cosseno entre resposta gerada, contexto recuperado e resposta esperada.
- Métricas RAGAS, quando aplicável.
- Tokens de entrada.
- Tokens de saída.
- Tokens totais por consulta.
- Tempo total de resposta.
- Uso médio e pico de CPU.
- Uso médio e pico de memória RAM.
- Taxa de recuperação de documentos relevantes.

### 7.3 Variáveis de controle

Para garantir comparação justa entre os cenários, deverão ser mantidos constantes:

- Conjunto de perguntas de teste por base de dados.
- Quantidade de documentos recuperados por consulta, quando aplicável.
- Tamanho dos chunks de texto.
- Estratégia de overlap entre chunks.
- Modelo gerativo utilizado para responder às perguntas.
- Temperatura e demais parâmetros de geração.
- Máquina ou ambiente computacional usado nos testes locais.
- Versão das bibliotecas utilizadas.

## 8. Desenho experimental

A pesquisa será executada em ciclos experimentais. Cada ciclo deverá testar uma combinação de componentes e registrar os resultados em arquivos estruturados.

O desenho experimental proposto é:

1. Selecionar uma base de dados.
2. Dividir os documentos em chunks com tamanho e overlap previamente definidos.
3. Gerar embeddings com um dos modelos selecionados.
4. Indexar os vetores em um dos bancos vetoriais.
5. Executar consultas padronizadas usando um método de RAG.
6. Aplicar ou não aplicar filtros de compressão contextual.
7. Gerar a resposta final com o mesmo modelo de linguagem.
8. Coletar métricas de hardware, tokens, latência e qualidade.
9. Repetir o processo para as demais combinações.

Como o número total de combinações pode ser alto, recomenda-se iniciar por um experimento piloto. O piloto deverá usar uma amostra reduzida de documentos, poucas perguntas e um subconjunto de configurações. Após validar o código, os logs e as métricas, a execução completa poderá ser realizada.

## 9. Bases de dados

### 9.1 SEC 10Q

Base composta por relatórios financeiros corporativos. Será utilizada para avaliar o comportamento do RAG em textos longos, técnicos, estruturados e com grande densidade numérica.

### 9.2 Llama2 ArXiv

Base composta por documentação científica relacionada ao modelo Llama 2. Será usada para avaliar recuperação em textos acadêmicos, com linguagem técnica e dependência de contexto conceitual.

### 9.3 MedQA

Base composta por questões da área médica. Será utilizada para avaliar recuperação e resposta em domínio sensível, no qual pequenas perdas semânticas podem comprometer a qualidade da resposta final.

## 10. Etapas da pesquisa

### Etapa 1: revisão bibliográfica

Levantar artigos e materiais técnicos sobre RAG, bancos vetoriais, modelos de embedding, compressão contextual, similaridade de cosseno, busca por vizinhos mais próximos e avaliação de respostas geradas por LLMs.

Resultados esperados:

- Referencial teórico consolidado.
- Lista organizada de referências em BibTeX.
- Definição dos conceitos que serão usados na análise.

### Etapa 2: definição do protocolo experimental

Definir exatamente como os experimentos serão executados, quais parâmetros serão fixos e quais serão alterados.

Decisões necessárias:

- Tamanho dos chunks.
- Percentual ou tamanho de overlap.
- Número de documentos recuperados por consulta.
- Quantidade de perguntas por base.
- Modelo gerativo usado.
- Parâmetros de geração.
- Formato dos arquivos de log.
- Métricas finais de comparação.

Resultados esperados:

- Protocolo experimental documentado.
- Estrutura de pastas definida.
- Planilha ou arquivo modelo para registro dos resultados.

### Etapa 3: preparação do ambiente

Configurar o ambiente em Python e instalar as bibliotecas necessárias para execução dos experimentos.

Ferramentas previstas:

- Python.
- LangChain ou LlamaIndex.
- ChromaDB.
- FAISS.
- Pinecone.
- Bibliotecas de embeddings.
- psutil para métricas de hardware.
- tiktoken ou ferramenta equivalente para contagem de tokens.
- pandas para tabulação.
- matplotlib, seaborn ou plotly para visualizações.
- RAGAS, caso seja utilizado para avaliação automática.

Resultados esperados:

- Ambiente de desenvolvimento funcional.
- Script simples validando carregamento de documentos, indexação e consulta.
- Registro das versões das bibliotecas.

### Etapa 4: construção do pipeline base

Implementar um pipeline RAG inicial, simples e reprodutível, para servir como referência.

O pipeline base deverá conter:

- Carregamento dos documentos.
- Pré-processamento.
- Quebra em chunks.
- Geração de embeddings.
- Indexação vetorial.
- Recuperação de contexto.
- Geração da resposta.
- Registro das métricas.

Resultados esperados:

- Protótipo funcional.
- Primeiras métricas coletadas.
- Identificação de ajustes necessários antes dos testes completos.

### Etapa 5: implementação dos métodos de RAG

Implementar ou configurar os métodos Stuff, Refine, Map Reduce, Map Re-rank, Query Step-Down e Reciprocal RAG.

Cada método deverá ser executado com a mesma entrada e com o mesmo conjunto de perguntas, para permitir comparação.

Resultados esperados:

- Métodos de RAG implementados ou configurados.
- Logs padronizados por método.
- Comparação preliminar de latência e tokens.

### Etapa 6: aplicação dos bancos vetoriais

Executar os testes usando ChromaDB, FAISS e Pinecone.

Critérios de observação:

- Tempo de indexação.
- Tempo de consulta.
- Consumo de memória.
- Facilidade de integração.
- Estabilidade durante múltiplas consultas.

Resultados esperados:

- Comparação quantitativa entre bancos.
- Identificação dos bancos mais adequados para cada cenário.

### Etapa 7: avaliação dos embeddings

Executar os experimentos com BGE-small, Text-embedding-v3-large e Cohere Embed v3.

Critérios de observação:

- Qualidade do contexto recuperado.
- Similaridade com respostas esperadas.
- Custo ou dependência de API.
- Tempo para gerar embeddings.
- Impacto no desempenho geral do pipeline.

Resultados esperados:

- Comparação entre embeddings leves e robustos.
- Identificação do impacto do embedding sobre acurácia e custo.

### Etapa 8: testes com compressão contextual

Aplicar filtros de compressão contextual com diferentes limiares de similaridade e redundância.

Configuração sugerida:

- Sem compressão.
- Compressão leve.
- Compressão média.
- Compressão agressiva.

Critérios de observação:

- Redução de tokens.
- Variação da acurácia.
- Mudança na latência.
- Perda de informações relevantes.

Resultados esperados:

- Medição do tradeoff entre economia de tokens e qualidade da resposta.
- Identificação de limiares recomendáveis.

### Etapa 9: análise estatística

Organizar os resultados coletados e aplicar análises comparativas.

Análises previstas:

- Média, mediana e desvio padrão de latência.
- Média e pico de uso de CPU.
- Média e pico de memória RAM.
- Tokens médios por consulta.
- Similaridade média por configuração.
- Comparação por domínio textual.
- Ranking das melhores combinações por objetivo.

Resultados esperados:

- Matrizes comparativas.
- Gráficos de barras, linhas e dispersão.
- Identificação de padrões e tradeoffs.

### Etapa 10: síntese das recomendações

Com base nos resultados, elaborar recomendações práticas.

As recomendações deverão responder a cenários como:

- Melhor configuração para menor consumo de tokens.
- Melhor configuração para maior acurácia.
- Melhor configuração para menor latência.
- Melhor configuração para menor uso de memória.
- Melhor configuração equilibrada.
- Configurações inadequadas ou instáveis.

Resultados esperados:

- Guia de decisão para escolha de componentes RAG.
- Discussão dos limites da pesquisa.
- Sugestões de trabalhos futuros.

## 11. Métricas de avaliação

### 11.1 Métricas de qualidade

- Similaridade de cosseno.
- Relevância do contexto recuperado.
- Fidelidade da resposta ao documento.
- Cobertura da resposta em relação à pergunta.
- Métricas RAGAS, se aplicável.

### 11.2 Métricas de custo

- Tokens de entrada.
- Tokens de saída.
- Tokens totais.
- Custo estimado por consulta, quando houver uso de API paga.

### 11.3 Métricas de infraestrutura

- Uso médio de CPU.
- Pico de CPU.
- Uso médio de memória RAM.
- Pico de memória RAM.
- Latência total.
- Tempo de indexação.
- Tempo de recuperação.
- Tempo de geração.

## 12. Organização dos resultados

Os resultados deverão ser organizados em arquivos separados por execução. Cada execução deverá registrar:

- Identificador do experimento.
- Base de dados utilizada.
- Método de RAG.
- Banco vetorial.
- Modelo de embedding.
- Configuração de compressão.
- Pergunta executada.
- Contextos recuperados.
- Resposta gerada.
- Resposta esperada, quando disponível.
- Métricas de qualidade.
- Métricas de tokens.
- Métricas de hardware.
- Data e horário da execução.

Recomenda-se salvar os dados brutos em formato CSV ou JSONL, preservando também notebooks ou scripts usados na análise.

## 13. Estrutura sugerida para a pasta da pesquisa

```text
Pesquisa/
  plano_de_pesquisa.md
  dados/
    brutos/
    processados/
  notebooks/
  scripts/
  resultados/
    metricas/
    graficos/
  referencias/
```

## 14. Cronograma operacional

### Maio de 2026

- Delimitação do tema.
- Definição da pergunta de pesquisa.
- Aprovação do plano de trabalho.
- Primeiras reuniões de orientação.

### Junho de 2026

- Revisão bibliográfica.
- Definição do protocolo experimental.
- Preparação do ambiente em Python.
- Construção do pipeline RAG base.
- Execução do experimento piloto.

### Julho de 2026

- Implementação dos métodos de RAG.
- Integração dos bancos vetoriais.
- Integração dos modelos de embedding.
- Execução dos testes principais.
- Coleta sistemática das métricas.

### Agosto de 2026

- Testes com compressão contextual.
- Consolidação dos resultados.
- Análise estatística.
- Geração dos gráficos.
- Escrita dos capítulos de metodologia e resultados.

### Setembro de 2026

- Discussão dos resultados.
- Escrita da conclusão.
- Revisão com o orientador.
- Ajustes finais do texto.
- Preparação da apresentação e defesa.

## 15. Riscos e estratégias de controle

### Alto número de combinações experimentais

Risco: o grid-search completo pode gerar muitas execuções e demandar tempo excessivo.

Controle: iniciar com experimento piloto, reduzir combinações pouco promissoras e priorizar as configurações mais relevantes.

### Dependência de APIs externas

Risco: custos, limites de requisição ou indisponibilidade de serviços como Pinecone, OpenAI ou Cohere.

Controle: registrar custos, limitar a amostra de testes e manter alternativas locais sempre que possível.

### Diferença entre ambientes de execução

Risco: métricas de hardware podem variar entre máquinas ou execuções.

Controle: executar os testes principais no mesmo ambiente, registrar configuração da máquina e repetir medições.

### Avaliação semântica limitada

Risco: a similaridade de cosseno pode não capturar toda a qualidade da resposta.

Controle: combinar similaridade de cosseno com métricas RAGAS e análise qualitativa de amostras selecionadas.

### Bases de dados muito grandes

Risco: bases extensas podem dificultar indexação, armazenamento e execução.

Controle: trabalhar com amostras representativas e documentar claramente os critérios de seleção.

## 16. Entregáveis da pesquisa

Ao final da execução, a pesquisa deverá produzir:

- Revisão bibliográfica sobre RAG, bancos vetoriais, embeddings e compressão contextual.
- Pipeline experimental reprodutível.
- Base de resultados com métricas coletadas.
- Gráficos comparativos.
- Análise dos tradeoffs entre acurácia, tokens e hardware.
- Guia de recomendações para configuração de pipelines RAG.
- Texto final do TCC.
- Apresentação para defesa.

## 17. Critério para conclusão da pesquisa

A pesquisa poderá ser considerada concluída quando:

- Todas as configurações selecionadas forem testadas ou justificadamente descartadas.
- As métricas principais forem coletadas e organizadas.
- Os resultados permitirem comparar claramente os métodos de RAG.
- For possível indicar quais combinações são mais adequadas para diferentes restrições.
- O texto final apresentar metodologia, resultados e discussão de forma reprodutível.

## 18. Resultado esperado da investigação

Espera-se que a pesquisa produza um panorama técnico sobre a eficiência de pipelines RAG. O resultado final deverá mostrar que a escolha dos componentes influencia diretamente o custo e a viabilidade de sistemas baseados em LLMs.

A principal contribuição será um conjunto de diretrizes práticas para orientar pesquisadores e engenheiros de software na escolha entre métodos de RAG, bancos vetoriais, embeddings e filtros de compressão, considerando diferentes prioridades de projeto.
