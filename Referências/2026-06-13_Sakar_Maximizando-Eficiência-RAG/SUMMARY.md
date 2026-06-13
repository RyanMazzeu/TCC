Maximizando a Eficiência do RAG: Uma Análise Comparativa de Métodos

Citação: Şakar, T., & Emekci, H. (2025). Maximizing RAG efficiency: A comparative analysis of RAG methods. Natural Language Processing, 31, pp. 1-25. DOI: 10.1017/nlp.2024.53  
Resumo curto

O artigo otimiza processos RAG ao executar uma exaustiva busca em grade (grid-search) com 23.625 iterações, comparando diferentes componentes e métodos. Os autores concluem que abordagens de re-ranking iterativo melhoram a similaridade semântica, enquanto a compressão contextual é crítica para poupar tokens e hardware, exigindo ajustes cuidadosos nos limiares para balancear precisão versus custos operacionais.  
O que este artigo descobriu

    O método Reciprocal RAG entrega a maior acurácia (mediana de similaridade global de 91%, atingindo 0,971 no dataset 10Q) devido ao seu mecanismo intrínseco de filtragem das respostas mais pertinentes geradas por queries alternativas.  

    O método Stuff é insuperável em eficiência temporal e orçamentária: foi 38,9% mais rápido que os líderes e gastou 70,5% menos tokens, com um sacrifício de acurácia de apenas 7,2% em relação ao topo do ranking.  

    O modelo open-source Bge-en-small foi surpreendentemente eficiente, alcançando 94% de similaridade mediana e ultrapassando as ofertas carro-chefe da OpenAI e Cohere.  

    Na avaliação dos LLMs acoplados à arquitetura, o Command-R (Cohere) bateu o GPT-3.5 e o GPT-4o-Mini no quesito similaridade (83%).  

    O uso sistemático de compressão contextual comete cortes profundos em tokens (ex.: queda de 18,6% de uso no método Refine) e tempo de execução, mas deteriora uniformemente os scores de similaridade de todos os métodos se o limiar não for ajustado pelo perfil do dado empírico.  

Resumo estendido

Este trabalho aborda o desafio inerente de calibrar pipelines de Retrieval-Augmented Generation (RAG) visando equilibrar alta similaridade de respostas contra limitações de tokens, gargalos de tempo de execução e consumo exacerbado de CPU e memória. A metodologia implementada utilizou uma massiva busca em grade composta por 23.625 configurações diferentes que processaram aproximadamente 42,12 milhões de tokens de embedding e mais de 18 milhões de tokens de entrada e saída ao longo de 112 horas ininterruptas de processamento computacional. Os pesquisadores combinaram três datasets distintos (o financeiro 10Q, o técnico científico Llama2 Paper e o de exames médicos MedQA) iterando-os através de modelos de LLM contemporâneos, as principais arquiteturas abertas de vectorstores e múltiplas variações lógicas do RAG (do simples injetor de texto Stuff aos geradores multi-etapas baseados em re-ranking como Map Re-rank e Reciprocal).  

A principal verificação metodológica recaiu sobre como a complexidade extra adicionada para mitigar o problema de ambiguidade nas perguntas do usuário impacta o pipeline de ponta a ponta. Constatou-se que métodos de alta complexidade elevam drasticamente a acurácia global e a qualidade do texto extraído nos domínios financeiro e técnico, mas exigem múltiplas requisições encadeadas. Por contrapartida, cenários que inseriram limitadores diretos através de Contextual Compression provaram a hipótese de que o excesso de filtros de similaridade pode varrer para fora da memória semântica dados cruciais dispersos nos documentos, o que afunda significativamente as pontuações do cosseno.  
Contribuições principais

    Execução de um grid-search extenso e documentado (23.625 iterações) mapeando todas as correlações essenciais num pipeline de RAG.  

    Quantificação analítica da correlação de forças entre similaridade (resposta esperada), limite de processamento de contexto (tokens), gargalo de rede/processamento (runtime) e custos em nuvem (CPU/RAM).  

    Comprovação do trade-off gerado pela introdução da Compressão Contextual, fornecendo recomendações numéricas de tolerância por domínio.  

Metodologia

    Pré-processamento: Configuração via Recursive Character Split com chunks definidos em 1000 caracteres e intersecção (overlap) de 100; codificação de tokens guiada via tiktoken.  

    Componentes do Grid Search: 3 datasets (Docugami 10Q, Llama2 Paper, MedQA); 3 Vector Databases (ChromaDB, FAISS, Pinecone); 3 Embeddings (text-embedding-v3-large, bge-en-small, cohere-en-v3); 3 LLMs (GPT-3.5, GPT-4o-mini, Command-R).  

    RAG: Estudo comparativo cruzando os métodos Stuff, Refine, Map Reduce, Map Re-rank, Query Step-Down e Reciprocal RAG.  

    Métricas Instrumentadas: Similaridade de cosseno (LLM gerado vs. Referência do dataset via GPT-4), Runtime (segundos), uso de CPU e memória (%) e contabilização de Tokens (Ti​=(LRi​×4)/3).  

Resultados

    Acurácia: O método Reciprocal apresentou o maior pico global nas respostas (97,1% no 10Q; 93,0% no Llama2), falhando mais acentuadamente apenas nas questões complexas médicas do MedQA (67,3%).  

    Latência/Performance: Stuff atingiu o piso de tempo computacional (mediana de 14,29 segundos). Map Re-rank obteve resultados competentes (16,49 segundos), contrastando com os lentos 34,33 segundos apurados no Step-Down.  

    Vectorstore vs Computação: O banco ChromaDB mostrou-se formidável na economia de recursos operacionais em combinação com o modelo text-embedding-v3-large: 18,34s de tempo de resposta aliado a meros 1,50% de CPU.  

    Tokens: A lógica centralizada e crua do Stuff engoliu 937 tokens frente aos brutais 5527 consumidos pelo Query Step-Down (dataset financeiro 10Q).  

Limitações

    A adoção de limiares punitivos de compressão contextual (ex.: 80% e 90%) bloqueia o fluxo de dados valiosos e enterra a utilidade da resposta.  

    Todos os métodos de RAG experimentaram declínio agudo de funcionalidade quando testados com o jargão do banco de exames médicos MedQA (onde até métodos robustos bateram apenas parcos 14,8% em similaridade como no Map Re-rank).  

Implicações / aplicações

    Engenheiros arquitetando assistentes de alta escalabilidade técnica devem emparelhar algoritmos mais diretos (Stuff) com modelos BAAI (Bge-en-small) e ChromaDB como vetorial para evitar paralisar infraestruturas sob alta carga simultânea.  

    Projetos financeiros (como análises de balanços) que exigem máxima proximidade com o texto-fonte devem assumir o custo financeiro computacional da rotação com algoritmos Reciprocal RAG para contornar ambiguidade natural de linguagem.  

Por que isso importa para o meu TCC?

Nota de relevância: 10/10 — Trata-se do artigo âncora e da principal fundação teórica que estrutura o grid de análise comparativa que este TCC propõe replicar ou analisar.

Encaixe no plano (Ryan & Afonso — otimização RAG):

    Executa e analisa quantitativamente os seis métodos mapeados no projeto: Stuff, Refine, Map-Reduce, Map Re-rank, Step-Down, Reciprocal, somado aos filtros de compressão contextual.

    Cobre todo o escopo de medição de eficiência proposta pela monografia (similaridade, tempo, tokens, uso de hardware).

    Descreve as infraestruturas que serão alvo de debate na pesquisa (ChromaDB vs FAISS vs Pinecone; modelos grandes proprietários vs open-source).

O que dá para usar na redação:

    Empregar a citação da equação de cálculo estimativo de consumo dos tokens em iterações de resposta (Ti​=(LRi​×4)/3).

    Justificar que modelos enxutos bge-en-small não significam invariavelmente perda de qualidade quando expostos a métodos de recuperação mais lógicos, utilizando a métrica mediana dele de 94% do artigo contra as contrapartes de alto peso da OpenAI.

    Validar textualmente a premissa de que não existe a "arquitetura mágica universal", já que todos os trade-offs foram expostos e a solução de compressão de contexto pune precisão na busca para salvar tokens.

Cuidados / o que não encaixa:

    A dimensão da busca da pesquisa atingiu métricas massivas incompatíveis com laboratório doméstico (~112 horas acumuladas e +42 milhões de tokens calculados no embedding). O TCC deve manter uma escopo enxuto ou valer-se desta literatura como benchmark norteador.

Em relação a Şakar & Emekci (grid-search RAG):

    É o próprio artigo-âncora — O texto fornece inteiramente os achados e a justificativa para basear as escolhas dos ensaios propostos na pesquisa.

Palavras-chave

Retrieval-Augmented Generation, Large Language Models, Vector Databases, Contextual Compression, Embedding Models, Token Budgeting