Augmenting LLMs with Knowledge: A survey on hallucination prevention

Citação: Andriopoulos, K., & Pouwelse, J. (2023). Augmenting LLMs with Knowledge: A survey on hallucination prevention. arXiv preprint arXiv:2309.16459.
Resumo curto

O artigo é uma revisão de literatura que mapeia abordagens para mitigar alucinações em LLMs integrando fontes de conhecimento externo. Os autores exploram como a memória não paramétrica (como vector databases e grafos de conhecimento) é unida a modelos generativos (como no RAG e REALM) para aumentar a precisão sem a necessidade de escalar os parâmetros do modelo.
O que este artigo descobriu

    O uso de memória não paramétrica (vector stores) permite que LLMs tenham acesso a conhecimento factual externo, reduzindo alucinações e o tamanho necessário do modelo.

    Modelos como RAG realizam a fusão do conhecimento de duas formas: pós-recuperação (marginalização das probabilidades dos documentos gerados, no RAG-token) ou sequencialmente (RAG-sequence).

    Problemas de "cold-start" em arquiteturas RAG com treinamento conjunto (como REALM) ocorrem quando a recuperação inicial traz documentos irrelevantes, forçando o gerador a ignorar o contexto; isso exige técnicas de inicialização específicas, como a Inverse Cloze Task.

    Bancos de dados vetoriais (utilizando FAISS e busca de Maximum Inner Product Search - MIPS) e Knowledge Graphs (utilizando Graph Convolutional Networks - GCNs) são as duas vertentes principais para memória externa.

Resumo estendido

O artigo apresenta um mapeamento teórico das técnicas de aumento de LLMs com conhecimento externo para contornar limitações de contexto e alucinação próprias da modelagem estatística tradicional. Os autores classificam as bases de conhecimento em não estruturadas (corpora de texto indexados em bancos de dados vetoriais como o FAISS) e estruturadas (Triplestores e Grafos de Conhecimento processados via GCNs). A revisão detalha a arquitetura original do RAG (Retrieval-Augmented Generation), explicando o uso de codificadores densos (DPR com modelos BERT) para recuperação e a forma como as probabilidades são agregadas no decoder usando algoritmos como o Beam Search. Em seguida, abordam o modelo REALM, que inova ao treinar o recuperador e o gerador de forma conjunta, superando desafios de inicialização aleatória dos encoders.
Contribuições principais

    Fornece uma taxonomia clara para modelos de linguagem aumentados.

    Sintetiza o funcionamento mecânico e matemático de arquiteturas fundamentais como RAG e REALM.

    Compara a incorporação de conhecimento não estruturado (vetores densos) com conhecimento estruturado (grafos).

Metodologia

    Revisão de literatura (survey) focada nos algoritmos, estruturas e técnicas matemáticas do aumento de LLMs. Não consta experimentação ou implementação própria de cenários no texto enviado.

Resultados

    Não consta no texto enviado (trata-se de uma revisão teórica, sem coleta primária de métricas de desempenho numéricas).

Limitações

    Não apresenta experimentação empírica comparando tempo, uso de CPU, RAM, similaridade ou consumo de tokens.

    Foca pesadamente na base arquitetural dos primeiros modelos (BART e BERT), sem avaliar otimizações de infraestrutura modernas do pipeline.

Implicações / aplicações

    O conhecimento sintético serve de alicerce teórico para a construção e treinamento de sistemas que requerem memória factual atualizada além dos pesos pré-treinados, guiando o uso de índices densos.

Por que isso importa para o meu TCC?

Nota de relevância: 4/10 — Uma boa fonte para a fundamentação teórica de RAG e MIPS, mas não contribui para a avaliação empírica de eficiência (tempo/RAM/tokens) do seu pipeline.

Encaixe no plano (Ryan & Afonso — otimização RAG):

    Fundamentação teórica útil (Eixo 6): Cobre os conceitos base da arquitetura RAG e explica o processo de Maximum Inner Product Search (MIPS).

    Componentes do pipeline (Eixo 2): Define teoricamente o papel e o funcionamento do FAISS e dos bancos de dados vetoriais.

O que dá para usar na redação:

    Podemos citar na revisão de literatura para definir formalmente a divisão entre memória paramétrica e memória não paramétrica.

    Útil para basear a seção de métodos ao explicar como a biblioteca FAISS e a similaridade vetorial operam nos bastidores antes de apresentar os experimentos comparativos.

Cuidados / o que não encaixa:

    Não usar como base para discutir o custo de infraestrutura ou trade-offs de desempenho, pois o artigo carece de números sobre tokens, runtime ou hardware.

    Foca muito em modelos de embedding mais antigos (BERT) e em Graph Neural Networks, o que foge do escopo restrito aos métodos RAG em texto puro do plano.

Em relação a Şakar & Emekci (grid-search RAG):

    Complementa teoricamente. Enquanto Şakar & Emekci fazem a análise empírica focada em hardware e acurácia de componentes atuais, este artigo fornece a base teórica e as equações originais (como a do RAG-token) que sustentam os frameworks otimizados no grid-search.

Palavras-chave

Retrieval-Augmented Generation (RAG), Vector Database, FAISS, Maximum Inner Product Search (MIPS), REALM, Memória Não Paramétrica, Alucinação.