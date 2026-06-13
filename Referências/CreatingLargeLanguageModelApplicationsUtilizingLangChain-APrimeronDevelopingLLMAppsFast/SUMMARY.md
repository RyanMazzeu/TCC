Criando Aplicações com Grandes Modelos de Linguagem Utilizando LangChain: Uma Introdução Rápida

Citação: Topsakal, O. & Akinci, T. C. (2023). Creating Large Language Model Applications Utilizing LangChain: A Primer on Developing LLM Apps Fast. 5th International Conference on Applied Engineering and Natural Sciences.
Resumo curto

O artigo apresenta uma visão conceitual do framework LangChain para a construção de aplicações com LLMs, detalhando os pipelines de Question Answering sobre documentos e definindo estruturalmente quatro métodos principais de injeção de contexto (Stuff, Map Reduce, Refine e Map Rerank).
O que este artigo descobriu

    Map Reduce viabiliza o processamento em paralelo para múltiplos documentos, mas trata as informações de forma independente (o que pode quebrar contextos globais) e exige um volume muito maior de chamadas (calls) à API.

    Refine produz respostas iterativas e mais construídas passando sobre múltiplos documentos, mas introduz um forte gargalo de tempo de execução porque as chamadas não podem ser paralelizadas.

    Map Rerank otimiza o uso do LLM ao fazer uma chamada para cada documento apenas para obter um escore de relevância, selecionando exclusivamente a resposta com a pontuação mais alta.

    Stuff continua sendo o método mais direto e simples, porém é rigidamente limitado pelo tamanho da janela de contexto do LLM.

Resumo estendido

Este trabalho atua como um guia introdutório (primer) sobre a arquitetura do LangChain para desenvolvimento de IA customizada. Os autores detalham o pipeline padrão de recuperação de informações (RAG), que engloba carregamento de dados, divisão de texto (chunking), geração de embeddings, armazenamento em vector stores e recuperação baseada na similaridade de cosseno.

A parte principal do artigo mapeia os métodos que podem ser utilizados quando os chunks recuperados não cabem em um único prompt. Os autores definem as arquiteturas de roteamento de dados para os LLMs, ilustrando graficamente como funcionam os fluxos do Stuff (inserção de todos os documentos recuperados de uma vez), Map Reduce (mapa em vários LLMs e sumarização final), Refine (construção sequencial da resposta) e Map Rerank (avaliação individual e seleção do top-1). O texto é essencialmente descritivo e arquitetural, não apresentando simulações ou coleta empírica de dados de performance.
Contribuições principais

    Definição estrutural e fluxogramas das quatro principais abordagens de processamento de documentos em pipelines RAG no LangChain (Stuff, Map Reduce, Refine, Map Rerank).

    Explicação teórica de como as representações quantitativas (embeddings) se baseiam no produto escalar (dot product/cosine similarity) para viabilizar a recuperação semântica em vector stores.

Metodologia

Não consta no texto enviado (trata-se de um artigo descritivo/tutorial conceitual, sem elaboração de setup experimental empírico).
Resultados

Não consta no texto enviado (o artigo não quantifica performance, uso de infraestrutura ou acurácia).
Limitações

    O estudo é puramente teórico e arquitetural. Faltam benchmarks e medições de trade-offs empíricos (como variação na similaridade da resposta, uso de tokens ou latência real) entre os métodos apresentados.

Implicações / aplicações

    Fornece um embasamento sólido de engenharia de software para justificar as escolhas de arquitetura de injeção de contexto ao desenhar pipelines baseados em LLM.

Por que isso importa para o meu TCC?

Nota de relevância: 5/10 — Oferece uma excelente base teórica para o capítulo de revisão, mas não fornece os dados empíricos comparativos que o núcleo do TCC exige.

Encaixe no plano (Ryan & Afonso — otimização RAG):

    Cobre diretamente a conceituação dos métodos RAG fundamentais (Stuff, Refine, Map-Reduce, Map Re-rank).

    Cita o embasamento teórico da similaridade de cosseno e de componentes como vector stores e embeddings.

O que dá para usar na redação:

    Podemos citar as descrições na fundamentação teórica para explicar o "sob o capô" do Stuff, Refine, Map Reduce e Map Rerank.

    Podemos usar os argumentos estruturais dos autores para embasar a discussão de trade-offs de tempo (ex.: apontar o gargalo de paralelismo do Refine e o volume de chamadas do Map Reduce).

Cuidados / o que não encaixa:

    O artigo não apresenta números (tokens, runtime, CPU/Memória, similaridade). Nenhuma afirmação quantitativa pode ser extraída daqui.

    Não menciona abordagens mais complexas como Reciprocal RAG, Compressão Contextual ou Query Step-Down.

Em relação a Şakar & Emekci (grid-search RAG):

    Complementa conceitualmente: Enquanto Şakar & Emekci fornecem os números pesados e a prova empírica dos trade-offs de custo/acurácia, este artigo de Topsakal & Akinci descreve o fluxograma básico de como essas abordagens funcionam em nível de engenharia dentro de um framework como o LangChain.

Palavras-chave

LangChain, RAG, LLM, Stuff, Map Reduce, Refine, Map Rerank, Vector Stores, Embeddings.