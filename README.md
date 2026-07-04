# Benchmark de Suavização Gaussiana Iterativa: Sequencial, OpenMP, MPI e CUDA

Este repositório contém o desenvolvimento e a análise de desempenho do algoritmo de desfoque Gaussiano iterativo, aplicado em imagens bidimensionais. O objetivo principal é transformar um algoritmo sequencial intensivo em uma solução de alta performance, explorando diferentes arquiteturas e modelos de programação paralela.

---

## Escopo do Projeto

O projeto consiste na implementação do filtro de convolução utilizando máscaras de $3\times3$ e $5\times5$. A carga de trabalho é definida por 100 iterações do filtro, o que permite medir a escalabilidade e os gargalos de comunicação de cada abordagem.

### Paradigmas a serem implementados

- [x] **Sequencial (Baseline):** Implementação de referência em C, otimizada com flags de compilação (`-O3`, `-march=native`) para servir de base comparativa.
- [x] **OpenMP (Memória Compartilhada):** Focado na exploração de múltiplos núcleos (P-cores e E-cores) em um único processador.
- [x] **MPI (Memória Distribuída):** Focado na divisão de domínio e troca de mensagens entre processos, permitindo a execução em clusters de computadores.
- [x] **CUDA (GPU Computing):** Aceleração massivamente paralela para processamento de pixels em larga escala utilizando unidades de processamento gráfico.

---

## Detalhes Técnicos

- **Processamento de Imagem:** Utiliza a biblioteca `stb_image` para leitura e escrita de arquivos de imagem de forma eficiente.
- **Tratamento de Bordas:** Aplicação da técnica de *clamping* para garantir a consistência dos cálculos nas extremidades da imagem.
- **Sincronização:** Implementação de barreiras de sincronização e troca de ponteiros para garantir a integridade dos dados entre as iterações.

---

## Métricas de Desempenho

O desempenho de cada versão é avaliado criteriosamente através de quatro métricas fundamentais:

- **Tempo de Execução ($T_n$):** Tempo total transcorrido para o processamento de todas as iterações.
- **Speedup ($S_p$):** Razão entre o tempo da versão sequencial e o tempo da versão paralela ($S_p = T_0 / T_{n,p}$).
- **Eficiência ($E_p$):** Grau de aproveitamento dos recursos de hardware disponíveis ($E_p = S_p / p$).
- **Granularidade ($G$):** Relação entre o tempo de computação útil e o tempo gasto com gerenciamento e comunicação ($G = T_{cp} / T_{cm}$).

---

## Ambiente de Desenvolvimento

As métricas e testes foram conduzidos no seguinte ambiente técnico:

- **Processador:** Intel Core i5-13450HX (10 núcleos físicos: 6 P-cores e 4 E-cores).
- **Threads:** 16 threads lógicos.
- **Arquitetura:** Híbrida (Performance e Eficiência).
- **Sistema Operacional:** Manjaro Linux x86_64.
- **Compilador:** GCC 13.2.
- **Bibliotecas:** OpenMP 4.5 e stb_image.

---

## Nota de Transparência sobre o Uso de IA

Declaramos que este projeto contou com o auxílio da ferramenta de IA **Google Gemini** exclusivamente para as tarefas de:

- **Refatoração de código:** Auxílio na adequação da implementação sequencial para espelhar a lógica de *padding* das versões paralelas, visando a paridade experimental necessária para a precisão dos benchmarks.
- **Orquestração de testes:** Apoio na estruturação de *scripts* em Bash (`.sh`) para automação da bateria de testes, garantindo a execução repetível das 5 rodadas de benchmark por cenário.
- **Estruturação de documentação:** Apoio na organização lógica dos argumentos no relatório técnico e revisão gramatical/estilística.

Como autores, atestamos que revisamos, testamos e validamos criticamente todo o conteúdo gerado, assumindo total e exclusiva responsabilidade pela correção lógica do código, precisão dos relatórios de desempenho e integridade acadêmica do material entregue.

- **Autores:** Anderson Morbeck Pires, Marco Túlio Macêdo Oliveira dos Santos.
- **Data:** 28 de Maio de 2026.
