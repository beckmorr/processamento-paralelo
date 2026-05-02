# BENCHMARK DE SUAVIZAÇÃO/DESFOQUE GAUSSIANO ITERATIVO: IMPLEMENTAÇÃO SEQUENCIAL E PARALELA EM MEMÓRIA COMPARTILHADA COM OPENMP

**Autores:** Anderson Morbeck Pires, Marco Túlio Macêdo Oliveira dos Santos  
**Disciplina:** Processamento Paralelo (DEC107)  
**Orientador:** Esbel Tomas Valero Orellana  
**Data:** 2026

---

## BENCHMARK DE SUAVIZAÇÃO/DESFOQUE GAUSSIANO ITERATIVO: IMPLEMENTAÇÃO SEQUENCIAL E PARALELA EM MEMÓRIA COMPARTILHADA COM OPENMP

### 1. INTRODUÇÃO
Este projeto avalia a aplicação de conceitos fundamentais de processamento paralelo em arquiteturas de memória compartilhada utilizando a biblioteca OpenMP. O benchmark baseia-se no algoritmo de desfoque Gaussiano iterativo, utilizando máscaras de $3\times3$ e $5\times5$ para redução de ruído em imagens bidimensionais[cite: 1]. A relevância do estudo reside na quantificação de ganhos de desempenho via métricas de Speedup e Eficiência em ambientes HPC[cite: 1].

### 2. METODOLOGIA
A metodologia foca na transição do paradigma serial para o paralelo em linguagem C[cite: 1].

*   **2.1. Implementação Sequencial:** Estruturada com três loops aninhados (iterações, altura e largura)[cite: 1]. Utilizou-se a biblioteca `stb_image` para gestão de matrizes e técnica de *clamping* para bordas[cite: 1]. Compilado com `-O3` e `-march=native` para otimizações SIMD[cite: 1].
*   **2.2. Implementação Paralela:** 
    *   **Loops:** Diretiva `#pragma omp for` no loop de linhas[cite: 1].
    *   **Escalonamento:** `schedule(static)` devido à carga uniforme[cite: 1].
    *   **Gestão de Threads:** Região paralela instanciada fora do loop de iterações para evitar *overhead* recorrente[cite: 1].
    *   **Sincronização:** `#pragma omp single` para troca de ponteiros[cite: 1].
*   **2.3. Ambiente:** Intel Core i5-13450HX (6 P-cores, 4 E-cores, 16 threads lógicos), Pop!_OS 24.04 LTS (Kernel 6.18)[cite: 1].
*   **2.5. Métricas:** Tempo de Execução ($T_n$), Speedup ($S_p = T_n / T_{n,p}$) e Eficiência ($E_p = S_p / p$)[cite: 1].

### 3. RESULTADOS

#### 3.1. Teste de Corretude
A integridade foi validada comparando as matrizes resultantes com a referência do docente; os resultados foram numericamente idênticos em todos os cenários[cite: 1].

#### 3.3. Cálculo dos Tempos de Execução

**Kernel $3\times3$:**
| Tamanho | $T_0$ (Seq) | $T_1$ | $T_2$ | $T_4$ | $T_8$ | $T_{16}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 512x512 | 0,0823 | 0,1772 | 0,0956 | 0,0524 | 0,0507 | 0,0413 |
| 1024x1024 | 0,3156 | 0,7096 | 0,3731 | 0,2174 | 0,1815 | 0,1692 |
| 2048x2048 | 1,2801 | 2,8397 | 1,4626 | 0,8073 | 0,6898 | 0,4785 |
| 4096x4096 | 5,2707 | 11,4398 | 6,0365 | 3,2713 | 2,3913 | 1,8818 |

**Kernel $5\times5$:**
| Tamanho | $T_0$ (Seq) | $T_1$ | $T_2$ | $T_4$ | $T_8$ | $T_{16}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 512x512 | 0,2386 | 0,2624 | 0,1359 | 0,0752 | 0,0692 | 0,0424 |
| 1024x1024 | 0,9978 | 1,0845 | 0,5774 | 0,3338 | 0,2946 | 0,2123 |
| 2048x2048 | 3,9290 | 4,3899 | 2,2728 | 1,2440 | 1,1217 | 0,7414 |
| 4096x4096 | 15,8536 | 18,1835 | 9,1936 | 5,1269 | 3,9008 | 2,8988 |

#### 3.5. Granularidade
Estimada via $G = T_{cp} / T_{cm}$, onde $T_{cm} \approx T_1 - T_0$[cite: 1].

| Métrica | Kernel $3\times3$ | Kernel $5\times5$ |
| :--- | :--- | :--- |
| Tempo de Computação ($T_{cp}$) | 5,2707 s | 15,8536 s |
| Tempo de Gerenciamento ($T_{cm}$) | 6,1691 s | 2,3299 s |
| **Granularidade ($G$)** | **0,85** | **6,80** |

*   **Kernel $3\times3$ (Granularidade Fina):** $G < 1$ indica que o *overhead* do OpenMP domina a execução[cite: 1].
*   **Kernel $5\times5$ (Granularidade Grossa):** $G \approx 6,80$ indica que o processador gasta mais tempo em computação útil, permitindo melhor escalabilidade[cite: 1].

### 4. DISCUSSÃO
*   **4.1. Intensidade Aritmética:** O Kernel $5\times5$ obteve $S_p$ de 5,46x contra 2,80x do $3\times3$, pois sua maior complexidade (25 op/pixel) amortece melhor os custos de sincronização[cite: 1].
*   **4.2. Arquitetura Híbrida:** A eficiência máxima ocorre entre 1 e 4 threads (P-cores). O decaimento após 8 threads reflete a entrada dos E-cores e saturação de memória (*memory-bound*)[cite: 1].
*   **4.3. Sincronização:** As 100 barreiras globais limitam a performance em resoluções baixas (512x512)[cite: 1].

### 5. CONCLUSÃO
A superioridade do kernel $5\times5$ deve-se à sua granularidade mais grossa, mascarando os custos de sincronização e latência de memória[cite: 1]. A arquitetura híbrida do i5-13450HX impõe retornos marginais após o uso dos P-cores físicos[cite: 1]. O estudo consolidou conhecimentos vitais para futuras implementações em MPI e CUDA[cite: 1].

### 6. REFERÊNCIAS
1. OPENMP ARB. OpenMP API: User's Guide[cite: 1].
2. ORELLANA, E. T. V. RepoDEC107-PP-2026_1. GitHub, 2026[cite: 1].
3. ORELLANA, E. T. V. Materiais de aula DEC107 (Aulas 03, 05, 06, 07). UESC, 2026[cite: 1].
