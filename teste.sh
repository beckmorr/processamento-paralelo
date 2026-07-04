#!/bin/bash

# Configurações do ambiente
SRC_FILE="gauss_blur_cuda.cu"
EXE_FILE="executavel_cuda"
LOG_FILE="resultados_cuda.md"
ITERATIONS=200

# Lista de resoluções de imagens de teste (certifique-se de que esses arquivos existem no diretório)
IMAGES=("image_512x512.png" "image_1024x1024.png" "image_2048x2048.png" "image_4096x4096.png")
KERNELS=(3 5)

# Inicializa o arquivo de log com o cabeçalho markdown
echo "# Resultados do Benchmark CUDA (Etapa 3)" > "$LOG_FILE"
echo "Data de execução: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

for K in "${KERNELS[@]}"; do
    echo "================================================"
    echo "Compilando e Otimizando para Kernel ${K}x${K}..."
    echo "================================================"
    
    # Compila definindo dinamicamente o K_SIZE para o pré-processador do C
    nvcc -O3 -arch=sm_86 -DK_SIZE=$K "$SRC_FILE" -o "${EXE_FILE}_k${K}" -lm
    
    if [ $? -ne 0 ]; then
        echo "Erro na compilação do Kernel ${K}x${K}. Abortando."
        exit 1
    fi
    
    echo "## Tabela de Medição: Kernel ${K}x${K}" >> "$LOG_FILE"
    echo "| Tamanho da Imagem | Tempo Total (s) | Tempo de Kernel Puro (s) | Tempo de Transferência (s) |" >> "$LOG_FILE"
    echo "| :--- | :---: | :---: | :---: |" >> "$LOG_FILE"
    
    for IMG in "${IMAGES[@]}"; do
        if [ ! -f "$IMG" ]; then
            echo "Aviso: Imagem $IMG não encontrada no diretório. Pulando..."
            continue
        fi
        
        echo "Executando bateria para $IMG..."
        
        # Executa o binário correspondente à matriz atual
        OUTPUT=$("./${EXE_FILE}_k${K}" "$IMG" "$ITERATIONS")
        
        # Extração dos tempos com base na saída textual do executável
        T_TOTAL=$(echo "$OUTPUT" | grep -i "total" | awk '{print $NF}' | tr -d 's')
        T_KERNEL=$(echo "$OUTPUT" | grep -i "kernel" | awk '{print $NF}' | tr -d 's')
        T_TRANS=$(echo "$OUTPUT" | grep -i "transferencia" | awk '{print $NF}' | tr -d 's')
        
        # Fallback caso os prints do main tenham formatos diferentes
        if [ -z "$T_TOTAL" ]; then T_TOTAL="Verificar"; fi
        if [ -z "$T_KERNEL" ]; then T_KERNEL="Verificar"; fi
        if [ -z "$T_TRANS" ]; then T_TRANS="Verificar"; fi
        
        echo "| $IMG | $T_TOTAL s | $T_KERNEL s | $T_TRANS s |" >> "$LOG_FILE"
    done
    echo "" >> "$LOG_FILE"
    
    # Limpa o binário temporário deste tamanho
    rm -f "${EXE_FILE}_k${K}"
done

echo "=== Bateria de testes concluída com sucesso! ==="
echo "Os resultados formatados foram salvos em: $LOG_FILE"
echo "------------------------------------------------"
cat "$LOG_FILE"