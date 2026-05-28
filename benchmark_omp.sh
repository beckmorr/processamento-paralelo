#!/bin/bash

export LC_NUMERIC="C"

EXEC="./gauss_blur_omp"

ITER_BLUR=500

RUNS=5

IMAGES=(
    "image_512x512.png"
    "image_1024x1024.png"
    "image_2048x2048.png"
    "image_4096x4096.png"
)

THREADS=(1 2 4 8 16)

echo "Iniciando Bateria de Testes OpenMP (Média de $RUNS execuções)"
echo "Iterações do filtro: $ITER_BLUR"

for img in "${IMAGES[@]}"; do
    echo ">>> Avaliando: $img"
    for t in "${THREADS[@]}"; do
        soma_tempo=0
        for ((i=1; i<=RUNS; i++)); do
            OMP_NUM_THREADS=$t $EXEC $img $ITER_BLUR > /dev/null 2>&1
            saida=$(OMP_NUM_THREADS=$t $EXEC $img $ITER_BLUR)
            tempo=$(echo "$saida" | grep "Tempo:" | awk -F'Tempo: ' '{print $2}' | awk '{print $1}')
            if [ -z "$tempo" ]; then
                tempo=0
                echo "Aviso: Falha ao ler o tempo na iteração $i com $t threads."
            fi
            soma_tempo=$(echo "$soma_tempo + $tempo" | bc -l)
        done
        media=$(echo "scale=4; $soma_tempo / $RUNS" | bc -l)
        media_formatada=$(printf "%.4f" "$media")
        echo "  - Tempo OpenMP ($t Threads) : $media_formatada s"
    done
done

echo "Benchmark OpenMP totalmente concluído."