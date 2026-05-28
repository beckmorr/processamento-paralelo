#!/bin/bash

export LC_NUMERIC="C"

EXEC="./gauss_blur_seq"

ITER_BLUR=500

RUNS=5

IMAGES=(
    "image_512x512.png"
    "image_1024x1024.png"
    "image_2048x2048.png"
    "image_4096x4096.png"
)

echo "Iniciando Bateria de Testes Sequenciais (Media de $RUNS execucoes)"
echo "Iteracoes do filtro: $ITER_BLUR"

for img in "${IMAGES[@]}"; do
    echo ">>> Avaliando: $img"
    soma_tempo=0
    
    for ((i=1; i<=RUNS; i++)); do
        saida=$($EXEC $img $ITER_BLUR)
        
        tempo=$(echo "$saida" | grep "Tempo:" | awk -F'Tempo: ' '{print $2}' | awk '{print $1}')
        
        if [ -z "$tempo" ]; then
            tempo=0
            echo "Aviso: Falha ao ler o tempo na iteracao $i."
        fi

        soma_tempo=$(echo "$soma_tempo + $tempo" | bc -l)
    done
    
    media=$(echo "scale=4; $soma_tempo / $RUNS" | bc -l)
    
    media_formatada=$(printf "%.4f" "$media")
    
    echo "  - Tempo Sequencial Medio : $media_formatada s"
done

echo "Benchmark sequencial totalmente concluido."