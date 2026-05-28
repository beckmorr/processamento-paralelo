#!/bin/bash

EXEC="./gauss_blur_mpi"

ITER_BLUR=500

RUNS=5

IMAGES=(
    "image_512x512.png"
    "image_1024x1024.png"
    "image_2048x2048.png"
    "image_4096x4096.png"
)

PROCESSES=(1 2 4 8 16)

echo "Iniciando Bateria de Testes MPI (Média de $RUNS execuções)"

for img in "${IMAGES[@]}"; do
    echo ">>> Avaliando: $img"
    
    for p in "${PROCESSES[@]}"; do
        soma_tempo=0
        
        for ((i=1; i<=RUNS; i++)); do
            saida=$(mpirun --use-hwthread-cpus --map-by :OVERSUBSCRIBE -np $p $EXEC $img $ITER_BLUR)
            
            tempo=$(echo "$saida" | grep "Tempo:" | awk -F'Tempo: ' '{print $2}' | awk '{print $1}')
            
            if [ -z "$tempo" ]; then
                tempo=0
                echo "Aviso: Falha ao ler o tempo na iteração $i com $p processos."
            fi

            soma_tempo=$(echo "$soma_tempo + $tempo" | bc -l)
        done
        
        media=$(echo "scale=4; $soma_tempo / $RUNS" | bc -l)
        
        tag_t="T${p}"
        
        echo "  - Tempo MPI ($p Processos) ($tag_t) : $media s"
    done
done

echo "Benchmark totalmente concluído."