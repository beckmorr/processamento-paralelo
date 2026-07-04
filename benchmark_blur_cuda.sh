#!/usr/bin/env bash
#
# benchmark_blur_cuda.sh
#
# Compila gauss_blur_cuda.cu e executa o benchmark para 4 imagens diferentes,
# variando o tamanho do bloco de threads CUDA (16x16, 32x32, etc).
# Executa cada configuracao multiplas vezes para calcular metricas estatisticas.
#
# Uso:
#   ./benchmark_blur_cuda.sh
#
set -euo pipefail

# ----------------------------- Configurações -----------------------------

SRC="gauss_blur_cuda.cu"
BIN="./gauss_blur_cuda"
ITERATIONS=500          # numero de iteracoes do blur dentro do binario
RUNS_PER_CONFIG=5       # numero de execucoes independentes para a media
IMG_DIR="."             # diretorio onde estao as imagens
OUT_CSV="resultados_benchmark.csv"
LOG_DIR="logs_benchmark"

IMAGES=(
    "image_512x512.png"
    "image_1024x1024.png"
    "image_2048x2048.png"
    "image_4096x4096.png"
)

BLOCK_SIZES=(
    "1 1"
    "4 4"
    "8 8"
    "16 16"
    "24 24"
    "32 32"
)

# ----------------------------- Funções -----------------------------

log() { echo -e "[$(date '+%H:%M:%S')] $*"; }

check_requirements() {
    if ! command -v nvcc &> /dev/null; then
        echo "ERRO: nvcc nao encontrado no PATH. Instale o CUDA Toolkit." >&2
        exit 1
    fi

    if ! command -v awk &> /dev/null; then
        echo "ERRO: awk nao encontrado. Necessario para os calculos estatisticos." >&2
        exit 1
    fi

    if [[ ! -f "$SRC" ]]; then
        echo "ERRO: arquivo fonte '$SRC' nao encontrado no diretorio atual." >&2
        exit 1
    fi

    for hdr in stb_image.h stb_image_write.h; do
        if [[ ! -f "$hdr" ]]; then
            echo "ERRO: '$hdr' nao encontrado no diretorio atual (necessario para compilar)." >&2
            exit 1
        fi
    done

    for img in "${IMAGES[@]}"; do
        if [[ ! -f "$IMG_DIR/$img" ]]; then
            echo "AVISO: imagem '$IMG_DIR/$img' nao encontrada. Ela sera pulada." >&2
        fi
    done
}

compile() {
    log "Compilando $SRC com nvcc..."
    nvcc -O3 -arch=sm_86 "$SRC" -o "$BIN" -lm
    log "Compilacao concluida: $BIN"
}

extract_dims_from_name() {
    local fname="$1"
    local base
    base=$(basename "$fname" .png)
    local dims
    dims=$(echo "$base" | grep -oE '[0-9]+x[0-9]+')
    echo "${dims%x*} ${dims#*x}"
}

run_benchmarks() {
    mkdir -p "$LOG_DIR"

    # Cabecalho do CSV atualizado com as metricas estatisticas
    echo "imagem,largura,altura,block_x,block_y,threads_por_bloco,iteracoes,execucoes,tempo_medio_s,desvio_padrao_s,tempo_min_s,tempo_max_s" > "$OUT_CSV"

    for img in "${IMAGES[@]}"; do
        local img_path="$IMG_DIR/$img"
        if [[ ! -f "$img_path" ]]; then
            log "Pulando imagem ausente: $img_path"
            continue
        fi

        read -r width height <<< "$(extract_dims_from_name "$img")"

        for bs in "${BLOCK_SIZES[@]}"; do
            read -r bx by <<< "$bs"
            local threads=$((bx * by))

            log "Iniciando bateria: imagem=$img bloco=${bx}x${by} (${RUNS_PER_CONFIG} execucoes)"

            local tempos=()
            local falhou=0

            for ((run=1; run<=RUNS_PER_CONFIG; run++)); do
                local logfile="$LOG_DIR/${img%.png}_block${bx}x${by}_run${run}.log"

                if ! "$BIN" "$img_path" "$ITERATIONS" "$bx" "$by" > "$logfile" 2>&1; then
                    log "  -> Execucao ${run}/${RUNS_PER_CONFIG} FALHOU (veja $logfile)"
                    falhou=1
                    break
                fi

                local tempo
                tempo=$(grep -oE 'Tempo: [0-9.]+' "$logfile" | grep -oE '[0-9.]+' || echo "")

                if [[ -z "$tempo" ]]; then
                    log "  -> Execucao ${run}/${RUNS_PER_CONFIG} nao retornou tempo valido"
                    falhou=1
                    break
                fi

                tempos+=("$tempo")
            done

            if (( falhou )); then
                echo "$img,$width,$height,$bx,$by,$threads,$ITERATIONS,$RUNS_PER_CONFIG,FALHOU,FALHOU,FALHOU,FALHOU" >> "$OUT_CSV"
                continue
            fi

            # Passa a lista de tempos para o awk calcular a estatistica
            local estatisticas
            estatisticas=$(awk -v t_list="${tempos[*]}" '
                BEGIN {
                    split(t_list, arr, " ");
                    n = length(arr);
                    sum = 0;
                    min = arr[1];
                    max = arr[1];

                    for (i = 1; i <= n; i++) {
                        v = arr[i];
                        sum += v;
                        if (v < min) min = v;
                        if (v > max) max = v;
                    }
                    mean = sum / n;

                    sq_sum = 0;
                    for (i = 1; i <= n; i++) {
                        sq_sum += (arr[i] - mean) ^ 2;
                    }
                    # Desvio padrao amostral se n > 1, caso contrario 0
                    sd = (n > 1) ? sqrt(sq_sum / (n - 1)) : 0;

                    printf "%.6f,%.6f,%.6f,%.6f", mean, sd, min, max;
                }
            ')

            IFS=',' read -r avg_t std_t min_t max_t <<< "$estatisticas"

            log "  -> Concluido: Media=${avg_t}s | Desvio Padrão=${std_t}s | Min=${min_t}s | Max=${max_t}s"
            echo "$img,$width,$height,$bx,$by,$threads,$ITERATIONS,$RUNS_PER_CONFIG,$avg_t,$std_t,$min_t,$max_t" >> "$OUT_CSV"
        done
    done
}

# ----------------------------- Execução -----------------------------

check_requirements
compile
run_benchmarks

log "Benchmark concluido. Resultados salvos em: $OUT_CSV"
log "Logs individuais salvos em: $LOG_DIR/"
