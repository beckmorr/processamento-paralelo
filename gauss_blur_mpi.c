#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define K_SIZE 5
#define HALF   (K_SIZE / 2)

/* Gera e normaliza o kernel Gaussiano para que a soma seja 1.0. */
void create_gaussian_kernel(float *kernel, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0f;
    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            float val = expf(-(i * i + j * j) / (2.0f * sigma * sigma));
            kernel[(i + half) * size + (j + half)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
}

/* Convolução 2D sobre uma fatia local com halos e mapeamento global->local. */
void apply_blur(float *src, float *dst, float *kernel, int width, int local_rows, int p_w, int local_start, int height) {
    int half = HALF;
    for (int li = 0; li < local_rows; li++) {
        int global_i = local_start + li;
        for (int j = 0; j < width; j++) {
            float pixel_sum = 0.0f;
            for (int ki = -half; ki <= half; ki++) {
                for (int kj = -half; kj <= half; kj++) {
                    int ni = global_i + ki;
                    int nj = j + kj;
                    if (ni < 0) ni = 0; else if (ni >= height) ni = height - 1;
                    if (nj < 0) nj = 0; else if (nj >= width)  nj = width - 1;
                    int local_ni = (ni - local_start) + half;
                    int local_nj = nj + half;
                    pixel_sum += src[local_ni * p_w + local_nj] * kernel[(ki + half) * K_SIZE + (kj + half)];
                }
            }
            dst[(li + half) * p_w + (j + half)] = pixel_sum;
        }
    }
}

/* Atualiza halos com vizinhos para manter corretude entre iteracoes. */
void exchange_halos(float *buf, int local_rows, int p_w, int rank, int nprocs) {
    int halo_elems = HALF * p_w;
    if (local_rows <= 0) return;

    if (rank > 0) {
        MPI_Sendrecv(buf + HALF * p_w, halo_elems, MPI_FLOAT, rank - 1, 10,
                     buf, halo_elems, MPI_FLOAT, rank - 1, 20,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        memcpy(buf, buf + HALF * p_w, halo_elems * sizeof(float));
    }

    if (rank < nprocs - 1) {
        MPI_Sendrecv(buf + local_rows * p_w, halo_elems, MPI_FLOAT, rank + 1, 20,
                     buf + (local_rows + HALF) * p_w, halo_elems, MPI_FLOAT, rank + 1, 10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        memcpy(buf + (local_rows + HALF) * p_w, buf + local_rows * p_w, halo_elems * sizeof(float));
    }
}

/* Caso de corretude fixo: matriz 5x5, kernel 3x3 e 2 iterações. */
int run_check_mode_mpi(int rank, int nprocs) {
    const int width = 5;
    const int height = 5;
    const int iterations = 2;
    const float input[25] = {
        1, 2, 3, 2, 1,
        2, 4, 6, 4, 2,
        3, 6, 9, 6, 3,
        2, 4, 6, 4, 2,
        1, 2, 3, 2, 1
    };

    int p_w = width + 2 * HALF;
    int p_h = height + 2 * HALF;
    float *full_padded = NULL;

    if (rank == 0) {
        /* Padding por replicação de borda via clamp de índices. */
        full_padded = (float *)malloc(p_w * p_h * sizeof(float));
        if (!full_padded) {
            printf("Erro de memoria no modo --check.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < p_h; i++) {
            for (int j = 0; j < p_w; j++) {
                int oi = i - HALF; if (oi < 0) oi = 0; if (oi >= height) oi = height - 1;
                int oj = j - HALF; if (oj < 0) oj = 0; if (oj >= width)  oj = width - 1;
                full_padded[i * p_w + j] = input[oi * width + oj];
            }
        }
    }

    float kernel[K_SIZE * K_SIZE];
    if (rank == 0) create_gaussian_kernel(kernel, K_SIZE, 1.0f);
    MPI_Bcast(kernel, K_SIZE * K_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int *sendcounts = (int *)malloc(nprocs * sizeof(int));
    int *displs = (int *)malloc(nprocs * sizeof(int));
    int *sendcounts_scatter = (int *)malloc(nprocs * sizeof(int));
    int *displs_scatter = (int *)malloc(nprocs * sizeof(int));
    int *row_count = (int *)malloc(nprocs * sizeof(int));
    int *row_start = (int *)malloc(nprocs * sizeof(int));

    if (rank == 0) {
        /* Particiona linhas entre processos e define deslocamentos para halo/gather. */
        int base = height / nprocs;
        int current_start = 0;
        for (int r = 0; r < nprocs; r++) {
            row_count[r] = base + (r < (height % nprocs) ? 1 : 0);
            row_start[r] = current_start;
            sendcounts[r] = row_count[r] * p_w;
            displs[r] = (current_start + HALF) * p_w;
            sendcounts_scatter[r] = (row_count[r] + 2 * HALF) * p_w;
            displs_scatter[r] = current_start * p_w;
            current_start += row_count[r];
        }
    }
    MPI_Bcast(row_count, nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_start, nprocs, MPI_INT, 0, MPI_COMM_WORLD);

    int local_rows = row_count[rank];
    int local_p_h = local_rows + 2 * HALF;
    int local_start = row_start[rank];

    float *lsrc = (float *)calloc(local_p_h * p_w, sizeof(float));
    float *ldst = (float *)calloc(local_p_h * p_w, sizeof(float));

    /* Distribui blocos locais já com linhas halo superior e inferior. */
    MPI_Scatterv(full_padded, sendcounts_scatter, displs_scatter, MPI_FLOAT,
                 lsrc, local_p_h * p_w, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Ping-pong de buffers entre iterações. */
    for (int it = 0; it < iterations; it++) {
        exchange_halos(lsrc, local_rows, p_w, rank, nprocs);
        apply_blur(lsrc, ldst, kernel, width, local_rows, p_w, local_start, height);
        float *tmp = lsrc; lsrc = ldst; ldst = tmp;
    }

    /* Reúne apenas a região útil (sem halos) de cada processo. */
    MPI_Gatherv(lsrc + HALF * p_w, local_rows * p_w, MPI_FLOAT,
                full_padded, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                printf("%8.4f", full_padded[(i + HALF) * p_w + (j + HALF)]);
            }
            printf("\n");
        }
        free(full_padded);
    }

    free(lsrc); free(ldst); free(sendcounts); free(displs);
    free(sendcounts_scatter); free(displs_scatter); free(row_count); free(row_start);
    return 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* Atalho para teste determinístico de corretude. */
    if (argc >= 2 && strcmp(argv[1], "--check") == 0) {
        int check_ret = run_check_mode_mpi(rank, nprocs);
        MPI_Finalize();
        return check_ret;
    }

    int width = 0, height = 0, iterations = 100;
    unsigned char *raw_image = NULL;
    float *full_padded = NULL;

    if (rank == 0) {
        if (argc < 2) { printf("Uso: mpirun -np <procs> %s <img.png>\n", argv[0]); MPI_Abort(MPI_COMM_WORLD, 1); }
        int ch;
        raw_image = stbi_load(argv[1], &width, &height, &ch, 1);
        if (argc > 2) iterations = atoi(argv[2]);
    }
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int p_w = width + 2 * HALF;
    int p_h = height + 2 * HALF;

    if (rank == 0) {
        /* Imagem padded completa montada no processo raiz. */
        full_padded = (float *)malloc(p_w * p_h * sizeof(float));
        for (int i = 0; i < p_h; i++) {
            for (int j = 0; j < p_w; j++) {
                int oi = i - HALF; if (oi < 0) oi = 0; if (oi >= height) oi = height - 1;
                int oj = j - HALF; if (oj < 0) oj = 0; if (oj >= width)  oj = width - 1;
                full_padded[i * p_w + j] = (float)raw_image[oi * width + oj];
            }
        }
    }

    float kernel[K_SIZE * K_SIZE];
    if (rank == 0) create_gaussian_kernel(kernel, K_SIZE, 1.0f);
    MPI_Bcast(kernel, K_SIZE * K_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int *sendcounts = (int *)malloc(nprocs * sizeof(int));
    int *displs = (int *)malloc(nprocs * sizeof(int));
    int *sendcounts_scatter = (int *)malloc(nprocs * sizeof(int));
    int *displs_scatter = (int *)malloc(nprocs * sizeof(int));
    int *row_count = (int *)malloc(nprocs * sizeof(int));
    int *row_start = (int *)malloc(nprocs * sizeof(int));

    if (rank == 0) {
        /* Particiona linhas entre processos e define deslocamentos para halo/gather. */
        int base = height / nprocs;
        int current_start = 0;
        for (int r = 0; r < nprocs; r++) {
            row_count[r] = base + (r < (height % nprocs) ? 1 : 0);
            row_start[r] = current_start;
            sendcounts[r] = row_count[r] * p_w;
            displs[r] = (current_start + HALF) * p_w;
            sendcounts_scatter[r] = (row_count[r] + 2 * HALF) * p_w;
            displs_scatter[r] = (current_start) * p_w;
            current_start += row_count[r];
        }
    }
    MPI_Bcast(row_count, nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_start, nprocs, MPI_INT, 0, MPI_COMM_WORLD);
    int local_rows = row_count[rank];
    int local_p_h = local_rows + 2 * HALF;
    int local_start = row_start[rank];

    float *lsrc = (float *)calloc(local_p_h * p_w, sizeof(float));
    float *ldst = (float *)calloc(local_p_h * p_w, sizeof(float));

    /* Distribui blocos locais já com linhas halo superior e inferior. */
    MPI_Scatterv(full_padded, sendcounts_scatter, displs_scatter, MPI_FLOAT, lsrc, local_p_h * p_w, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /* Ping-pong de buffers entre iterações. */
    for (int it = 0; it < iterations; it++) {
        exchange_halos(lsrc, local_rows, p_w, rank, nprocs);
        apply_blur(lsrc, ldst, kernel, width, local_rows, p_w, local_start, height);
        float *tmp = lsrc; lsrc = ldst; ldst = tmp;
    }

    /* Reúne apenas a região útil (sem halos) de cada processo. */
    MPI_Gatherv(lsrc + HALF * p_w, local_rows * p_w, MPI_FLOAT, full_padded, sendcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Benchmark MPI - Tempo: %.4f s\n", MPI_Wtime() - start_time);
        unsigned char *out = malloc(width * height);
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                out[i * width + j] = (unsigned char)full_padded[(i + HALF) * p_w + (j + HALF)];
        stbi_write_png("saida_blur_mpi.png", width, height, 1, out, width);
        free(out); free(full_padded);
    }
    free(lsrc); free(ldst); free(sendcounts); free(displs); free(row_count); free(sendcounts_scatter); free(displs_scatter); free(row_start);
    MPI_Finalize();
    return 0;
}