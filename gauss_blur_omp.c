#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

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

/* Convolução 2D*/
void apply_blur(float *src, float *dst, float *kernel, int width, int local_rows, int p_w, int local_start, int height) {
    int half = HALF;
    /* Paralelização dentro de linhas locais */
#pragma omp parallel for schedule(static)
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

/* Função para realizar a corretude */
int run_check_mode_omp(void) {
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
    float kernel[K_SIZE * K_SIZE];
    create_gaussian_kernel(kernel, K_SIZE, 1.0f);

    float *lsrc = (float *)calloc(p_w * p_h, sizeof(float));
    float *ldst = (float *)calloc(p_w * p_h, sizeof(float));
    if (!lsrc || !ldst) {
        printf("Erro de memoria no modo --check.\n");
        free(lsrc); free(ldst);
        return 1;
    }

    for (int i = 0; i < p_h; i++) {
        for (int j = 0; j < p_w; j++) {
            int oi = i - HALF; if (oi < 0) oi = 0; if (oi >= height) oi = height - 1;
            int oj = j - HALF; if (oj < 0) oj = 0; if (oj >= width)  oj = width - 1;
            lsrc[i * p_w + j] = input[oi * width + oj];
        }
    }

    int local_rows = height;
    int local_start = 0;

    for (int it = 0; it < iterations; it++) {
        apply_blur(lsrc, ldst, kernel, width, local_rows, p_w, local_start, height);
        float *tmp = lsrc; lsrc = ldst; ldst = tmp;
    }

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%8.4f", lsrc[(i + HALF) * p_w + (j + HALF)]);
        }
        printf("\n");
    }

    free(lsrc); free(ldst);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc >= 2 && strcmp(argv[1], "--check") == 0) {
        return run_check_mode_omp();
    }

    if (argc < 2) {
        printf("Uso: %s <arquivo_imagem> <iteracoes>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int iterations = (argc > 2) ? atoi(argv[2]) : 100;
    float sigma = 1.0f;

    int width, height, channels;
    unsigned char *raw_image = stbi_load(filename, &width, &height, &channels, 1);
    if (!raw_image) {
        printf("Erro ao carregar a imagem.\n");
        return 1;
    }

    int p_w = width + 2 * HALF;
    int p_h = height + 2 * HALF;

    float *full_padded = (float *)malloc(p_w * p_h * sizeof(float));
    for (int i = 0; i < p_h; i++) {
        for (int j = 0; j < p_w; j++) {
            int oi = i - HALF; if (oi < 0) oi = 0; if (oi >= height) oi = height - 1;
            int oj = j - HALF; if (oj < 0) oj = 0; if (oj >= width)  oj = width - 1;
            full_padded[i * p_w + j] = (float)raw_image[oi * width + oj];
        }
    }

    float kernel[K_SIZE * K_SIZE];
    create_gaussian_kernel(kernel, K_SIZE, sigma);

    int local_rows = height;
    int local_start = 0;
    int local_p_h = local_rows + 2 * HALF;

    float *lsrc = (float *)calloc(local_p_h * p_w, sizeof(float));
    float *ldst = (float *)calloc(local_p_h * p_w, sizeof(float));

    memcpy(lsrc, full_padded, p_w * p_h * sizeof(float));

    double start = omp_get_wtime();
    for (int it = 0; it < iterations; it++) {
        apply_blur(lsrc, ldst, kernel, width, local_rows, p_w, local_start, height);
        float *tmp = lsrc; lsrc = ldst; ldst = tmp;
    }
    double end = omp_get_wtime();

    printf("Benchmark OpenMP - Tempo: %.4f s\n", end - start);

    unsigned char *out = malloc(width * height);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            out[i * width + j] = (unsigned char)lsrc[(i + HALF) * p_w + (j + HALF)];
    stbi_write_png("saida_blur_omp.png", width, height, 1, out, width);

    free(out); free(full_padded); free(lsrc); free(ldst); free(raw_image);
    return 0;
}