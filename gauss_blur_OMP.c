#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
    int width;
    int height;
    float *data;
} Image;

void create_gaussian_kernel(float *kernel, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0;
    for (int i = 0; i < size * size; i++) {
        int r = i / size - half;
        int c = i % size - half;
        float val = expf(-(r * r + c * c) / (2.0f * sigma * sigma));
        kernel[i] = val;
        sum += val;
    }
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <arquivo_imagem> <iteracoes>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int iterations = (argc > 2) ? atoi(argv[2]) : 10;
    int k_size = 3;
    float sigma = 1.0f;

    int width, height, channels;
    unsigned char *raw_image = stbi_load(filename, &width, &height, &channels, 1);
    if (!raw_image) return 1;

    Image img1 = {width, height, (float *)malloc(width * height * sizeof(float))};
    Image img2 = {width, height, (float *)malloc(width * height * sizeof(float))};
    float *kernel = (float *)malloc(k_size * k_size * sizeof(float));

    for (int i = 0; i < width * height; i++) img1.data[i] = (float)raw_image[i];
    create_gaussian_kernel(kernel, k_size, sigma);

    Image *src = &img1;
    Image *dst = &img2;
    int h = height;
    int w = width;
    int half = k_size / 2;

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        for (int it = 0; it < iterations; it++) {
            #pragma omp for schedule(static)
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float pixel_sum = 0.0f;
                    for (int ki = -half; ki <= half; ki++) {
                        for (int kj = -half; kj <= half; kj++) {
                            int ni = i + ki;
                            int nj = j + kj;
                            if (ni < 0) ni = 0; else if (ni >= h) ni = h - 1;
                            if (nj < 0) nj = 0; else if (nj >= w) nj = w - 1;
                            pixel_sum += src->data[ni * w + nj] * kernel[(ki + half) * k_size + (kj + half)];
                        }
                    }
                    dst->data[i * w + j] = pixel_sum;
                }
            }
            #pragma omp single
            {
                Image *temp = src;
                src = dst;
                dst = temp;
            }
        }
    }

    double end = omp_get_wtime();

    printf("Arquivo: %s | Threads: %d | Tempo OMP: %.4f s\n", 
            filename, omp_get_max_threads(), end - start);

    stbi_image_free(raw_image);
    free(img1.data); free(img2.data); free(kernel);
    return 0;
}