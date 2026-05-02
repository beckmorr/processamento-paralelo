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
    int stride;
    float *data;
} Image;

void create_gaussian_kernel(float *kernel, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0;
    for (int i = -half; i <= half; i++) {
        for (int j = -half; j <= half; j++) {
            float val = expf(-(i * i + j * j) / (2.0f * sigma * sigma));
            kernel[(i + half) * size + (j + half)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < size * size; i++) kernel[i] /= sum;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Uso: %s <arquivo_imagem> <iteracoes>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int iterations = (argc > 2) ? atoi(argv[2]) : 100;
    int k_size = 3;
    int half = k_size / 2;
    float sigma = 1.0f;

    int width, height, channels;
    unsigned char *raw_image = stbi_load(filename, &width, &height, &channels, 1);
    if (!raw_image) return 1;

    int p_w = width + 2 * half;
    int p_h = height + 2 * half;
    
    Image img1 = {width, height, p_w, (float *)malloc(p_w * p_h * sizeof(float))};
    Image img2 = {width, height, p_w, (float *)malloc(p_w * p_h * sizeof(float))};
    float *kernel = (float *)malloc(k_size * k_size * sizeof(float));
    create_gaussian_kernel(kernel, k_size, sigma);

    for (int i = 0; i < p_h; i++) {
        for (int j = 0; j < p_w; j++) {
            int orig_i = i - half;
            int orig_j = j - half;
            if (orig_i < 0) orig_i = 0; else if (orig_i >= height) orig_i = height - 1;
            if (orig_j < 0) orig_j = 0; else if (orig_j >= width) orig_j = width - 1;
            img1.data[i * p_w + j] = (float)raw_image[orig_i * width + orig_j];
        }
    }

    Image *src = &img1;
    Image *dst = &img2;

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        for (int it = 0; it < iterations; it++) {
            #pragma omp for schedule(static)
            for (int i = half; i < height + half; i++) {
                for (int j = half; j < width + half; j++) {
                    float pixel_sum = 0.0f;
                    
                    for (int ki = -half; ki <= half; ki++) {
                        for (int kj = -half; kj <= half; kj++) {
                            pixel_sum += src->data[(i + ki) * p_w + (j + kj)] * 
                                         kernel[(ki + half) * k_size + (kj + half)];
                        }
                    }
                    dst->data[i * p_w + j] = pixel_sum;
                }
            }

            #pragma omp single
            {
                Image *temp = src; src = dst; dst = temp;
            } 
        }
    }

    double end = omp_get_wtime();
    
    printf("Arquivo: %s | Threads: %d | Iteracoes: %d | Tempo: %.4f s\n", 
           filename, omp_get_max_threads(), iterations, end - start);

    stbi_image_free(raw_image);
    free(img1.data); free(img2.data); free(kernel);
    return 0;
}