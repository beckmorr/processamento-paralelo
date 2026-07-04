#ifdef __NVCC__
    #pragma nv_diag_suppress 550
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define K_SIZE 5
#define HALF   (K_SIZE / 2)

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

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

/* Kernel Otimizado: Resolve o problema do ping-pong atualizando toda a imagem perfeitamente */
__global__ void apply_blur_kernel(float *src, float *dst, float *kernel, int width, int height) {
    int half = HALF;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= height || j >= width) return;

    float pixel_sum = 0.0f;
    for (int ki = -half; ki <= half; ki++) {
        for (int kj = -half; kj <= half; kj++) {
            int ni = i + ki;
            int nj = j + kj;
            
            // Garante o espelhamento por replicação nas bordas (Clamp)
            if (ni < 0) ni = 0; else if (ni >= height) ni = height - 1;
            if (nj < 0) nj = 0; else if (nj >= width)  nj = width - 1;
            
            pixel_sum += src[ni * width + nj] * kernel[(ki + half) * K_SIZE + (kj + half)];
        }
    }
    dst[i * width + j] = pixel_sum;
}

void apply_blur(float *d_src, float *d_dst, float *d_kernel, int width, int height, int block_x, int block_y) {
    dim3 dimBloco(block_x, block_y);
    dim3 dimGrade((width + dimBloco.x - 1) / dimBloco.x,
                  (height + dimBloco.y - 1) / dimBloco.y);
    apply_blur_kernel<<<dimGrade, dimBloco>>>(d_src, d_dst, d_kernel, width, height);
    CUDA_CHECK(cudaGetLastError());
}

/* Caso de teste corrigido */
int run_check_mode_cuda(void) {
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

    float kernel[K_SIZE * K_SIZE];
    create_gaussian_kernel(kernel, K_SIZE, 1.0f);

    float *h_output = (float *)malloc(width * height * sizeof(float));

    float *d_src, *d_dst, *d_kernel;
    CUDA_CHECK(cudaMalloc((void **)&d_src, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_kernel, K_SIZE * K_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_src, input, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel, K_SIZE * K_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    for (int it = 0; it < iterations; it++) {
        apply_blur(d_src, d_dst, d_kernel, width, height, 16, 16);
        float *tmp = d_src; d_src = d_dst; d_dst = tmp;
    }

    CUDA_CHECK(cudaMemcpy(h_output, d_src, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%8.4f", h_output[i * width + j]);
        }
        printf("\n");
    }

    free(h_output);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_kernel);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc >= 2 && strcmp(argv[1], "--check") == 0) {
        return run_check_mode_cuda();
    }

    if (argc < 2) {
        printf("Uso: %s <arquivo_imagem> <iteracoes> <block_x> <block_y>\n", argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int iterations = (argc > 2) ? atoi(argv[2]) : 100;
    int block_x = (argc > 3) ? atoi(argv[3]) : 16;
    int block_y = (argc > 4) ? atoi(argv[4]) : 16;
    float sigma = 1.0f;

    int width, height, channels;
    unsigned char *raw_image = stbi_load(filename, &width, &height, &channels, 1);
    if (!raw_image) {
        printf("Erro ao carregar a imagem.\n");
        return 1;
    }

    float *h_img_float = (float *)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) {
        h_img_float[i] = (float)raw_image[i];
    }

    float kernel[K_SIZE * K_SIZE];
    create_gaussian_kernel(kernel, K_SIZE, sigma);

    float *d_src, *d_dst, *d_kernel;
    CUDA_CHECK(cudaMalloc((void **)&d_src, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, width * height * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_kernel, K_SIZE * K_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_src, h_img_float, width * height * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, width * height * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_kernel, kernel, K_SIZE * K_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));

    for (int it = 0; it < iterations; it++) {
        apply_blur(d_src, d_dst, d_kernel, width, height, block_x, block_y);
        float *tmp = d_src; d_src = d_dst; d_dst = tmp;
    }

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float time_taken_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_taken_ms, start, end));
    printf("Benchmark CUDA - Bloco: %dx%d - Tempo: %.4f s\n", block_x, block_y, time_taken_ms / 1000.0f);

    CUDA_CHECK(cudaMemcpy(h_img_float, d_src, width * height * sizeof(float), cudaMemcpyDeviceToHost));

    unsigned char *out = (unsigned char *)malloc(width * height);
    for (int i = 0; i < width * height; i++) {
        // Trunca valores para garantir o range ideal de cores de 0 a 255
        float p = h_img_float[i];
        if (p < 0.0f) p = 0.0f;
        if (p > 255.0f) p = 255.0f;
        out[i] = (unsigned char)p;
    }
    
    stbi_write_png("saida_blur_cuda.png", width, height, 1, out, width);

    cudaEventDestroy(start); cudaEventDestroy(end);
    free(out); free(h_img_float); free(raw_image);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_kernel);
    return 0;
}
