#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE 1000000
#define MULTIPLIER 2.5f
#define BLOCK_SIZE 256

// Ядро с использованием только глобальной памяти
__global__ void multiplyGlobalMemory(float *input, float *output, float multiplier, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * multiplier;
    }
}

// Ядро с использованием разделяемой памяти
__global__ void multiplySharedMemory(float *input, float *output, float multiplier, int n) {
    __shared__ float shared_data[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Загрузка данных в разделяемую память
    if (idx < n) {
        shared_data[tid] = input[idx];
    }
    __syncthreads();

    // Обработка данных из разделяемой памяти
    if (idx < n) {
        output[idx] = shared_data[tid] * multiplier;
    }
}

void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

double measureKernelTime(void (*kernel)(float*, float*, float, int),
                         float *d_input, float *d_output,
                         float multiplier, int n, int blockSize) {
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_input, d_output, multiplier, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

int main() {
    printf("=== Задание 1: Сравнение глобальной и разделяемой памяти ===\n");
    printf("Размер массива: %d элементов\n", ARRAY_SIZE);
    printf("Множитель: %.2f\n", MULTIPLIER);
    printf("Размер блока: %d потоков\n\n", BLOCK_SIZE);

    // Выделение памяти на хосте
    size_t size = ARRAY_SIZE * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Инициализация входных данных
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = (float)(i % 100);
    }

    // Выделение памяти на устройстве
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, size), "cudaMalloc d_output");

    // Копирование данных на устройство
    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice),
                   "cudaMemcpy to device");

    // Тест 1: Глобальная память
    printf("1. Использование только глобальной памяти:\n");
    double time_global = measureKernelTime(multiplyGlobalMemory, d_input, d_output,
                                          MULTIPLIER, ARRAY_SIZE, BLOCK_SIZE);
    printf("   Время выполнения: %.4f мс\n\n", time_global);

    // Копирование результата для проверки
    checkCudaError(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost),
                   "cudaMemcpy from device");

    // Тест 2: Разделяемая память
    printf("2. Использование разделяемой памяти:\n");
    double time_shared = measureKernelTime(multiplySharedMemory, d_input, d_output,
                                          MULTIPLIER, ARRAY_SIZE, BLOCK_SIZE);
    printf("   Время выполнения: %.4f мс\n\n", time_shared);

    // Проверка корректности результатов
    checkCudaError(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost),
                   "cudaMemcpy from device");

    bool correct = true;
    for (int i = 0; i < 10 && correct; i++) {
        float expected = h_input[i] * MULTIPLIER;
        if (fabs(h_output[i] - expected) > 0.001f) {
            correct = false;
            printf("Ошибка в элементе %d: ожидалось %.2f, получено %.2f\n",
                   i, expected, h_output[i]);
        }
    }

    if (correct) {
        printf("✓ Результаты корректны\n\n");
    }

    // Сравнение производительности
    printf("=== Результаты сравнения ===\n");
    printf("Глобальная память:    %.4f мс\n", time_global);
    printf("Разделяемая память:   %.4f мс\n", time_shared);
    printf("Разница:              %.4f мс (%.2f%%)\n",
           time_global - time_shared,
           ((time_global - time_shared) / time_global) * 100);

    if (time_shared < time_global) {
        printf("Разделяемая память быстрее в %.2f раз\n", time_global / time_shared);
    } else {
        printf("Глобальная память быстрее в %.2f раз\n", time_shared / time_global);
    }

    // Освобождение памяти
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
