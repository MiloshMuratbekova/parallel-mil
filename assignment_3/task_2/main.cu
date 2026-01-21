#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1000000

// Ядро для поэлементного сложения двух массивов
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

double measurePerformance(const float *d_A, const float *d_B, float *d_C,
                         int n, int blockSize) {
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Прогрев
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    // Измерение времени
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / 10.0; // Среднее время
}

int main() {
    printf("=== Задание 2: Влияние размера блока на производительность ===\n");
    printf("Размер массива: %d элементов\n\n", ARRAY_SIZE);

    // Выделение памяти на хосте
    size_t size = ARRAY_SIZE * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Инициализация данных
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B");
    checkCudaError(cudaMalloc(&d_C, size), "cudaMalloc d_C");

    // Копирование данных на устройство
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy B");

    // Тестирование различных размеров блока
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numTests = sizeof(blockSizes) / sizeof(blockSizes[0]);

    printf("%-15s | %-15s | %-15s | %-15s\n",
           "Размер блока", "Размер сетки", "Время (мс)", "Пропускная способность (GB/s)");
    printf("----------------+-----------------+-----------------+------------------\n");

    double minTime = 1e9;
    int bestBlockSize = 0;

    for (int i = 0; i < numTests; i++) {
        int blockSize = blockSizes[i];
        int gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;

        double time = measurePerformance(d_A, d_B, d_C, ARRAY_SIZE, blockSize);

        // Расчет пропускной способности (3 массива: 2 чтения + 1 запись)
        double bandwidth = (3.0 * size) / (time * 1e-3) / 1e9; // GB/s

        printf("%-15d | %-15d | %-15.4f | %-15.2f\n",
               blockSize, gridSize, time, bandwidth);

        if (time < minTime) {
            minTime = time;
            bestBlockSize = blockSize;
        }
    }

    printf("\n=== Результаты анализа ===\n");
    printf("Лучший размер блока: %d потоков\n", bestBlockSize);
    printf("Минимальное время: %.4f мс\n", minTime);

    // Проверка корректности
    int gridSize = (ARRAY_SIZE + bestBlockSize - 1) / bestBlockSize;
    vectorAdd<<<gridSize, bestBlockSize>>>(d_A, d_B, d_C, ARRAY_SIZE);
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    bool correct = true;
    for (int i = 0; i < 10 && correct; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 0.001f) {
            correct = false;
            printf("Ошибка в элементе %d: ожидалось %.2f, получено %.2f\n",
                   i, expected, h_C[i]);
        }
    }

    if (correct) {
        printf("\n✓ Результаты корректны\n");
    }

    // Информация об устройстве
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n=== Информация об устройстве ===\n");
    printf("Устройство: %s\n", prop.name);
    printf("Максимальный размер блока: %d потоков\n", prop.maxThreadsPerBlock);
    printf("Размер варпа: %d потоков\n", prop.warpSize);
    printf("Число мультипроцессоров: %d\n", prop.multiProcessorCount);

    // Освобождение памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
