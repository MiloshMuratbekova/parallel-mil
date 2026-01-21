#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1000000
#define BLOCK_SIZE 256

// Коалесцированный доступ к памяти (последовательный)
__global__ void processCoalesced(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Последовательный доступ: потоки читают последовательные элементы
        output[idx] = input[idx] * 2.0f + 1.0f;
    }
}

// Некоалесцированный доступ к памяти (со смещением)
__global__ void processNonCoalesced(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Доступ со смещением: каждый поток читает элемент со смещением
        // Это нарушает коалесценцию, так как соседние потоки обращаются
        // к несмежным адресам памяти
        int strided_idx = (idx * 32) % n; // Шаг 32 элемента
        output[idx] = input[strided_idx] * 2.0f + 1.0f;
    }
}

// Некоалесцированный доступ (обратный порядок)
__global__ void processReversed(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Обратный порядок доступа
        int reversed_idx = n - 1 - idx;
        output[idx] = input[reversed_idx] * 2.0f + 1.0f;
    }
}

void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

template<typename Kernel>
double measureKernelTime(Kernel kernel, float *d_input, float *d_output,
                         int n, int blockSize, int numRuns = 10) {
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Прогрев
    kernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    // Измерение времени
    cudaEventRecord(start);
    for (int i = 0; i < numRuns; i++) {
        kernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / numRuns;
}

int main() {
    printf("=== Задание 3: Коалесцированный vs Некоалесцированный доступ ===\n");
    printf("Размер массива: %d элементов\n", ARRAY_SIZE);
    printf("Размер блока: %d потоков\n\n", BLOCK_SIZE);

    // Выделение памяти на хосте
    size_t size = ARRAY_SIZE * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Инициализация данных
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = (float)i;
    }

    // Выделение памяти на устройстве
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, size), "cudaMalloc d_output");

    // Копирование данных на устройство
    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice),
                   "cudaMemcpy to device");

    printf("=== Тестирование различных шаблонов доступа ===\n\n");

    // Тест 1: Коалесцированный доступ
    printf("1. Коалесцированный доступ (последовательный):\n");
    printf("   Описание: потоки читают последовательные элементы\n");
    double time_coalesced = measureKernelTime(processCoalesced, d_input, d_output,
                                             ARRAY_SIZE, BLOCK_SIZE);
    printf("   Время выполнения: %.4f мс\n", time_coalesced);

    double bandwidth_coalesced = (2.0 * size) / (time_coalesced * 1e-3) / 1e9;
    printf("   Пропускная способность: %.2f GB/s\n\n", bandwidth_coalesced);

    // Тест 2: Некоалесцированный доступ (со смещением)
    printf("2. Некоалесцированный доступ (со смещением):\n");
    printf("   Описание: доступ с шагом 32 элемента\n");
    double time_strided = measureKernelTime(processNonCoalesced, d_input, d_output,
                                           ARRAY_SIZE, BLOCK_SIZE);
    printf("   Время выполнения: %.4f мс\n", time_strided);

    double bandwidth_strided = (2.0 * size) / (time_strided * 1e-3) / 1e9;
    printf("   Пропускная способность: %.2f GB/s\n\n", bandwidth_strided);

    // Тест 3: Некоалесцированный доступ (обратный порядок)
    printf("3. Некоалесцированный доступ (обратный порядок):\n");
    printf("   Описание: чтение элементов в обратном порядке\n");
    double time_reversed = measureKernelTime(processReversed, d_input, d_output,
                                            ARRAY_SIZE, BLOCK_SIZE);
    printf("   Время выполнения: %.4f мс\n", time_reversed);

    double bandwidth_reversed = (2.0 * size) / (time_reversed * 1e-3) / 1e9;
    printf("   Пропускная способность: %.2f GB/s\n\n", bandwidth_reversed);

    // Сравнительный анализ
    printf("=== Сравнительный анализ ===\n");
    printf("%-30s | %-15s | %-15s | %-15s\n",
           "Тип доступа", "Время (мс)", "Пропускная (GB/s)", "Замедление");
    printf("-------------------------------+-----------------+------------------+-----------------\n");
    printf("%-30s | %-15.4f | %-15.2f | %-15.2fx\n",
           "Коалесцированный", time_coalesced, bandwidth_coalesced, 1.0);
    printf("%-30s | %-15.4f | %-15.2f | %-15.2fx\n",
           "Со смещением", time_strided, bandwidth_strided,
           time_strided / time_coalesced);
    printf("%-30s | %-15.4f | %-15.2f | %-15.2fx\n",
           "Обратный порядок", time_reversed, bandwidth_reversed,
           time_reversed / time_coalesced);

    printf("\n=== Выводы ===\n");
    printf("• Коалесцированный доступ обеспечивает максимальную производительность\n");
    printf("• Некоалесцированный доступ замедляет выполнение в %.2f-%.2f раз\n",
           time_strided / time_coalesced,
           time_reversed / time_coalesced);
    printf("• Пропускная способность снижается на %.0f%%-%.0f%%\n",
           (1 - bandwidth_strided / bandwidth_coalesced) * 100,
           (1 - bandwidth_reversed / bandwidth_coalesced) * 100);

    // Информация о памяти устройства
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\n=== Информация об устройстве ===\n");
    printf("Устройство: %s\n", prop.name);
    printf("Размер кэш-линии глобальной памяти: 128 байт\n");
    printf("Размер варпа: %d потоков\n", prop.warpSize);
    printf("Оптимальная транзакция: %d последовательных float (32 потока * 4 байта)\n",
           prop.warpSize);

    // Освобождение памяти
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
