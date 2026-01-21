#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ARRAY_SIZE 1000000

// Ядро для умножения элементов массива на константу
__global__ void multiplyArray(const float *input, float *output, float multiplier, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * multiplier;
    }
}

void checkCudaError(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

double measurePerformance(const float *d_input, float *d_output, float multiplier,
                         int n, int blockSize, int gridSize, int numRuns = 10) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Прогрев
    multiplyArray<<<gridSize, blockSize>>>(d_input, d_output, multiplier, n);
    cudaDeviceSynchronize();

    // Измерение времени
    cudaEventRecord(start);
    for (int i = 0; i < numRuns; i++) {
        multiplyArray<<<gridSize, blockSize>>>(d_input, d_output, multiplier, n);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / numRuns;
}

void testConfiguration(const float *d_input, float *d_output, float multiplier,
                      int n, int blockSize, int gridSize,
                      const char *description) {
    double time = measurePerformance(d_input, d_output, multiplier, n,
                                    blockSize, gridSize);

    int totalThreads = gridSize * blockSize;
    double bandwidth = (2.0 * n * sizeof(float)) / (time * 1e-3) / 1e9;
    double efficiency = (double)n / totalThreads * 100.0;

    printf("%-40s | %5d | %6d | %8d | %7.4f | %6.2f | %7.2f%%\n",
           description, blockSize, gridSize, totalThreads,
           time, bandwidth, efficiency);
}

int main() {
    printf("=== Задание 4: Оптимизация конфигурации сетки и блоков ===\n");
    printf("Размер массива: %d элементов\n\n", ARRAY_SIZE);

    // Получение информации об устройстве
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("=== Информация об устройстве ===\n");
    printf("Устройство: %s\n", prop.name);
    printf("Число мультипроцессоров: %d\n", prop.multiProcessorCount);
    printf("Макс. потоков на блок: %d\n", prop.maxThreadsPerBlock);
    printf("Макс. потоков на мультипроцессор: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Размер варпа: %d\n", prop.warpSize);
    printf("Макс. размер блока по X: %d\n", prop.maxThreadsDim[0]);
    printf("Макс. размер сетки по X: %d\n\n", prop.maxGridSize[0]);

    // Выделение памяти
    size_t size = ARRAY_SIZE * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_input[i] = (float)i;
    }

    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, size), "cudaMalloc d_output");
    checkCudaError(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice),
                   "cudaMemcpy to device");

    float multiplier = 2.5f;

    printf("=== Тестирование различных конфигураций ===\n");
    printf("%-40s | %-5s | %-6s | %-8s | %-7s | %-6s | %-10s\n",
           "Конфигурация", "Block", "Grid", "Threads", "Time(ms)", "BW(GB/s)", "Efficiency");
    printf("%-40s-+-------+--------+----------+---------+--------+-----------\n",
           "----------------------------------------");

    double minTime = 1e9;
    int bestBlockSize = 0;
    int bestGridSize = 0;

    // 1. Неоптимальные конфигурации
    printf("\n--- Неоптимальные конфигурации ---\n");

    // Слишком маленький блок
    int blockSize = 32;
    int gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    testConfiguration(d_input, d_output, multiplier, ARRAY_SIZE,
                     blockSize, gridSize, "Маленький блок (32)");

    // Слишком большой блок
    blockSize = 1024;
    gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    testConfiguration(d_input, d_output, multiplier, ARRAY_SIZE,
                     blockSize, gridSize, "Большой блок (1024)");

    // Неоптимальный размер (не кратен warp size)
    blockSize = 100;
    gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    testConfiguration(d_input, d_output, multiplier, ARRAY_SIZE,
                     blockSize, gridSize, "Не кратен варпу (100)");

    // Слишком мало блоков
    blockSize = 256;
    gridSize = prop.multiProcessorCount;
    testConfiguration(d_input, d_output, multiplier, ARRAY_SIZE,
                     blockSize, gridSize, "Мало блоков (SM count)");

    // 2. Оптимальные конфигурации
    printf("\n--- Оптимальные конфигурации ---\n");

    // Стандартная конфигурация: 256 потоков на блок
    blockSize = 256;
    gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    double time256 = measurePerformance(d_input, d_output, multiplier,
                                       ARRAY_SIZE, blockSize, gridSize);
    testConfiguration(d_input, d_output, multiplier, ARRAY_SIZE,
                     blockSize, gridSize, "Стандартная (256)");
    if (time256 < minTime) {
        minTime = time256;
        bestBlockSize = blockSize;
        bestGridSize = gridSize;
    }

    // Оптимальная: 512 потоков на блок
    blockSize = 512;
    gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    double time512 = measurePerformance(d_input, d_output, multiplier,
                                       ARRAY_SIZE, blockSize, gridSize);
    testConfiguration(d_input, d_output, multiplier, ARRAY_SIZE,
                     blockSize, gridSize, "Оптимальная (512)");
    if (time512 < minTime) {
        minTime = time512;
        bestBlockSize = blockSize;
        bestGridSize = gridSize;
    }

    // С учетом occupancy: блоков кратно числу SM
    blockSize = 256;
    int blocksPerSM = 4; // Типичное значение для хорошей occupancy
    gridSize = prop.multiProcessorCount * blocksPerSM;
    // Корректировка для покрытия всего массива
    int minGridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    if (gridSize < minGridSize) gridSize = minGridSize;

    double timeOccupancy = measurePerformance(d_input, d_output, multiplier,
                                             ARRAY_SIZE, blockSize, gridSize);
    testConfiguration(d_input, d_output, multiplier, ARRAY_SIZE,
                     blockSize, gridSize, "С учетом occupancy");
    if (timeOccupancy < minTime) {
        minTime = timeOccupancy;
        bestBlockSize = blockSize;
        bestGridSize = gridSize;
    }

    // Балансная конфигурация
    blockSize = 128;
    gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    double time128 = measurePerformance(d_input, d_output, multiplier,
                                       ARRAY_SIZE, blockSize, gridSize);
    testConfiguration(d_input, d_output, multiplier, ARRAY_SIZE,
                     blockSize, gridSize, "Балансная (128)");
    if (time128 < minTime) {
        minTime = time128;
        bestBlockSize = blockSize;
        bestGridSize = gridSize;
    }

    // 3. Результаты оптимизации
    printf("\n=== Результаты оптимизации ===\n");
    printf("Лучшая конфигурация:\n");
    printf("  Размер блока: %d потоков\n", bestBlockSize);
    printf("  Размер сетки: %d блоков\n", bestGridSize);
    printf("  Общее число потоков: %d\n", bestBlockSize * bestGridSize);
    printf("  Время выполнения: %.4f мс\n", minTime);

    double bestBandwidth = (2.0 * size) / (minTime * 1e-3) / 1e9;
    printf("  Пропускная способность: %.2f GB/s\n", bestBandwidth);

    // Сравнение с неоптимальной конфигурацией
    double worstTime = measurePerformance(d_input, d_output, multiplier,
                                         ARRAY_SIZE, 32,
                                         (ARRAY_SIZE + 31) / 32);
    printf("\nУлучшение по сравнению с неоптимальной конфигурацией:\n");
    printf("  Ускорение: %.2fx\n", worstTime / minTime);
    printf("  Экономия времени: %.2f%%\n", (1 - minTime / worstTime) * 100);

    // Проверка корректности
    multiplyArray<<<bestGridSize, bestBlockSize>>>(d_input, d_output, multiplier, ARRAY_SIZE);
    checkCudaError(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost),
                   "cudaMemcpy from device");

    bool correct = true;
    for (int i = 0; i < 10 && correct; i++) {
        float expected = h_input[i] * multiplier;
        if (fabs(h_output[i] - expected) > 0.001f) {
            correct = false;
        }
    }

    if (correct) {
        printf("\n✓ Результаты корректны\n");
    }

    printf("\n=== Рекомендации по оптимизации ===\n");
    printf("1. Используйте размеры блоков кратные размеру варпа (%d)\n", prop.warpSize);
    printf("2. Оптимальные размеры блоков: 128-512 потоков\n");
    printf("3. Обеспечьте достаточное число блоков для загрузки всех SM\n");
    printf("4. Минимум блоков: %d (число мультипроцессоров)\n", prop.multiProcessorCount);
    printf("5. Рекомендуемое число блоков: %d (для лучшей occupancy)\n",
           prop.multiProcessorCount * 4);

    // Освобождение памяти
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
