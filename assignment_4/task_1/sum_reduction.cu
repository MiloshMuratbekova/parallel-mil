#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// CUDA kernel для редукции суммы с использованием глобальной памяти
__global__ void sum_reduction_global(float* input, float* output, int n) {
    // Вычисляем глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Каждый поток суммирует элементы с шагом stride
    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }

    // Записываем частичную сумму в глобальную память
    output[idx] = sum;
}

// Оптимизированный CUDA kernel с использованием атомарных операций
__global__ void sum_reduction_atomic(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        sum += input[i];
    }

    // Атомарное добавление к результату
    atomicAdd(output, sum);
}

// Последовательная реализация на CPU
float sum_cpu(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    return sum;
}

// Функция проверки ошибок CUDA
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

int main() {
    // Параметры
    const int N = 100000;  // Размер массива
    const int BLOCK_SIZE = 256;
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cout << "=== CUDA: Вычисление суммы элементов массива ===" << endl;
    cout << "Размер массива: " << N << " элементов" << endl;
    cout << "Размер блока: " << BLOCK_SIZE << " потоков" << endl;
    cout << "Количество блоков: " << GRID_SIZE << endl;
    cout << "Общее количество потоков: " << BLOCK_SIZE * GRID_SIZE << endl;
    cout << "================================================\n" << endl;

    // Выделение памяти на хосте
    float* h_input = new float[N];
    float* h_output_partial = new float[BLOCK_SIZE * GRID_SIZE];

    // Инициализация данных
    cout << "Инициализация массива..." << endl;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;  // Простое значение для проверки (сумма = N)
    }

    // ==================== CPU ВЫЧИСЛЕНИЯ ====================
    cout << "\n--- Вычисления на CPU ---" << endl;
    auto start_cpu = high_resolution_clock::now();
    float sum_cpu_result = sum_cpu(h_input, N);
    auto end_cpu = high_resolution_clock::now();
    auto duration_cpu = duration_cast<microseconds>(end_cpu - start_cpu);

    cout << "Результат CPU: " << sum_cpu_result << endl;
    cout << "Время выполнения CPU: " << duration_cpu.count() << " мкс" << endl;

    // ==================== GPU ВЫЧИСЛЕНИЯ (метод 1: глобальная память) ====================
    cout << "\n--- Вычисления на GPU (глобальная память) ---" << endl;

    // Выделение памяти на устройстве
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, N * sizeof(float)), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, BLOCK_SIZE * GRID_SIZE * sizeof(float)),
                   "cudaMalloc d_output");

    // Копирование данных на устройство
    checkCudaError(cudaMemcpy(d_input, h_input, N * sizeof(float),
                   cudaMemcpyHostToDevice), "cudaMemcpy to device");

    // Создание событий для измерения времени
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Запуск kernel
    cudaEventRecord(start_gpu);
    sum_reduction_global<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    // Вычисление времени выполнения kernel
    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);

    // Копирование результатов обратно
    checkCudaError(cudaMemcpy(h_output_partial, d_output,
                   BLOCK_SIZE * GRID_SIZE * sizeof(float),
                   cudaMemcpyDeviceToHost), "cudaMemcpy to host");

    // Финальная редукция на CPU
    float sum_gpu_result = 0.0f;
    for (int i = 0; i < BLOCK_SIZE * GRID_SIZE; i++) {
        sum_gpu_result += h_output_partial[i];
    }

    cout << "Результат GPU (глобальная память): " << sum_gpu_result << endl;
    cout << "Время выполнения kernel: " << gpu_time_ms * 1000 << " мкс" << endl;

    // ==================== GPU ВЫЧИСЛЕНИЯ (метод 2: атомарные операции) ====================
    cout << "\n--- Вычисления на GPU (атомарные операции) ---" << endl;

    float *d_output_atomic;
    checkCudaError(cudaMalloc(&d_output_atomic, sizeof(float)),
                   "cudaMalloc d_output_atomic");
    checkCudaError(cudaMemset(d_output_atomic, 0, sizeof(float)),
                   "cudaMemset d_output_atomic");

    cudaEventRecord(start_gpu);
    sum_reduction_atomic<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_output_atomic, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time_atomic_ms;
    cudaEventElapsedTime(&gpu_time_atomic_ms, start_gpu, stop_gpu);

    float sum_gpu_atomic_result;
    checkCudaError(cudaMemcpy(&sum_gpu_atomic_result, d_output_atomic,
                   sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy atomic result");

    cout << "Результат GPU (атомарные операции): " << sum_gpu_atomic_result << endl;
    cout << "Время выполнения kernel: " << gpu_time_atomic_ms * 1000 << " мкс" << endl;

    // ==================== ПРОВЕРКА КОРРЕКТНОСТИ ====================
    cout << "\n--- Проверка корректности ---" << endl;
    float expected = N * 1.0f;
    float epsilon = 1e-2;

    bool correct_global = fabs(sum_gpu_result - expected) < epsilon;
    bool correct_atomic = fabs(sum_gpu_atomic_result - expected) < epsilon;
    bool correct_cpu = fabs(sum_cpu_result - expected) < epsilon;

    cout << "Ожидаемый результат: " << expected << endl;
    cout << "CPU корректен: " << (correct_cpu ? "✓" : "✗") << endl;
    cout << "GPU (глобальная память) корректен: " << (correct_global ? "✓" : "✗") << endl;
    cout << "GPU (атомарные операции) корректен: " << (correct_atomic ? "✓" : "✗") << endl;

    // ==================== СТАТИСТИКА ====================
    cout << "\n=== Статистика производительности ===" << endl;
    cout << "Размер массива: " << N << " элементов" << endl;
    cout << "Размер данных: " << (N * sizeof(float)) / 1024.0 << " KB" << endl;
    cout << "\nВремя выполнения:" << endl;
    cout << "  CPU: " << duration_cpu.count() << " мкс" << endl;
    cout << "  GPU (глобальная память): " << gpu_time_ms * 1000 << " мкс" << endl;
    cout << "  GPU (атомарные операции): " << gpu_time_atomic_ms * 1000 << " мкс" << endl;

    double speedup_global = static_cast<double>(duration_cpu.count()) / (gpu_time_ms * 1000);
    double speedup_atomic = static_cast<double>(duration_cpu.count()) / (gpu_time_atomic_ms * 1000);

    cout << "\nУскорение:" << endl;
    cout << "  GPU (глобальная память): " << speedup_global << "x" << endl;
    cout << "  GPU (атомарные операции): " << speedup_atomic << "x" << endl;

    // Пропускная способность
    double bandwidth_cpu = (N * sizeof(float)) / (duration_cpu.count() * 1e-6) / 1e9; // GB/s
    double bandwidth_gpu_global = (N * sizeof(float)) / (gpu_time_ms * 1e-3) / 1e9;
    double bandwidth_gpu_atomic = (N * sizeof(float)) / (gpu_time_atomic_ms * 1e-3) / 1e9;

    cout << "\nПропускная способность:" << endl;
    cout << "  CPU: " << bandwidth_cpu << " GB/s" << endl;
    cout << "  GPU (глобальная память): " << bandwidth_gpu_global << " GB/s" << endl;
    cout << "  GPU (атомарные операции): " << bandwidth_gpu_atomic << " GB/s" << endl;

    // Информация об устройстве
    cout << "\n=== Информация об устройстве ===" << endl;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Устройство: " << prop.name << endl;
    cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "Глобальная память: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    cout << "Количество SM: " << prop.multiProcessorCount << endl;
    cout << "Max потоков на блок: " << prop.maxThreadsPerBlock << endl;

    // ==================== ОЧИСТКА ====================
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_atomic);
    delete[] h_input;
    delete[] h_output_partial;

    cout << "\nПрограмма успешно завершена." << endl;

    return 0;
}
