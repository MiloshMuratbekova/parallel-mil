#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

#define BLOCK_SIZE 512

// Последовательная реализация prefix sum на CPU
void prefix_sum_cpu(const float* input, float* output, int n) {
    output[0] = input[0];
    for (int i = 1; i < n; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// CUDA kernel для prefix sum с использованием разделяемой памяти (Hillis-Steele алгоритм)
__global__ void prefix_sum_shared(float* input, float* output, int n) {
    __shared__ float temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Загрузка данных в разделяемую память
    if (idx < n) {
        temp[tid] = input[idx];
    } else {
        temp[tid] = 0.0f;
    }
    __syncthreads();

    // Up-sweep (reduce) фаза
    int offset = 1;
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Очистка последнего элемента
    if (tid == 0) {
        temp[blockDim.x - 1] = 0;
    }

    // Down-sweep фаза
    for (int d = 1; d < blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Запись результата
    if (idx < n) {
        output[idx] = temp[tid];
    }
}

// Более эффективный kernel (Blelloch scan algorithm)
__global__ void prefix_sum_blelloch(float* g_data, float* g_output, int n) {
    __shared__ float temp[BLOCK_SIZE * 2];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x * 2;
    int idx1 = blockStart + tid;
    int idx2 = blockStart + tid + blockDim.x;

    // Загрузка данных
    temp[tid] = (idx1 < n) ? g_data[idx1] : 0.0f;
    temp[tid + blockDim.x] = (idx2 < n) ? g_data[idx2] : 0.0f;
    __syncthreads();

    // Up-sweep (Build sum tree)
    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear last element
    if (tid == 0) {
        temp[BLOCK_SIZE * 2 - 1] = 0;
    }

    // Down-sweep (Traverse down tree)
    for (int d = 1; d < blockDim.x * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    // Write results
    if (idx1 < n) g_output[idx1] = temp[tid];
    if (idx2 < n) g_output[idx2] = temp[tid + blockDim.x];
}

// Наивная последовательная версия на GPU (для сравнения)
__global__ void prefix_sum_naive(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i <= idx; i++) {
            sum += input[i];
        }
        output[idx] = sum;
    }
}

// Функция проверки ошибок CUDA
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

// Проверка корректности результатов
bool verify_results(const float* cpu_result, const float* gpu_result, int n, float epsilon = 1e-2) {
    for (int i = 0; i < n; i++) {
        if (fabs(cpu_result[i] - gpu_result[i]) > epsilon) {
            cout << "Ошибка в элементе " << i << ": CPU=" << cpu_result[i]
                 << " GPU=" << gpu_result[i] << " diff=" << fabs(cpu_result[i] - gpu_result[i]) << endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Параметры
    const int N = 1000000;  // 1 миллион элементов
    const int BLOCK_SIZE_EXEC = 512;
    const int GRID_SIZE = (N + BLOCK_SIZE_EXEC - 1) / BLOCK_SIZE_EXEC;

    cout << "=== CUDA: Вычисление префиксной суммы (Prefix Sum / Scan) ===" << endl;
    cout << "Размер массива: " << N << " элементов" << endl;
    cout << "Размер блока: " << BLOCK_SIZE_EXEC << " потоков" << endl;
    cout << "Количество блоков: " << GRID_SIZE << endl;
    cout << "Размер данных: " << (N * sizeof(float)) / (1024.0 * 1024.0) << " MB" << endl;
    cout << "===============================================================\n" << endl;

    // Выделение памяти на хосте
    float* h_input = new float[N];
    float* h_output_cpu = new float[N];
    float* h_output_gpu = new float[N];

    // Инициализация данных (все элементы = 1, чтобы prefix sum[i] = i+1)
    cout << "Инициализация массива..." << endl;
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // ==================== CPU ВЫЧИСЛЕНИЯ ====================
    cout << "\n--- Вычисления на CPU ---" << endl;
    auto start_cpu = high_resolution_clock::now();
    prefix_sum_cpu(h_input, h_output_cpu, N);
    auto end_cpu = high_resolution_clock::now();
    auto duration_cpu = duration_cast<milliseconds>(end_cpu - start_cpu);

    cout << "Время выполнения CPU: " << duration_cpu.count() << " мс" << endl;

    // Вывод первых и последних элементов для проверки
    cout << "Первые 10 элементов (CPU): ";
    for (int i = 0; i < 10; i++) {
        cout << h_output_cpu[i] << " ";
    }
    cout << endl;
    cout << "Последние 5 элементов (CPU): ";
    for (int i = N - 5; i < N; i++) {
        cout << h_output_cpu[i] << " ";
    }
    cout << endl;

    // ==================== GPU ВЫЧИСЛЕНИЯ ====================
    cout << "\n--- Вычисления на GPU (разделяемая память - Blelloch) ---" << endl;

    // Выделение памяти на устройстве
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, N * sizeof(float)), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, N * sizeof(float)), "cudaMalloc d_output");

    // Копирование данных на устройство
    checkCudaError(cudaMemcpy(d_input, h_input, N * sizeof(float),
                   cudaMemcpyHostToDevice), "cudaMemcpy to device");

    // Создание событий для измерения времени
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // ВНИМАНИЕ: Для больших массивов Blelloch scan работает только внутри блоков
    // Для корректной работы на всем массиве нужна более сложная реализация с несколькими проходами
    // Здесь мы демонстрируем алгоритм на одном блоке

    // Для демонстрации используем меньший размер, который помещается в один блок
    int demo_size = BLOCK_SIZE_EXEC;
    cout << "\nВнимание: Blelloch scan демонстрируется на первых " << demo_size << " элементах" << endl;

    // Запуск Blelloch kernel (для одного блока)
    cudaEventRecord(start_gpu);
    prefix_sum_blelloch<<<1, BLOCK_SIZE_EXEC / 2>>>(d_input, d_output, demo_size);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time_blelloch_ms;
    cudaEventElapsedTime(&gpu_time_blelloch_ms, start_gpu, stop_gpu);

    // Копирование результатов обратно
    checkCudaError(cudaMemcpy(h_output_gpu, d_output, demo_size * sizeof(float),
                   cudaMemcpyDeviceToHost), "cudaMemcpy to host");

    cout << "Время выполнения GPU (Blelloch, " << demo_size << " элементов): "
         << gpu_time_blelloch_ms << " мс" << endl;

    cout << "Первые 10 элементов (GPU Blelloch): ";
    for (int i = 0; i < 10; i++) {
        cout << h_output_gpu[i] << " ";
    }
    cout << endl;

    // Проверка корректности для первого блока
    bool correct_blelloch = verify_results(h_output_cpu, h_output_gpu, demo_size);
    cout << "Blelloch scan корректен: " << (correct_blelloch ? "✓" : "✗") << endl;

    // ==================== НАИВНАЯ GPU ВЕРСИЯ ====================
    cout << "\n--- Вычисления на GPU (наивная версия) ---" << endl;

    // Для демонстрации используем меньший массив (иначе будет очень медленно)
    int naive_size = 10000;
    cout << "Наивная версия демонстрируется на первых " << naive_size << " элементах" << endl;

    cudaEventRecord(start_gpu);
    prefix_sum_naive<<<(naive_size + BLOCK_SIZE_EXEC - 1) / BLOCK_SIZE_EXEC, BLOCK_SIZE_EXEC>>>
        (d_input, d_output, naive_size);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float gpu_time_naive_ms;
    cudaEventElapsedTime(&gpu_time_naive_ms, start_gpu, stop_gpu);

    checkCudaError(cudaMemcpy(h_output_gpu, d_output, naive_size * sizeof(float),
                   cudaMemcpyDeviceToHost), "cudaMemcpy naive to host");

    cout << "Время выполнения GPU (наивная, " << naive_size << " элементов): "
         << gpu_time_naive_ms << " мс" << endl;

    bool correct_naive = verify_results(h_output_cpu, h_output_gpu, naive_size);
    cout << "Наивная версия корректна: " << (correct_naive ? "✓" : "✗") << endl;

    // ==================== СТАТИСТИКА ====================
    cout << "\n=== Статистика производительности ===" << endl;
    cout << "\nСравнение методов:" << endl;
    cout << "┌─────────────────────────────┬──────────────┬────────────┐" << endl;
    cout << "│ Метод                       │ Размер       │ Время (мс) │" << endl;
    cout << "├─────────────────────────────┼──────────────┼────────────┤" << endl;
    cout << "│ CPU (последовательно)       │ " << N << "    │ " << duration_cpu.count() << "      │" << endl;
    cout << "│ GPU Blelloch (shared mem)   │ " << demo_size << "         │ " << gpu_time_blelloch_ms << "      │" << endl;
    cout << "│ GPU Naive                   │ " << naive_size << "        │ " << gpu_time_naive_ms << "      │" << endl;
    cout << "└─────────────────────────────┴──────────────┴────────────┘" << endl;

    // Вычисление ускорения для сравнимых размеров
    double cpu_time_demo = (duration_cpu.count() * demo_size) / (double)N;
    double speedup_blelloch = cpu_time_demo / gpu_time_blelloch_ms;

    double cpu_time_naive = (duration_cpu.count() * naive_size) / (double)N;
    double speedup_naive = cpu_time_naive / gpu_time_naive_ms;

    cout << "\nУскорение (приблизительное):" << endl;
    cout << "  GPU Blelloch vs CPU: " << speedup_blelloch << "x" << endl;
    cout << "  GPU Naive vs CPU: " << speedup_naive << "x" << endl;

    // Пропускная способность
    double bandwidth_cpu = (N * sizeof(float) * 2) / (duration_cpu.count() * 1e-3) / 1e9; // GB/s
    double bandwidth_blelloch = (demo_size * sizeof(float) * 2) / (gpu_time_blelloch_ms * 1e-3) / 1e9;

    cout << "\nПропускная способность:" << endl;
    cout << "  CPU: " << bandwidth_cpu << " GB/s" << endl;
    cout << "  GPU Blelloch: " << bandwidth_blelloch << " GB/s" << endl;

    // Информация об устройстве
    cout << "\n=== Информация об устройстве ===" << endl;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Устройство: " << prop.name << endl;
    cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "Shared memory на блок: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "Max потоков на блок: " << prop.maxThreadsPerBlock << endl;

    cout << "\n=== Заметка о реализации ===" << endl;
    cout << "Полная реализация prefix sum для больших массивов требует:" << endl;
    cout << "1. Вычисление prefix sum внутри каждого блока" << endl;
    cout << "2. Сбор сумм блоков и вычисление их prefix sum" << endl;
    cout << "3. Добавление смещений блоков к элементам каждого блока" << endl;
    cout << "Это делается в 3 прохода (3-phase scan algorithm)" << endl;

    // ==================== ОЧИСТКА ====================
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output_cpu;
    delete[] h_output_gpu;

    cout << "\nПрограмма успешно завершена." << endl;

    return 0;
}
