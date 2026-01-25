#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <cmath>

using namespace std;
using namespace std::chrono;

// Простая задача обработки: возведение в квадрат и умножение на коэффициент
// y[i] = (x[i] * x[i]) * 2.5 + sqrt(x[i])

// CPU функция обработки
void process_cpu(const float* input, float* output, int start, int end) {
    for (int i = start; i < end; i++) {
        output[i] = (input[i] * input[i]) * 2.5f + sqrtf(input[i]);
    }
}

// GPU kernel для обработки
__global__ void process_gpu(const float* input, float* output, int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        output[idx] = (input[idx] * input[idx]) * 2.5f + sqrtf(input[idx]);
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
bool verify_results(const float* expected, const float* actual, int n, float epsilon = 1e-3) {
    for (int i = 0; i < n; i++) {
        if (fabs(expected[i] - actual[i]) > epsilon) {
            cout << "Ошибка в элементе " << i << ": expected=" << expected[i]
                 << " actual=" << actual[i] << endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Параметры
    const int N = 10000000;  // 10 миллионов элементов
    const int BLOCK_SIZE = 256;

    cout << "=== Гибридные вычисления: CPU + GPU ===" << endl;
    cout << "Размер массива: " << N << " элементов" << endl;
    cout << "Размер данных: " << (N * sizeof(float)) / (1024.0 * 1024.0) << " MB" << endl;
    cout << "Операция: y[i] = (x[i]² * 2.5) + sqrt(x[i])" << endl;
    cout << "=========================================\n" << endl;

    // Выделение памяти на хосте
    float* h_input = new float[N];
    float* h_output_cpu_only = new float[N];
    float* h_output_gpu_only = new float[N];
    float* h_output_hybrid = new float[N];

    // Инициализация данных
    cout << "Инициализация массива..." << endl;
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i % 1000) / 10.0f;
    }

    // ==================== ВАРИАНТ 1: ТОЛЬКО CPU ====================
    cout << "\n--- Вариант 1: Только CPU ---" << endl;
    auto start_cpu = high_resolution_clock::now();

    process_cpu(h_input, h_output_cpu_only, 0, N);

    auto end_cpu = high_resolution_clock::now();
    auto duration_cpu = duration_cast<milliseconds>(end_cpu - start_cpu);

    cout << "Время выполнения (только CPU): " << duration_cpu.count() << " мс" << endl;

    // ==================== ВАРИАНТ 2: ТОЛЬКО GPU ====================
    cout << "\n--- Вариант 2: Только GPU ---" << endl;

    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, N * sizeof(float)), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, N * sizeof(float)), "cudaMalloc d_output");

    auto start_gpu = high_resolution_clock::now();

    // Копирование данных на GPU
    checkCudaError(cudaMemcpy(d_input, h_input, N * sizeof(float),
                   cudaMemcpyHostToDevice), "cudaMemcpy to device");

    // Запуск kernel
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    process_gpu<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, N, 0);
    checkCudaError(cudaGetLastError(), "kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Копирование результатов обратно
    checkCudaError(cudaMemcpy(h_output_gpu_only, d_output, N * sizeof(float),
                   cudaMemcpyDeviceToHost), "cudaMemcpy to host");

    auto end_gpu = high_resolution_clock::now();
    auto duration_gpu = duration_cast<milliseconds>(end_gpu - start_gpu);

    cout << "Время выполнения (только GPU): " << duration_gpu.count() << " мс" << endl;
    cout << "  - Включает передачу данных Host→Device→Host" << endl;

    // ==================== ВАРИАНТ 3: ГИБРИДНЫЙ CPU+GPU ====================
    cout << "\n--- Вариант 3: Гибридный (CPU + GPU) ---" << endl;

    // Определение распределения работы
    // Экспериментально подбираем оптимальное соотношение
    float cpu_ratio = 0.3f;  // 30% на CPU, 70% на GPU
    int cpu_elements = static_cast<int>(N * cpu_ratio);
    int gpu_elements = N - cpu_elements;

    cout << "Распределение работы:" << endl;
    cout << "  CPU: " << cpu_elements << " элементов (" << (cpu_ratio * 100) << "%)" << endl;
    cout << "  GPU: " << gpu_elements << " элементов (" << ((1 - cpu_ratio) * 100) << "%)" << endl;

    auto start_hybrid = high_resolution_clock::now();

    // Копирование части данных на GPU (асинхронно)
    checkCudaError(cudaMemcpy(d_input, h_input + cpu_elements,
                   gpu_elements * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy hybrid to device");

    // Создание потока для CPU обработки
    thread cpu_thread([&]() {
        process_cpu(h_input, h_output_hybrid, 0, cpu_elements);
    });

    // GPU обработка (параллельно с CPU)
    int gpu_grid_size = (gpu_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    process_gpu<<<gpu_grid_size, BLOCK_SIZE>>>(d_input, d_output, cpu_elements + gpu_elements, cpu_elements);
    checkCudaError(cudaGetLastError(), "hybrid kernel launch");

    // Ожидание завершения CPU
    cpu_thread.join();

    // Ожидание завершения GPU и копирование результатов
    checkCudaError(cudaDeviceSynchronize(), "hybrid sync");
    checkCudaError(cudaMemcpy(h_output_hybrid + cpu_elements, d_output + cpu_elements,
                   gpu_elements * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy hybrid to host");

    auto end_hybrid = high_resolution_clock::now();
    auto duration_hybrid = duration_cast<milliseconds>(end_hybrid - start_hybrid);

    cout << "Время выполнения (гибридный): " << duration_hybrid.count() << " мс" << endl;

    // ==================== ПРОВЕРКА КОРРЕКТНОСТИ ====================
    cout << "\n--- Проверка корректности ---" << endl;

    bool gpu_correct = verify_results(h_output_cpu_only, h_output_gpu_only, N);
    bool hybrid_correct = verify_results(h_output_cpu_only, h_output_hybrid, N);

    cout << "GPU результаты корректны: " << (gpu_correct ? "✓" : "✗") << endl;
    cout << "Гибридные результаты корректны: " << (hybrid_correct ? "✓" : "✗") << endl;

    // ==================== СТАТИСТИКА ====================
    cout << "\n=== Статистика производительности ===" << endl;

    cout << "\nВремя выполнения:" << endl;
    cout << "┌────────────────────────┬──────────────┬────────────────┐" << endl;
    cout << "│ Метод                  │ Время (мс)   │ Ускорение      │" << endl;
    cout << "├────────────────────────┼──────────────┼────────────────┤" << endl;

    double speedup_gpu = static_cast<double>(duration_cpu.count()) / duration_gpu.count();
    double speedup_hybrid = static_cast<double>(duration_cpu.count()) / duration_hybrid.count();

    printf("│ CPU только             │ %12ld │ %14.2fx │\n", duration_cpu.count(), 1.0);
    printf("│ GPU только             │ %12ld │ %14.2fx │\n", duration_gpu.count(), speedup_gpu);
    printf("│ Гибридный (CPU+GPU)    │ %12ld │ %14.2fx │\n", duration_hybrid.count(), speedup_hybrid);
    cout << "└────────────────────────┴──────────────┴────────────────┘" << endl;

    // Сравнение гибридного с GPU
    if (duration_hybrid.count() < duration_gpu.count()) {
        double improvement = ((double)duration_gpu.count() / duration_hybrid.count() - 1.0) * 100;
        cout << "\n✓ Гибридный метод на " << improvement << "% быстрее чем только GPU!" << endl;
    } else {
        double slowdown = ((double)duration_hybrid.count() / duration_gpu.count() - 1.0) * 100;
        cout << "\n✗ Гибридный метод на " << slowdown << "% медленнее чем только GPU" << endl;
    }

    // Пропускная способность
    double data_size_gb = (N * sizeof(float)) / 1e9;
    double throughput_cpu = data_size_gb / (duration_cpu.count() * 1e-3);
    double throughput_gpu = data_size_gb / (duration_gpu.count() * 1e-3);
    double throughput_hybrid = data_size_gb / (duration_hybrid.count() * 1e-3);

    cout << "\nПропускная способность (обработка данных):" << endl;
    printf("  CPU: %.2f GB/s\n", throughput_cpu);
    printf("  GPU: %.2f GB/s\n", throughput_gpu);
    printf("  Гибридный: %.2f GB/s\n", throughput_hybrid);

    // Эффективность использования ресурсов
    cout << "\n=== Анализ распределения нагрузки ===" << endl;

    // Оценка реального времени вычислений
    double cpu_compute_time = cpu_elements / (double)N * duration_cpu.count();
    double gpu_compute_time_estimate = gpu_elements / (double)N * (duration_gpu.count() * 0.7); // ~70% на вычисления

    cout << "Оценка времени вычислений:" << endl;
    printf("  CPU часть: ~%.1f мс\n", cpu_compute_time);
    printf("  GPU часть: ~%.1f мс\n", gpu_compute_time_estimate);

    double balance = cpu_compute_time / gpu_compute_time_estimate;
    cout << "\nБаланс нагрузки: " << balance << endl;
    if (balance > 0.9 && balance < 1.1) {
        cout << "✓ Отличный баланс! Оба устройства завершают работу примерно одновременно." << endl;
    } else if (balance < 0.9) {
        cout << "→ CPU завершает раньше. Можно увеличить cpu_ratio для лучшей загрузки." << endl;
    } else {
        cout << "→ GPU завершает раньше. Можно уменьшить cpu_ratio для лучшей загрузки." << endl;
    }

    // Рекомендации по оптимальному соотношению
    cout << "\n=== Рекомендации ===" << endl;
    cout << "Текущее соотношение: CPU=" << (cpu_ratio * 100) << "% / GPU=" << ((1 - cpu_ratio) * 100) << "%" << endl;

    // Простая эвристика для оптимального соотношения
    double cpu_speed = 1.0 / (duration_cpu.count() / (double)N);
    double gpu_speed = 1.0 / ((duration_gpu.count() * 0.7) / (double)N); // Исключаем overhead передачи
    double optimal_cpu_ratio = cpu_speed / (cpu_speed + gpu_speed);

    printf("Рекомендуемое соотношение: CPU=%.1f%% / GPU=%.1f%%\n",
           optimal_cpu_ratio * 100, (1 - optimal_cpu_ratio) * 100);

    // Информация об устройстве
    cout << "\n=== Информация об устройствах ===" << endl;
    cout << "CPU:" << endl;
    cout << "  Потоков: " << thread::hardware_concurrency() << endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "\nGPU:" << endl;
    cout << "  Устройство: " << prop.name << endl;
    cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "  SM: " << prop.multiProcessorCount << endl;
    cout << "  Глобальная память: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;

    // ==================== ЭКСПЕРИМЕНТ С РАЗНЫМИ СООТНОШЕНИЯМИ ====================
    cout << "\n=== Эксперимент: разные соотношения CPU/GPU ===" << endl;
    cout << "┌──────────────┬──────────────┬──────────────┐" << endl;
    cout << "│ CPU %        │ GPU %        │ Время (мс)   │" << endl;
    cout << "├──────────────┼──────────────┼──────────────┤" << endl;

    float ratios[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    for (float ratio : ratios) {
        int cpu_els = static_cast<int>(N * ratio);
        int gpu_els = N - cpu_els;

        auto t_start = high_resolution_clock::now();

        if (gpu_els > 0) {
            cudaMemcpy(d_input, h_input + cpu_els, gpu_els * sizeof(float),
                      cudaMemcpyHostToDevice);
        }

        thread t([&]() {
            if (cpu_els > 0) {
                process_cpu(h_input, h_output_hybrid, 0, cpu_els);
            }
        });

        if (gpu_els > 0) {
            int gs = (gpu_els + BLOCK_SIZE - 1) / BLOCK_SIZE;
            process_gpu<<<gs, BLOCK_SIZE>>>(d_input, d_output, cpu_els + gpu_els, cpu_els);
        }

        t.join();
        if (gpu_els > 0) {
            cudaDeviceSynchronize();
        }

        auto t_end = high_resolution_clock::now();
        auto t_dur = duration_cast<milliseconds>(t_end - t_start);

        printf("│ %12.0f │ %12.0f │ %12ld │\n",
               ratio * 100, (1 - ratio) * 100, t_dur.count());
    }
    cout << "└──────────────┴──────────────┴──────────────┘" << endl;

    // ==================== ОЧИСТКА ====================
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output_cpu_only;
    delete[] h_output_gpu_only;
    delete[] h_output_hybrid;

    cout << "\nПрограмма успешно завершена." << endl;

    return 0;
}
