#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                 << cudaGetErrorString(error) << endl; \
            exit(1); \
        } \
    } while(0)

// Функция для создания массива случайных чисел
vector<int> createRandomArray(int size) {
    vector<int> arr(size);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 100000);

    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    return arr;
}

// CUDA kernel для сортировки подмассива (простая сортировка вставками)
__global__ void sortSubarrays(int* data, int n, int subarraySize) {
    // Каждый блок сортирует свой подмассив
    int blockStart = blockIdx.x * subarraySize;
    int blockEnd = min(blockStart + subarraySize, n);

    // Сортировка вставками для подмассива
    // Простая, но работает хорошо для небольших массивов
    for (int i = blockStart + threadIdx.x + 1; i < blockEnd; i += blockDim.x) {
        int key = data[i];
        int j = i - 1;

        // Двигаем элементы больше key на одну позицию вперед
        while (j >= blockStart && data[j] > key) {
            data[j + 1] = data[j];
            j--;
        }
        data[j + 1] = key;
    }
}

// CUDA kernel для слияния двух отсортированных подмассивов
__global__ void mergeSubarrays(int* data, int* temp, int n, int subarraySize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Каждый поток обрабатывает одну пару подмассивов
    int start1 = tid * 2 * subarraySize;
    if (start1 >= n) return;

    int end1 = min(start1 + subarraySize, n);
    int start2 = end1;
    int end2 = min(start2 + subarraySize, n);

    // Если второго подмассива нет, просто копируем первый
    if (start2 >= n) {
        for (int i = start1; i < end1; i++) {
            temp[i] = data[i];
        }
        return;
    }

    // Слияние двух подмассивов
    int i = start1;
    int j = start2;
    int k = start1;

    while (i < end1 && j < end2) {
        if (data[i] <= data[j]) {
            temp[k++] = data[i++];
        } else {
            temp[k++] = data[j++];
        }
    }

    // Копируем оставшиеся элементы
    while (i < end1) {
        temp[k++] = data[i++];
    }
    while (j < end2) {
        temp[k++] = data[j++];
    }
}

// Функция сортировки слиянием на GPU
void mergeSortGPU(vector<int>& arr) {
    int n = arr.size();

    // Выделяем память на GPU
    int* d_data;
    int* d_temp;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(int)));

    // Копируем данные на GPU
    CUDA_CHECK(cudaMemcpy(d_data, arr.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // Размер подмассива для каждого блока
    // Начинаем с небольших подмассивов
    int subarraySize = 256;

    // Количество блоков и потоков
    int numBlocks = (n + subarraySize - 1) / subarraySize;
    int threadsPerBlock = 256;

    // Шаг 1: Сортируем подмассивы на GPU
    sortSubarrays<<<numBlocks, threadsPerBlock>>>(d_data, n, subarraySize);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Шаг 2: Последовательно сливаем подмассивы
    // Размер подмассива удваивается на каждой итерации
    for (int currentSize = subarraySize; currentSize < n; currentSize *= 2) {
        int numMerges = (n + currentSize * 2 - 1) / (currentSize * 2);

        // Запускаем kernel слияния
        mergeSubarrays<<<numMerges, 1>>>(d_data, d_temp, n, currentSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Меняем местами data и temp
        swap(d_data, d_temp);
    }

    // Копируем результат обратно на CPU
    CUDA_CHECK(cudaMemcpy(arr.data(), d_data, n * sizeof(int), cudaMemcpyDeviceToHost));

    // Освобождаем память
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_temp));
}

// Обычная сортировка для сравнения (используем std::sort)
void sortCPU(vector<int>& arr) {
    sort(arr.begin(), arr.end());
}

// Проверка, отсортирован ли массив
bool isSorted(const vector<int>& arr) {
    for (int i = 0; i < arr.size() - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return false;
        }
    }
    return true;
}

// Функция для тестирования сортировки
void testSort(int array_size) {
    cout << "\n=============================================" << endl;
    cout << "Тестирование для массива размером " << array_size << endl;
    cout << "=============================================" << endl;

    // Создаем исходный массив
    cout << "Создаем массив..." << endl;
    vector<int> original = createRandomArray(array_size);

    // Копии для разных алгоритмов
    vector<int> arr_cpu = original;
    vector<int> arr_gpu = original;

    // ===== СОРТИРОВКА НА CPU =====
    cout << "\n--- Сортировка на CPU (std::sort) ---" << endl;

    auto start_cpu = chrono::high_resolution_clock::now();
    sortCPU(arr_cpu);
    auto end_cpu = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> time_cpu = end_cpu - start_cpu;

    cout << "Время выполнения: " << fixed << setprecision(4)
         << time_cpu.count() << " мс" << endl;
    cout << "Проверка сортировки: " << (isSorted(arr_cpu) ? "✓ УСПЕХ" : "✗ ОШИБКА") << endl;

    // ===== СОРТИРОВКА НА GPU =====
    cout << "\n--- Сортировка на GPU (CUDA) ---" << endl;

    // Получаем информацию о GPU
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "GPU: " << prop.name << endl;

    auto start_gpu = chrono::high_resolution_clock::now();
    mergeSortGPU(arr_gpu);
    auto end_gpu = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> time_gpu = end_gpu - start_gpu;

    cout << "Время выполнения: " << fixed << setprecision(4)
         << time_gpu.count() << " мс" << endl;
    cout << "Проверка сортировки: " << (isSorted(arr_gpu) ? "✓ УСПЕХ" : "✗ ОШИБКА") << endl;

    // ===== СРАВНЕНИЕ =====
    cout << "\n--- Результаты ---" << endl;

    // Проверяем, что результаты одинаковые
    bool results_match = (arr_cpu == arr_gpu);
    cout << "Результаты совпадают: " << (results_match ? "✓ ДА" : "✗ НЕТ") << endl;

    // Считаем ускорение
    double speedup = time_cpu.count() / time_gpu.count();
    cout << "Ускорение GPU относительно CPU: " << fixed << setprecision(2)
         << speedup << "x" << endl;

    if (speedup > 1.0) {
        cout << "GPU быстрее на "
             << fixed << setprecision(1) << ((speedup - 1.0) * 100) << "%" << endl;
    } else {
        cout << "CPU быстрее на "
             << fixed << setprecision(1) << ((1.0 / speedup - 1.0) * 100) << "%" << endl;
    }
}

int main() {
    cout << "=== ЗАДАЧА 4: Сортировка на GPU с использованием CUDA ===" << endl;

    // Проверяем наличие CUDA-устройства
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        cerr << "ОШИБКА: CUDA-устройство не найдено!" << endl;
        return 1;
    }

    cout << "Найдено CUDA-устройств: " << deviceCount << endl;

    // Тестируем на массивах разного размера
    testSort(10000);
    testSort(100000);

    // ===== ОБЩИЕ ВЫВОДЫ =====
    cout << "\n\n========================================" << endl;
    cout << "ОБЩИЕ ВЫВОДЫ" << endl;
    cout << "========================================" << endl;

    cout << "\n1. Реализация на CUDA:" << endl;
    cout << "   - Массив разделяется на подмассивы" << endl;
    cout << "   - Каждый блок GPU сортирует свой подмассив" << endl;
    cout << "   - Затем подмассивы сливаются параллельно" << endl;
    cout << "   - Используется два буфера для эффективного слияния" << endl;

    cout << "\n2. Производительность:" << endl;
    cout << "   - Для малых массивов (10K): накладные расходы могут быть большими" << endl;
    cout << "   - Для больших массивов (100K+): GPU показывает преимущество" << endl;
    cout << "   - Копирование данных CPU↔GPU занимает значительное время" << endl;
    cout << "   - std::sort очень оптимизирован, поэтому конкуренция серьезная" << endl;

    cout << "\n3. Архитектура решения:" << endl;
    cout << "   - sortSubarrays: сортирует небольшие блоки данных" << endl;
    cout << "   - mergeSubarrays: параллельно сливает отсортированные блоки" << endl;
    cout << "   - Используется сортировка вставками для малых подмассивов" << endl;
    cout << "   - Это эффективно для GPU, т.к. избегает сложной синхронизации" << endl;

    cout << "\n4. Ограничения и улучшения:" << endl;
    cout << "   - Текущая реализация проста и понятна для обучения" << endl;
    cout << "   - Можно улучшить используя shared memory" << endl;
    cout << "   - Можно оптимизировать размер блоков и количество потоков" << endl;
    cout << "   - Можно использовать библиотеки типа Thrust для production" << endl;

    cout << "\n5. Когда GPU эффективен:" << endl;
    cout << "   - Очень большие массивы (миллионы элементов)" << endl;
    cout << "   - Данные уже находятся на GPU" << endl;
    cout << "   - Множество независимых сортировок" << endl;
    cout << "   - Интеграция с другими GPU-вычислениями" << endl;

    cout << "\n6. Понятия CUDA:" << endl;
    cout << "   - Kernel: функция, выполняемая на GPU" << endl;
    cout << "   - Block: группа потоков, выполняющихся вместе" << endl;
    cout << "   - Thread: отдельный поток выполнения на GPU" << endl;
    cout << "   - cudaMemcpy: копирование данных между CPU и GPU" << endl;
    cout << "   - cudaDeviceSynchronize: ожидание завершения GPU" << endl;

    return 0;
}
