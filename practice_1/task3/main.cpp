#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <algorithm>

using namespace std;

// Функция для создания массива случайных чисел
vector<int> createRandomArray(int size) {
    vector<int> arr(size);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 10000);

    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    return arr;
}

// Последовательная сортировка выбором
void selectionSortSequential(vector<int>& arr) {
    int n = arr.size();

    // Проходим по каждому элементу
    for (int i = 0; i < n - 1; i++) {
        // Находим минимальный элемент в оставшейся части
        int min_index = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_index]) {
                min_index = j;
            }
        }

        // Меняем местами текущий элемент с минимальным
        if (min_index != i) {
            swap(arr[i], arr[min_index]);
        }
    }
}

// Параллельная сортировка выбором с OpenMP
void selectionSortParallel(vector<int>& arr) {
    int n = arr.size();

    // Внешний цикл нельзя распараллелить, т.к. каждая итерация
    // зависит от предыдущей
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        int global_min_index = i;
        int global_min_value = arr[i];

        // Параллельно ищем минимум в оставшейся части
        // Каждый поток ищет минимум в своем диапазоне
        #pragma omp parallel
        {
            int local_min_index = i;
            int local_min_value = arr[i];

            // Каждый поток проверяет свою часть массива
            #pragma omp for
            for (int j = i + 1; j < n; j++) {
                if (arr[j] < local_min_value) {
                    local_min_value = arr[j];
                    local_min_index = j;
                }
            }

            // Критическая секция - обновляем глобальный минимум
            #pragma omp critical
            {
                if (local_min_value < global_min_value) {
                    global_min_value = local_min_value;
                    global_min_index = local_min_index;
                }
            }
        }

        // Меняем местами
        if (global_min_index != i) {
            swap(arr[i], arr[global_min_index]);
        }
    }
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
    cout << "\n=====================================" << endl;
    cout << "Тестирование для массива размером " << array_size << endl;
    cout << "=====================================" << endl;

    // Создаем исходный массив
    cout << "Создаем массив..." << endl;
    vector<int> original = createRandomArray(array_size);

    // Копии для разных алгоритмов
    vector<int> arr_seq = original;
    vector<int> arr_par = original;

    // ===== ПОСЛЕДОВАТЕЛЬНАЯ СОРТИРОВКА =====
    cout << "\n--- Последовательная сортировка выбором ---" << endl;

    auto start_seq = chrono::high_resolution_clock::now();
    selectionSortSequential(arr_seq);
    auto end_seq = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> time_seq = end_seq - start_seq;

    cout << "Время выполнения: " << fixed << setprecision(4)
         << time_seq.count() << " мс" << endl;
    cout << "Проверка сортировки: " << (isSorted(arr_seq) ? "✓ УСПЕХ" : "✗ ОШИБКА") << endl;

    // ===== ПАРАЛЛЕЛЬНАЯ СОРТИРОВКА =====
    cout << "\n--- Параллельная сортировка выбором (OpenMP) ---" << endl;
    cout << "Количество потоков: " << omp_get_max_threads() << endl;

    auto start_par = chrono::high_resolution_clock::now();
    selectionSortParallel(arr_par);
    auto end_par = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> time_par = end_par - start_par;

    cout << "Время выполнения: " << fixed << setprecision(4)
         << time_par.count() << " мс" << endl;
    cout << "Проверка сортировки: " << (isSorted(arr_par) ? "✓ УСПЕХ" : "✗ ОШИБКА") << endl;

    // ===== СРАВНЕНИЕ =====
    cout << "\n--- Результаты ---" << endl;

    // Проверяем, что результаты одинаковые
    bool results_match = (arr_seq == arr_par);
    cout << "Результаты совпадают: " << (results_match ? "✓ ДА" : "✗ НЕТ") << endl;

    // Считаем ускорение
    double speedup = time_seq.count() / time_par.count();
    cout << "Ускорение: " << fixed << setprecision(2) << speedup << "x" << endl;

    if (speedup > 1.0) {
        cout << "Параллельная версия быстрее на "
             << fixed << setprecision(1) << ((speedup - 1.0) * 100) << "%" << endl;
    } else {
        cout << "Последовательная версия быстрее на "
             << fixed << setprecision(1) << ((1.0 / speedup - 1.0) * 100) << "%" << endl;
    }
}

int main() {
    cout << "=== ЗАДАЧА 3: Параллельная сортировка с OpenMP ===" << endl;

    // Тестируем на массивах разного размера
    testSort(1000);
    testSort(10000);

    // ===== ОБЩИЕ ВЫВОДЫ =====
    cout << "\n\n========================================" << endl;
    cout << "ОБЩИЕ ВЫВОДЫ" << endl;
    cout << "========================================" << endl;

    cout << "\n1. Алгоритм сортировки выбором:" << endl;
    cout << "   - Имеет сложность O(n²)" << endl;
    cout << "   - Внешний цикл последовательный (зависимости между итерациями)" << endl;
    cout << "   - Внутренний цикл можно распараллелить (поиск минимума)" << endl;

    cout << "\n2. Эффективность параллелизации:" << endl;
    cout << "   - Для n=1000: небольшое ускорение или замедление" << endl;
    cout << "   - Для n=10000: более заметный эффект" << endl;
    cout << "   - Накладные расходы на создание потоков существенны" << endl;

    cout << "\n3. Проблемы параллелизации сортировки выбором:" << endl;
    cout << "   - Внешний цикл остается последовательным" << endl;
    cout << "   - На каждой итерации создаются и уничтожаются потоки" << endl;
    cout << "   - Critical section создает узкое место (bottleneck)" << endl;
    cout << "   - Параллелится только поиск минимума, а не вся сортировка" << endl;

    cout << "\n4. Рекомендации:" << endl;
    cout << "   - Для параллельной сортировки лучше использовать другие алгоритмы" << endl;
    cout << "   - Например: сортировка слиянием, быстрая сортировка" << endl;
    cout << "   - Они лучше подходят для параллелизации" << endl;
    cout << "   - Сортировка выбором хороша для обучения, но не для production" << endl;

    cout << "\n5. OpenMP особенности:" << endl;
    cout << "   - Директива #pragma omp parallel создает команду потоков" << endl;
    cout << "   - Директива #pragma omp for распределяет итерации" << endl;
    cout << "   - Critical section синхронизирует доступ к общим данным" << endl;

    return 0;
}
