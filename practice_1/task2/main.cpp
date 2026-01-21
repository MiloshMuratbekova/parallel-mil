#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <iomanip>

using namespace std;

// Функция для создания массива случайных чисел
vector<int> createRandomArray(int size) {
    vector<int> arr(size);

    // Генератор случайных чисел
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 10000);

    // Заполняем массив
    for (int i = 0; i < size; i++) {
        arr[i] = dist(gen);
    }

    return arr;
}

// Последовательный поиск минимума и максимума
void findMinMaxSequential(const vector<int>& arr, int& min_val, int& max_val) {
    min_val = arr[0];
    max_val = arr[0];

    for (int i = 1; i < arr.size(); i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
}

// Параллельный поиск минимума и максимума с OpenMP
void findMinMaxParallel(const vector<int>& arr, int& min_val, int& max_val) {
    min_val = arr[0];
    max_val = arr[0];

    // reduction - каждый поток находит свой min и max,
    // потом они объединяются
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] < min_val) {
            min_val = arr[i];
        }
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
}

int main() {
    const int ARRAY_SIZE = 10000;

    cout << "=== ЗАДАЧА 2: Работа с массивами и OpenMP ===" << endl;
    cout << endl;

    // Создаем массив
    cout << "Создаем массив из " << ARRAY_SIZE << " случайных чисел..." << endl;
    vector<int> arr = createRandomArray(ARRAY_SIZE);
    cout << "Массив создан!" << endl;
    cout << endl;

    // Переменные для результатов
    int min_seq, max_seq;
    int min_par, max_par;

    // ===== ПОСЛЕДОВАТЕЛЬНАЯ РЕАЛИЗАЦИЯ =====
    cout << "--- Последовательная реализация ---" << endl;

    auto start_seq = chrono::high_resolution_clock::now();
    findMinMaxSequential(arr, min_seq, max_seq);
    auto end_seq = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> time_seq = end_seq - start_seq;

    cout << "Минимум: " << min_seq << endl;
    cout << "Максимум: " << max_seq << endl;
    cout << "Время выполнения: " << fixed << setprecision(6)
         << time_seq.count() << " мс" << endl;
    cout << endl;

    // ===== ПАРАЛЛЕЛЬНАЯ РЕАЛИЗАЦИЯ =====
    cout << "--- Параллельная реализация (OpenMP) ---" << endl;
    cout << "Количество потоков: " << omp_get_max_threads() << endl;

    auto start_par = chrono::high_resolution_clock::now();
    findMinMaxParallel(arr, min_par, max_par);
    auto end_par = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> time_par = end_par - start_par;

    cout << "Минимум: " << min_par << endl;
    cout << "Максимум: " << max_par << endl;
    cout << "Время выполнения: " << fixed << setprecision(6)
         << time_par.count() << " мс" << endl;
    cout << endl;

    // ===== СРАВНЕНИЕ =====
    cout << "--- Сравнение результатов ---" << endl;

    // Проверяем корректность
    if (min_seq == min_par && max_seq == max_par) {
        cout << "✓ Результаты совпадают - реализация корректна!" << endl;
    } else {
        cout << "✗ ОШИБКА: Результаты не совпадают!" << endl;
    }

    // Сравниваем время
    double speedup = time_seq.count() / time_par.count();
    cout << "Ускорение: " << fixed << setprecision(2) << speedup << "x" << endl;

    if (speedup > 1.0) {
        cout << "Параллельная версия быстрее!" << endl;
    } else if (speedup < 1.0) {
        cout << "Последовательная версия быстрее!" << endl;
    } else {
        cout << "Одинаковая производительность" << endl;
    }
    cout << endl;

    // ===== ВЫВОДЫ =====
    cout << "--- Выводы ---" << endl;
    cout << "1. Для массива из " << ARRAY_SIZE << " элементов:" << endl;

    if (speedup > 1.2) {
        cout << "   - Параллелизация дала существенное ускорение" << endl;
        cout << "   - OpenMP эффективно распределил работу между потоками" << endl;
    } else if (speedup > 0.8) {
        cout << "   - Ускорение небольшое из-за накладных расходов на создание потоков" << endl;
        cout << "   - Для небольших массивов overhead может быть существенным" << endl;
    } else {
        cout << "   - Последовательная версия быстрее" << endl;
        cout << "   - Накладные расходы на параллелизацию превышают выгоду" << endl;
    }

    cout << endl;
    cout << "2. OpenMP автоматически распределяет итерации цикла между потоками" << endl;
    cout << "3. Директива reduction автоматически объединяет результаты" << endl;
    cout << "4. Для больших массивов (миллионы элементов) эффект был бы сильнее" << endl;

    return 0;
}
