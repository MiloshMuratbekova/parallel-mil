#include <iostream>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Функция обработки данных
void process_data(const vector<float>& input, vector<float>& output) {
    for (size_t i = 0; i < input.size(); i++) {
        // Вычислительно интенсивная операция
        output[i] = sin(input[i]) * cos(input[i]) + sqrt(fabs(input[i]));
    }
}

int main(int argc, char** argv) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int N = 10000000;  // 10 миллионов элементов

    vector<float> data;
    vector<float> result;

    auto start_time = high_resolution_clock::now();

    // ==================== МАСТЕР ПРОЦЕСС ====================
    if (world_rank == 0) {
        cout << "=== Распределённые вычисления с MPI ===" << endl;
        cout << "Общее количество процессов: " << world_size << endl;
        cout << "Размер массива: " << N << " элементов" << endl;
        cout << "Размер данных: " << (N * sizeof(float)) / (1024.0 * 1024.0) << " MB" << endl;
        cout << "Операция: y[i] = sin(x[i]) * cos(x[i]) + sqrt(|x[i]|)" << endl;
        cout << "=======================================\n" << endl;

        // Инициализация данных
        data.resize(N);
        result.resize(N);

        cout << "Инициализация данных..." << endl;
        for (int i = 0; i < N; i++) {
            data[i] = static_cast<float>(i % 1000) / 10.0f;
        }
    }

    // ==================== РАСПРЕДЕЛЕНИЕ ДАННЫХ ====================

    // Вычисление размера чанка для каждого процесса
    int chunk_size = N / world_size;
    int remainder = N % world_size;

    // Размер данных для каждого процесса
    vector<int> send_counts(world_size);
    vector<int> displacements(world_size);

    int offset = 0;
    for (int i = 0; i < world_size; i++) {
        send_counts[i] = chunk_size + (i < remainder ? 1 : 0);
        displacements[i] = offset;
        offset += send_counts[i];
    }

    // Локальные данные процесса
    int local_size = send_counts[world_rank];
    vector<float> local_data(local_size);
    vector<float> local_result(local_size);

    if (world_rank == 0) {
        cout << "Распределение данных между процессами:" << endl;
        for (int i = 0; i < world_size; i++) {
            cout << "  Процесс " << i << ": " << send_counts[i]
                 << " элементов (смещение: " << displacements[i] << ")" << endl;
        }
        cout << endl;
    }

    // Разбросать данные между процессами
    auto scatter_start = high_resolution_clock::now();

    MPI_Scatterv(data.data(), send_counts.data(), displacements.data(),
                 MPI_FLOAT, local_data.data(), local_size, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    auto scatter_end = high_resolution_clock::now();
    auto scatter_time = duration_cast<milliseconds>(scatter_end - scatter_start);

    // ==================== ЛОКАЛЬНАЯ ОБРАБОТКА ====================

    auto compute_start = high_resolution_clock::now();

    // Каждый процесс обрабатывает свою часть данных
    process_data(local_data, local_result);

    auto compute_end = high_resolution_clock::now();
    auto compute_time = duration_cast<milliseconds>(compute_end - compute_start);

    // ==================== СБОР РЕЗУЛЬТАТОВ ====================

    auto gather_start = high_resolution_clock::now();

    // Собрать результаты от всех процессов
    MPI_Gatherv(local_result.data(), local_size, MPI_FLOAT,
                result.data(), send_counts.data(), displacements.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);

    auto gather_end = high_resolution_clock::now();
    auto gather_time = duration_cast<milliseconds>(gather_end - gather_start);

    auto end_time = high_resolution_clock::now();
    auto total_time = duration_cast<milliseconds>(end_time - start_time);

    // ==================== ВЫВОД СТАТИСТИКИ ====================

    // Собрать статистику от всех процессов
    long compute_time_ms = compute_time.count();
    vector<long> all_compute_times;

    if (world_rank == 0) {
        all_compute_times.resize(world_size);
    }

    MPI_Gather(&compute_time_ms, 1, MPI_LONG,
               all_compute_times.data(), 1, MPI_LONG,
               0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        cout << "=== Результаты выполнения ===" << endl;
        cout << "\nВремена обработки по процессам:" << endl;

        long min_time = all_compute_times[0];
        long max_time = all_compute_times[0];
        long sum_time = 0;

        for (int i = 0; i < world_size; i++) {
            cout << "  Процесс " << i << ": " << all_compute_times[i] << " мс" << endl;
            min_time = min(min_time, all_compute_times[i]);
            max_time = max(max_time, all_compute_times[i]);
            sum_time += all_compute_times[i];
        }

        double avg_time = sum_time / (double)world_size;
        double load_balance = min_time / (double)max_time;

        cout << "\nСтатистика вычислений:" << endl;
        cout << "  Минимальное время: " << min_time << " мс" << endl;
        cout << "  Максимальное время: " << max_time << " мс" << endl;
        cout << "  Среднее время: " << avg_time << " мс" << endl;
        cout << "  Баланс нагрузки: " << (load_balance * 100) << "%" << endl;

        cout << "\n=== Фазы выполнения ===" << endl;
        cout << "  Scatter (распределение): " << scatter_time.count() << " мс" << endl;
        cout << "  Compute (вычисления): " << max_time << " мс" << endl;
        cout << "  Gather (сбор): " << gather_time.count() << " мс" << endl;
        cout << "  Общее время: " << total_time.count() << " мс" << endl;

        // Вычисление эффективности
        cout << "\n=== Анализ производительности ===" << endl;

        double communication_time = scatter_time.count() + gather_time.count();
        double computation_time = max_time;
        double parallel_overhead = communication_time / total_time.count() * 100;

        cout << "  Время коммуникации: " << communication_time << " мс ("
             << parallel_overhead << "%)" << endl;
        cout << "  Время вычислений: " << computation_time << " мс ("
             << (100 - parallel_overhead) << "%)" << endl;

        // Проверка результатов (первые и последние элементы)
        cout << "\n=== Проверка результатов ===" << endl;
        cout << "Первые 10 элементов результата: ";
        for (int i = 0; i < 10 && i < N; i++) {
            cout << result[i] << " ";
        }
        cout << endl;

        cout << "Последние 5 элементов результата: ";
        for (int i = max(0, N - 5); i < N; i++) {
            cout << result[i] << " ";
        }
        cout << endl;

        // Пропускная способность
        double data_size_gb = (N * sizeof(float) * 2) / 1e9; // Read + Write
        double throughput = data_size_gb / (total_time.count() * 1e-3);
        cout << "\nПропускная способность: " << throughput << " GB/s" << endl;
    }

    // ==================== ПОСЛЕДОВАТЕЛЬНОЕ СРАВНЕНИЕ ====================

    if (world_rank == 0) {
        cout << "\n=== Последовательное выполнение для сравнения ===" << endl;

        vector<float> sequential_result(N);

        auto seq_start = high_resolution_clock::now();
        process_data(data, sequential_result);
        auto seq_end = high_resolution_clock::now();
        auto seq_time = duration_cast<milliseconds>(seq_end - seq_start);

        cout << "Время последовательного выполнения: " << seq_time.count() << " мс" << endl;

        // Вычисление ускорения
        double speedup = seq_time.count() / (double)total_time.count();
        double efficiency = speedup / world_size * 100;

        cout << "\n=== Метрики масштабируемости ===" << endl;
        cout << "  Ускорение (Speedup): " << speedup << "x" << endl;
        cout << "  Эффективность: " << efficiency << "%" << endl;
        cout << "  Идеальное ускорение: " << world_size << "x" << endl;
        cout << "  Parallel overhead: " << parallel_overhead << "%" << endl;

        // Проверка корректности
        bool correct = true;
        for (int i = 0; i < N && i < 100; i++) {  // Проверяем первые 100
            if (fabs(result[i] - sequential_result[i]) > 1e-5) {
                correct = false;
                cout << "Ошибка в элементе " << i << ": MPI=" << result[i]
                     << " Sequential=" << sequential_result[i] << endl;
                break;
            }
        }

        cout << "\nКорректность результатов: " << (correct ? "✓" : "✗") << endl;

        // Рекомендации
        cout << "\n=== Рекомендации ===" << endl;
        if (parallel_overhead > 30) {
            cout << "⚠ Высокий overhead коммуникации (" << parallel_overhead << "%)" << endl;
            cout << "  → Увеличьте размер задачи" << endl;
            cout << "  → Используйте меньше процессов" << endl;
        } else if (parallel_overhead < 10) {
            cout << "✓ Отличное соотношение вычислений к коммуникации" << endl;
            cout << "  → Можно увеличить количество процессов" << endl;
        } else {
            cout << "✓ Приемлемый overhead коммуникации" << endl;
        }

        if (efficiency > 80) {
            cout << "✓ Отличная эффективность параллелизма" << endl;
        } else if (efficiency > 50) {
            cout << "→ Хорошая эффективность, но есть потенциал для улучшения" << endl;
        } else {
            cout << "⚠ Низкая эффективность параллелизма" << endl;
            cout << "  → Проверьте баланс нагрузки" << endl;
            cout << "  → Оптимизируйте коммуникацию" << endl;
        }
    }

    MPI_Finalize();
    return 0;
}
