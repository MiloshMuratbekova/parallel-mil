#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <CL/cl.h>

using namespace std;
using namespace std::chrono;

// Функция для последовательного умножения матриц на CPU
void matrix_multiply_cpu(const vector<float>& A, const vector<float>& B,
                         vector<float>& C, int N, int M, int K) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < M; k++) {
                sum += A[i * M + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// Функция для загрузки исходного кода OpenCL ядра из файла
string load_kernel(const char* filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть файл " << filename << endl;
        exit(1);
    }
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Функция для проверки ошибок OpenCL
void check_error(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        cerr << "Ошибка при выполнении " << operation << ": " << err << endl;
        exit(1);
    }
}

// Функция для сравнения результатов
bool compare_results(const vector<float>& C_cpu, const vector<float>& C_gpu,
                     int size, float epsilon = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (fabs(C_cpu[i] - C_gpu[i]) > epsilon) {
            cout << "Различие в элементе " << i << ": CPU=" << C_cpu[i]
                 << " GPU=" << C_gpu[i] << endl;
            return false;
        }
    }
    return true;
}

// Функция для вывода матрицы (для отладки маленьких матриц)
void print_matrix(const vector<float>& matrix, int rows, int cols, const char* name) {
    cout << "\nМатрица " << name << " (" << rows << "x" << cols << "):" << endl;
    for (int i = 0; i < rows && i < 10; i++) {  // Выводим максимум 10 строк
        for (int j = 0; j < cols && j < 10; j++) {  // Выводим максимум 10 столбцов
            cout << matrix[i * cols + j] << " ";
        }
        if (cols > 10) cout << "...";
        cout << endl;
    }
    if (rows > 10) cout << "...\n";
}

int main() {
    // Размеры матриц
    // A: N x M
    // B: M x K
    // C: N x K
    const int N = 512;  // Строки A и C
    const int M = 512;  // Столбцы A и строки B
    const int K = 512;  // Столбцы B и C

    cout << "=== Умножение матриц с использованием OpenCL ===" << endl;
    cout << "Размеры матриц:" << endl;
    cout << "  A: " << N << " x " << M << endl;
    cout << "  B: " << M << " x " << K << endl;
    cout << "  C: " << N << " x " << K << endl;
    cout << "================================================\n" << endl;

    // Выделение памяти для матриц
    vector<float> A(N * M);
    vector<float> B(M * K);
    vector<float> C_cpu(N * K);
    vector<float> C_gpu(N * K);

    // Инициализация матриц случайными значениями
    cout << "Инициализация матриц..." << endl;
    for (int i = 0; i < N * M; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * K; i++) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // ==================== CPU ВЫЧИСЛЕНИЯ ====================
    cout << "\n--- Вычисления на CPU ---" << endl;
    auto start_cpu = high_resolution_clock::now();
    matrix_multiply_cpu(A, B, C_cpu, N, M, K);
    auto end_cpu = high_resolution_clock::now();
    auto duration_cpu = duration_cast<milliseconds>(end_cpu - start_cpu);
    cout << "Время выполнения на CPU: " << duration_cpu.count() << " мс" << endl;

    // ==================== GPU ВЫЧИСЛЕНИЯ (OpenCL) ====================
    cout << "\n--- Вычисления на GPU (OpenCL) ---" << endl;

    cl_int err;

    // 1. Получение платформы и устройства
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "clGetPlatformIDs");

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        cout << "GPU не найден, используем CPU" << endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }
    check_error(err, "clGetDeviceIDs");

    // Вывод информации об устройстве
    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    cout << "Используется устройство: " << device_name << endl;

    // 2. Создание контекста
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "clCreateContext");

    // 3. Создание командной очереди
    cl_command_queue queue = clCreateCommandQueue(context, device,
                                                   CL_QUEUE_PROFILING_ENABLE, &err);
    check_error(err, "clCreateCommandQueue");

    // 4. Загрузка и компиляция ядра
    string kernel_source = load_kernel("kernel.cl");
    const char* source_str = kernel_source.c_str();
    size_t source_size = kernel_source.length();

    cl_program program = clCreateProgramWithSource(context, 1, &source_str,
                                                    &source_size, &err);
    check_error(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = new char[log_size];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        cerr << "Ошибка компиляции ядра:\n" << log << endl;
        delete[] log;
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", &err);
    check_error(err, "clCreateKernel");

    // 5. Создание буферов памяти
    cl_mem buffer_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     N * M * sizeof(float), A.data(), &err);
    check_error(err, "clCreateBuffer A");

    cl_mem buffer_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     M * K * sizeof(float), B.data(), &err);
    check_error(err, "clCreateBuffer B");

    cl_mem buffer_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     N * K * sizeof(float), NULL, &err);
    check_error(err, "clCreateBuffer C");

    // 6. Установка аргументов ядра
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_A);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_B);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_C);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &N);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &M);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &K);
    check_error(err, "clSetKernelArg");

    // 7. Настройка размеров рабочей группы
    size_t global_work_size[2] = {static_cast<size_t>(N), static_cast<size_t>(K)};
    size_t local_work_size[2] = {16, 16};  // Локальная рабочая группа 16x16

    cout << "Глобальная рабочая группа: " << global_work_size[0] << " x "
         << global_work_size[1] << endl;
    cout << "Локальная рабочая группа: " << local_work_size[0] << " x "
         << local_work_size[1] << endl;

    // 8. Запуск ядра
    auto start_gpu = high_resolution_clock::now();

    cl_event event;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size,
                                 local_work_size, 0, NULL, &event);
    check_error(err, "clEnqueueNDRangeKernel");

    // Ожидание завершения выполнения
    clWaitForEvents(1, &event);

    auto end_gpu = high_resolution_clock::now();
    auto duration_gpu = duration_cast<milliseconds>(end_gpu - start_gpu);

    // Получение точного времени выполнения ядра
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                           sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                           sizeof(time_end), &time_end, NULL);
    double kernel_time = (time_end - time_start) / 1e6;  // в миллисекундах

    // 9. Чтение результатов
    err = clEnqueueReadBuffer(queue, buffer_C, CL_TRUE, 0,
                             N * K * sizeof(float), C_gpu.data(), 0, NULL, NULL);
    check_error(err, "clEnqueueReadBuffer");

    cout << "Время выполнения на GPU: " << duration_gpu.count() << " мс" << endl;
    cout << "Время выполнения ядра: " << kernel_time << " мс" << endl;

    // ==================== СРАВНЕНИЕ РЕЗУЛЬТАТОВ ====================
    cout << "\n--- Проверка корректности результатов ---" << endl;
    bool results_match = compare_results(C_cpu, C_gpu, N * K);

    if (results_match) {
        cout << "✓ Результаты CPU и GPU совпадают!" << endl;
    } else {
        cout << "✗ Результаты CPU и GPU различаются!" << endl;
    }

    // ==================== СТАТИСТИКА ====================
    cout << "\n=== Статистика производительности ===" << endl;
    cout << "Время CPU: " << duration_cpu.count() << " мс" << endl;
    cout << "Время GPU (полное): " << duration_gpu.count() << " мс" << endl;
    cout << "Время GPU (только ядро): " << kernel_time << " мс" << endl;

    double speedup = static_cast<double>(duration_cpu.count()) / duration_gpu.count();
    cout << "Ускорение: " << speedup << "x" << endl;

    // Вычисление GFLOPS
    double flops = 2.0 * N * M * K;  // Каждый элемент требует M умножений и M сложений
    double gflops_cpu = flops / (duration_cpu.count() * 1e6);
    double gflops_gpu = flops / (kernel_time * 1e6);
    cout << "Производительность CPU: " << gflops_cpu << " GFLOPS" << endl;
    cout << "Производительность GPU: " << gflops_gpu << " GFLOPS" << endl;

    // Вывод фрагментов результата для проверки
    if (N <= 10 && M <= 10 && K <= 10) {
        print_matrix(A, N, M, "A");
        print_matrix(B, M, K, "B");
        print_matrix(C_cpu, N, K, "C (CPU)");
        print_matrix(C_gpu, N, K, "C (GPU)");
    }

    // ==================== ОЧИСТКА ====================
    clReleaseEvent(event);
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    cout << "\nПрограмма успешно завершена." << endl;

    return 0;
}
