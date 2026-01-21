// OpenCL kernel для умножения матриц
// Матрица A: N x M
// Матрица B: M x K
// Матрица C: N x K (результат)

__kernel void matrix_multiply(__global const float* A,
                               __global const float* B,
                               __global float* C,
                               const int N,
                               const int M,
                               const int K) {
    // Получаем глобальные индексы текущего рабочего элемента
    int row = get_global_id(0);  // Строка в матрице C
    int col = get_global_id(1);  // Столбец в матрице C

    // Проверяем границы матрицы
    if (row < N && col < K) {
        float sum = 0.0f;

        // Вычисляем скалярное произведение строки A и столбца B
        for (int i = 0; i < M; i++) {
            sum += A[row * M + i] * B[i * K + col];
        }

        // Записываем результат в матрицу C
        C[row * K + col] = sum;
    }
}

// Оптимизированная версия с использованием локальной памяти
__kernel void matrix_multiply_local(__global const float* A,
                                    __global const float* B,
                                    __global float* C,
                                    const int N,
                                    const int M,
                                    const int K,
                                    __local float* A_local,
                                    __local float* B_local) {
    // Размер блока (рабочей группы)
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    int global_row = get_global_id(0);
    int global_col = get_global_id(1);

    int block_size = get_local_size(0);

    float sum = 0.0f;

    // Итерируем по блокам матриц
    int num_blocks = (M + block_size - 1) / block_size;

    for (int block = 0; block < num_blocks; block++) {
        // Загружаем блок матрицы A в локальную память
        int a_col = block * block_size + local_col;
        if (global_row < N && a_col < M) {
            A_local[local_row * block_size + local_col] = A[global_row * M + a_col];
        } else {
            A_local[local_row * block_size + local_col] = 0.0f;
        }

        // Загружаем блок матрицы B в локальную память
        int b_row = block * block_size + local_row;
        if (b_row < M && global_col < K) {
            B_local[local_row * block_size + local_col] = B[b_row * K + global_col];
        } else {
            B_local[local_row * block_size + local_col] = 0.0f;
        }

        // Синхронизация потоков в рабочей группе
        barrier(CLK_LOCAL_MEM_FENCE);

        // Вычисляем частичную сумму для текущего блока
        for (int i = 0; i < block_size; i++) {
            sum += A_local[local_row * block_size + i] * B_local[i * block_size + local_col];
        }

        // Синхронизация перед загрузкой следующего блока
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Записываем результат
    if (global_row < N && global_col < K) {
        C[global_row * K + global_col] = sum;
    }
}
