#!/bin/bash

# Скрипт для запуска всех задач по очереди

echo "========================================="
echo "Запуск всех задач лабораторной работы"
echo "========================================="
echo ""

# Задача 1
echo "=== ЗАДАЧА 1: Теоретическое введение ==="
echo "Открываем файл с ответами..."
echo ""
cat task1/answer.txt
echo ""
echo "Нажмите Enter для продолжения..."
read

# Задача 2
echo ""
echo "=== ЗАДАЧА 2: Работа с массивами и OpenMP ==="
cd task2
if [ -f "Makefile" ]; then
    echo "Компилируем..."
    make clean > /dev/null 2>&1
    make
    if [ $? -eq 0 ]; then
        echo ""
        echo "Запускаем..."
        ./task2
    else
        echo "ОШИБКА: Не удалось скомпилировать task2"
    fi
else
    echo "ОШИБКА: Makefile не найден в task2"
fi
cd ..
echo ""
echo "Нажмите Enter для продолжения..."
read

# Задача 3
echo ""
echo "=== ЗАДАЧА 3: Параллельная сортировка с OpenMP ==="
cd task3
if [ -f "Makefile" ]; then
    echo "Компилируем..."
    make clean > /dev/null 2>&1
    make
    if [ $? -eq 0 ]; then
        echo ""
        echo "Запускаем..."
        ./task3
    else
        echo "ОШИБКА: Не удалось скомпилировать task3"
    fi
else
    echo "ОШИБКА: Makefile не найден в task3"
fi
cd ..
echo ""
echo "Нажмите Enter для продолжения..."
read

# Задача 4
echo ""
echo "=== ЗАДАЧА 4: Сортировка на GPU с CUDA ==="
echo "ВНИМАНИЕ: Для этой задачи необходима NVIDIA GPU и CUDA Toolkit"
echo "Если у вас нет GPU, эта задача будет пропущена"
echo ""

# Проверяем наличие nvcc
if command -v nvcc &> /dev/null; then
    cd task4
    if [ -f "Makefile" ]; then
        echo "Компилируем..."
        make clean > /dev/null 2>&1
        make
        if [ $? -eq 0 ]; then
            echo ""
            echo "Запускаем..."
            ./task4
        else
            echo "ОШИБКА: Не удалось скомпилировать task4"
        fi
    else
        echo "ОШИБКА: Makefile не найден в task4"
    fi
    cd ..
else
    echo "CUDA компилятор (nvcc) не найден. Задача 4 пропущена."
    echo "Для установки CUDA посетите: https://developer.nvidia.com/cuda-downloads"
fi

echo ""
echo "========================================="
echo "Все задачи выполнены!"
echo "========================================="
