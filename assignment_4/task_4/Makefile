# Makefile для MPI программы

# Компилятор MPI
MPICXX = mpic++

# Флаги компиляции
CXXFLAGS = -std=c++11 -O3 -Wall

# Целевой файл
TARGET = mpi_processing

# Исходные файлы
SOURCES = mpi_processing.cpp

# Правило по умолчанию
all: $(TARGET)

# Компиляция
$(TARGET): $(SOURCES)
	$(MPICXX) $(CXXFLAGS) -o $@ $^
	@echo "Компиляция завершена успешно!"

# Запуск с 2 процессами
run2: $(TARGET)
	@echo "Запуск с 2 процессами..."
	mpirun -np 2 ./$(TARGET)

# Запуск с 4 процессами
run4: $(TARGET)
	@echo "Запуск с 4 процессами..."
	mpirun -np 4 ./$(TARGET)

# Запуск с 8 процессами
run8: $(TARGET)
	@echo "Запуск с 8 процессами..."
	mpirun -np 8 ./$(TARGET)

# Запуск по умолчанию (4 процесса)
run: run4

# Бенчмарк: запуск с разным количеством процессов
benchmark: $(TARGET)
	@echo "=== Бенчмарк с разным количеством процессов ==="
	@echo ""
	@echo ">>> 2 процесса:"
	@mpirun -np 2 ./$(TARGET)
	@echo ""
	@echo ">>> 4 процесса:"
	@mpirun -np 4 ./$(TARGET)
	@echo ""
	@echo ">>> 8 процессов:"
	@mpirun -np 8 ./$(TARGET)

# Очистка
clean:
	rm -f $(TARGET)
	@echo "Очистка выполнена."

# Пересборка
rebuild: clean all

# Проверка MPI
check:
	@echo "=== Проверка MPI ==="
	@which mpirun || echo "mpirun не найден"
	@which mpic++ || echo "mpic++ не найден"
	@mpirun --version 2>/dev/null || echo "Не удалось получить версию MPI"

# Помощь
help:
	@echo "Доступные команды:"
	@echo "  make          - Компиляция программы"
	@echo "  make run      - Запуск с 4 процессами (по умолчанию)"
	@echo "  make run2     - Запуск с 2 процессами"
	@echo "  make run4     - Запуск с 4 процессами"
	@echo "  make run8     - Запуск с 8 процессами"
	@echo "  make benchmark - Запуск бенчмарка (2, 4, 8 процессов)"
	@echo "  make clean    - Удаление скомпилированных файлов"
	@echo "  make rebuild  - Пересборка проекта"
	@echo "  make check    - Проверка установки MPI"
	@echo "  make help     - Вывод этой справки"

.PHONY: all run run2 run4 run8 benchmark clean rebuild check help
