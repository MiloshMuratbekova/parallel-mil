# Makefile для CUDA программы вычисления суммы элементов массива

# Компилятор CUDA
NVCC = nvcc

# Флаги компиляции
NVCCFLAGS = -std=c++11 -O3 -arch=sm_50

# Целевой файл
TARGET = sum_reduction

# Исходные файлы
SOURCES = sum_reduction.cu

# Правило по умолчанию
all: $(TARGET)

# Компиляция
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCCFLAGS) -o $@ $^
	@echo "Компиляция завершена успешно!"

# Запуск
run: $(TARGET)
	@echo "Запуск программы..."
	./$(TARGET)

# Очистка
clean:
	rm -f $(TARGET)
	@echo "Очистка выполнена."

# Пересборка
rebuild: clean all

# Информация о CUDA
info:
	@echo "=== Информация о CUDA ==="
	@which nvcc || echo "nvcc не найден"
	@nvcc --version 2>/dev/null || echo "Не удалось получить версию nvcc"
	@nvidia-smi 2>/dev/null || echo "nvidia-smi не доступен"

# Помощь
help:
	@echo "Доступные команды:"
	@echo "  make        - Компиляция программы"
	@echo "  make run    - Компиляция и запуск программы"
	@echo "  make clean  - Удаление скомпилированных файлов"
	@echo "  make rebuild - Пересборка проекта"
	@echo "  make info   - Информация о CUDA"
	@echo "  make help   - Вывод этой справки"

.PHONY: all run clean rebuild info help
