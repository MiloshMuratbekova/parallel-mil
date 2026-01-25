# Makefile для гибридной CPU+GPU программы

# Компилятор CUDA
NVCC = nvcc

# Флаги компиляции
NVCCFLAGS = -std=c++11 -O3 -arch=sm_50

# Целевой файл
TARGET = hybrid_processing

# Исходные файлы
SOURCES = hybrid_processing.cu

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

# Помощь
help:
	@echo "Доступные команды:"
	@echo "  make        - Компиляция программы"
	@echo "  make run    - Компиляция и запуск программы"
	@echo "  make clean  - Удаление скомпилированных файлов"
	@echo "  make rebuild - Пересборка проекта"
	@echo "  make help   - Вывод этой справки"

.PHONY: all run clean rebuild help
