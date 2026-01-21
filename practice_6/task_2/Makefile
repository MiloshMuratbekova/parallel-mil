# Makefile для компиляции программы умножения матриц с OpenCL

# Компилятор и флаги
CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3
LDFLAGS = -framework OpenCL

# Имя исполняемого файла
TARGET = matrix_multiply

# Исходные файлы
SOURCES = main.cpp
KERNEL = kernel.cl

# Объектные файлы
OBJECTS = $(SOURCES:.cpp=.o)

# Правило по умолчанию
all: $(TARGET)

# Компиляция программы
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Компиляция завершена успешно!"

# Компиляция объектных файлов
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Запуск программы
run: $(TARGET)
	@echo "Запуск программы..."
	./$(TARGET)

# Очистка
clean:
	rm -f $(TARGET) $(OBJECTS)
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
