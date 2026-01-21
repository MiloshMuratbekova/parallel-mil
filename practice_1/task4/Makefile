# Makefile для сборки CUDA программы

NVCC = nvcc
NVCCFLAGS = -std=c++14 -O2 -arch=sm_50
TARGET = task4
SOURCE = main.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
