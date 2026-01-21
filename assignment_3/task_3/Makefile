NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_50
TARGET = task3

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVCC_FLAGS) main.cu -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean
