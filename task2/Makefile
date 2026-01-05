# Makefile для сборки программы с OpenMP

CXX = g++
CXXFLAGS = -std=c++17 -fopenmp -O2 -Wall
TARGET = task2
SOURCE = main.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
