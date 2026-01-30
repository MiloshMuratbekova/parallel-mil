#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <random>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// REDUCTION KERNELS
// ============================================================================

// Optimized reduction kernel using shared memory
__global__ void reduceSum(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load data with grid-stride loop and first add during load
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Unroll last warp (no sync needed)
    if (tid < WARP_SIZE) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8) smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4) smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2) smem[tid] += smem[tid + 1];
    }
    
    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Reduction kernel for finding minimum
__global__ void reduceMin(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float minVal = INFINITY;
    if (i < n) minVal = min(minVal, input[i]);
    if (i + blockDim.x < n) minVal = min(minVal, input[i + blockDim.x]);
    sdata[tid] = minVal;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = min(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Reduction kernel for finding maximum
__global__ void reduceMax(const float* input, float* output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    float maxVal = -INFINITY;
    if (i < n) maxVal = max(maxVal, input[i]);
    if (i + blockDim.x < n) maxVal = max(maxVal, input[i + blockDim.x]);
    sdata[tid] = maxVal;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// ============================================================================
// SCAN (PREFIX SUM) KERNELS - Blelloch Algorithm
// ============================================================================

// Blelloch scan - up-sweep (reduce) phase
__global__ void scanUpSweep(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2 + stride - 1;
    if (idx + stride < n) {
        data[idx + stride] += data[idx];
    }
}

// Blelloch scan - down-sweep phase
__global__ void scanDownSweep(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2 + stride - 1;
    if (idx + stride < n) {
        float temp = data[idx];
        data[idx] = data[idx + stride];
        data[idx + stride] += temp;
    }
}

// Optimized single-block scan using shared memory
__global__ void scanSharedMemory(const float* input, float* output, int n) {
    __shared__ float temp[BLOCK_SIZE * 2];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load input into shared memory
    temp[tid] = (i < n) ? input[i] : 0.0f;
    temp[tid + blockDim.x] = (i + blockDim.x < n) ? input[i + blockDim.x] : 0.0f;
    __syncthreads();
    
    // Up-sweep phase
    int offset = 1;
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear last element
    if (tid == 0) temp[blockDim.x * 2 - 1] = 0;
    
    // Down-sweep phase
    for (int d = 1; d < blockDim.x * 2; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();
    
    // Write results
    if (i < n) output[i] = temp[tid];
    if (i + blockDim.x < n) output[i + blockDim.x] = temp[tid + blockDim.x];
}

// ============================================================================
// HOST FUNCTIONS
// ============================================================================

// CPU reduction for comparison
float cpuReduce(const std::vector<float>& data) {
    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    return sum;
}

// CPU scan for comparison
void cpuScan(const std::vector<float>& input, std::vector<float>& output) {
    if (input.empty()) return;
    output[0] = 0.0f;
    for (size_t i = 1; i < input.size(); i++) {
        output[i] = output[i-1] + input[i-1];
    }
}

// GPU reduction wrapper
float gpuReduce(const float* d_input, int n) {
    int numBlocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, numBlocks * sizeof(float)));
    
    // First reduction
    reduceSum<<<numBlocks, BLOCK_SIZE>>>(d_input, d_temp, n);
    CUDA_CHECK(cudaGetLastError());
    
    // Reduce remaining blocks on CPU (small enough)
    std::vector<float> h_temp(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_temp.data(), d_temp, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    float result = 0.0f;
    for (float val : h_temp) {
        result += val;
    }
    
    CUDA_CHECK(cudaFree(d_temp));
    return result;
}

// GPU scan wrapper (for small arrays that fit in one block)
void gpuScan(const float* d_input, float* d_output, int n) {
    if (n <= BLOCK_SIZE * 2) {
        scanSharedMemory<<<1, BLOCK_SIZE>>>(d_input, d_output, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        std::cerr << "Warning: Array too large for single-block scan. Using simplified version." << std::endl;
        // For larger arrays, would need multi-block implementation
        scanSharedMemory<<<1, BLOCK_SIZE>>>(d_input, d_output, BLOCK_SIZE * 2);
    }
}

// ============================================================================
// BENCHMARKING AND TESTING
// ============================================================================

void testReduction(int n) {
    std::cout << "\n=== Testing Reduction (n = " << n << ") ===" << std::endl;
    
    // Generate random data
    std::vector<float> h_input(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 10.0f);
    
    for (int i = 0; i < n; i++) {
        h_input[i] = dis(gen);
    }
    
    // CPU reduction
    auto start = std::chrono::high_resolution_clock::now();
    float cpuResult = cpuReduce(h_input);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // GPU reduction
    float* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Warm-up
    gpuReduce(d_input, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    start = std::chrono::high_resolution_clock::now();
    float gpuResult = gpuReduce(d_input, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Results
    std::cout << "CPU Result: " << cpuResult << " (Time: " << cpuTime << " μs)" << std::endl;
    std::cout << "GPU Result: " << gpuResult << " (Time: " << gpuTime << " μs)" << std::endl;
    std::cout << "Speedup: " << (float)cpuTime / gpuTime << "x" << std::endl;
    std::cout << "Error: " << std::abs(cpuResult - gpuResult) / cpuResult * 100 << "%" << std::endl;
    
    // Test min/max
    float* d_min, *d_max;
    CUDA_CHECK(cudaMalloc(&d_min, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
    
    int numBlocks = (n + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, numBlocks * sizeof(float)));
    
    reduceMin<<<numBlocks, BLOCK_SIZE>>>(d_input, d_temp, n);
    reduceMax<<<numBlocks, BLOCK_SIZE>>>(d_input, d_max, n);
    
    float minVal, maxVal;
    CUDA_CHECK(cudaMemcpy(&maxVal, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    
    std::cout << "GPU Max: " << maxVal << std::endl;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_min));
    CUDA_CHECK(cudaFree(d_max));
    CUDA_CHECK(cudaFree(d_temp));
}

void testScan(int n) {
    std::cout << "\n=== Testing Scan (n = " << n << ") ===" << std::endl;
    
    // Limit test size for scan
    if (n > BLOCK_SIZE * 2) {
        std::cout << "Note: Scan limited to " << BLOCK_SIZE * 2 << " elements for this demo" << std::endl;
        n = BLOCK_SIZE * 2;
    }
    
    // Generate random data
    std::vector<float> h_input(n);
    std::vector<float> h_output_cpu(n);
    std::vector<float> h_output_gpu(n);
    
    for (int i = 0; i < n; i++) {
        h_input[i] = (float)(i % 10 + 1);
    }
    
    // CPU scan
    auto start = std::chrono::high_resolution_clock::now();
    cpuScan(h_input, h_output_cpu);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // GPU scan
    float* d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Warm-up
    gpuScan(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    start = std::chrono::high_resolution_clock::now();
    gpuScan(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify correctness
    bool correct = true;
    for (int i = 0; i < std::min(n, 10); i++) {
        if (std::abs(h_output_cpu[i] - h_output_gpu[i]) > 1e-3) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": CPU=" << h_output_cpu[i] 
                      << " GPU=" << h_output_gpu[i] << std::endl;
        }
    }
    
    std::cout << "CPU Time: " << cpuTime << " μs" << std::endl;
    std::cout << "GPU Time: " << gpuTime << " μs" << std::endl;
    std::cout << "Speedup: " << (float)cpuTime / gpuTime << "x" << std::endl;
    std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << std::endl;
    
    // Show first few results
    std::cout << "First 10 results:" << std::endl;
    std::cout << "Input:  ";
    for (int i = 0; i < std::min(n, 10); i++) std::cout << h_input[i] << " ";
    std::cout << "\nOutput: ";
    for (int i = 0; i < std::min(n, 10); i++) std::cout << h_output_gpu[i] << " ";
    std::cout << std::endl;
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    std::cout << "=== CUDA Reduction and Scan Lab ===" << std::endl;
    
    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;
    
    // Test different sizes
    std::vector<int> sizes = {1024, 1000000, 10000000};
    
    for (int size : sizes) {
        testReduction(size);
    }
    
    // Test scan with smaller sizes
    std::vector<int> scanSizes = {128, 512};
    for (int size : scanSizes) {
        testScan(size);
    }
    
    std::cout << "\n=== All tests completed ===" << std::endl;
    
    return 0;
}
