#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <ctime>
#include <cstdlib>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define ARRAY_SIZE 10000000

std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during operation '" << operation << "': " << err << std::endl;
        exit(1);
    }
}

void vectorAddCPU(const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    std::vector<float> A(ARRAY_SIZE);
    std::vector<float> B(ARRAY_SIZE);
    std::vector<float> C_cpu(ARRAY_SIZE);
    std::vector<float> C_gpu(ARRAY_SIZE);

    srand(42);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU execution
    clock_t cpu_start = clock();
    vectorAddCPU(A.data(), B.data(), C_cpu.data(), ARRAY_SIZE);
    clock_t cpu_end = clock();
    double cpu_time = static_cast<double>(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;

    // OpenCL setup
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Step 1: Get platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    checkError(err, "clGetPlatformIDs");

    // Step 2: Get device (try GPU first, fallback to CPU)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "GPU not found, using CPU device" << std::endl;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        checkError(err, "clGetDeviceIDs");
    }

    char deviceName[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "Using device: " << deviceName << std::endl;

    // Step 3: Create context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    // Step 4: Create command queue
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    checkError(err, "clCreateCommandQueue");

    // Step 5: Load and build kernel
    std::string kernelSource = loadKernelSource("kernel.cl");
    const char* source = kernelSource.c_str();
    size_t sourceSize = kernelSource.size();

    program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, nullptr);
        std::cerr << "Build error:\n" << buildLog << std::endl;
        exit(1);
    }

    kernel = clCreateKernel(program, "vector_add", &err);
    checkError(err, "clCreateKernel");

    // Step 6: Create buffers
    size_t bufferSize = ARRAY_SIZE * sizeof(float);
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, A.data(), &err);
    checkError(err, "clCreateBuffer A");
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, B.data(), &err);
    checkError(err, "clCreateBuffer B");
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &err);
    checkError(err, "clCreateBuffer C");

    // Step 7: Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    checkError(err, "clSetKernelArg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    checkError(err, "clSetKernelArg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    checkError(err, "clSetKernelArg 2");

    // Step 8: Execute kernel with profiling
    size_t globalSize = ARRAY_SIZE;
    cl_event event;

    clock_t gpu_start = clock();
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, &event);
    checkError(err, "clEnqueueNDRangeKernel");

    clWaitForEvents(1, &event);
    clock_t gpu_end = clock();

    // Get precise GPU timing from profiling
    cl_ulong time_start, time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, nullptr);
    double gpu_kernel_time = (time_end - time_start) / 1000000.0; // Convert to ms

    double gpu_total_time = static_cast<double>(gpu_end - gpu_start) / CLOCKS_PER_SEC * 1000.0;

    // Step 9: Read results
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, bufferSize, C_gpu.data(), 0, nullptr, nullptr);
    checkError(err, "clEnqueueReadBuffer");

    // Verify results
    bool correct = true;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        if (std::abs(C_cpu[i] - C_gpu[i]) > 1e-5) {
            correct = false;
            std::cerr << "Mismatch at index " << i << ": CPU=" << C_cpu[i] << ", GPU=" << C_gpu[i] << std::endl;
            break;
        }
    }

    // Print results
    std::cout << "\n========== Results ==========" << std::endl;
    std::cout << "Array size: " << ARRAY_SIZE << " elements" << std::endl;
    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU kernel time: " << gpu_kernel_time << " ms" << std::endl;
    std::cout << "GPU total time (including overhead): " << gpu_total_time << " ms" << std::endl;
    std::cout << "Speedup (kernel only): " << cpu_time / gpu_kernel_time << "x" << std::endl;
    std::cout << "Results match: " << (correct ? "Yes" : "No") << std::endl;

    // Cleanup
    clReleaseEvent(event);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
