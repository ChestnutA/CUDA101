
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <sstream>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main1()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
// 辅助函数：将布尔值转换为"Yes/No"
const char* boolToStr(int val) {
    return val ? "Yes" : "No";
}

// 辅助函数：将二进制UUID转换为字符串
std::string uuidToString(const cudaUUID_t& uuid) {
    std::stringstream ss;
    for (int i = 0; i < 16; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(uuid.bytes[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) ss << "-";
    }
    return ss.str();
}

void printDeviceProperties(const cudaDeviceProp& prop) {
    std::cout << "========================================\n";
    std::cout << "Device Name: " << prop.name << "\n";
    std::cout << "----------------------------------------\n";

    // 基础信息
    std::cout << "| Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "| UUID: " << uuidToString(prop.uuid) << "\n";
    std::cout << "| PCI Bus ID: " << prop.pciBusID << "\n";
    std::cout << "| PCI Device ID: " << prop.pciDeviceID << "\n";

    // 内存信息
    std::cout << "| Global Memory: " << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB\n";
    std::cout << "| Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "| Constant Memory: " << prop.totalConstMem / 1024 << " KB\n";
    std::cout << "| L2 Cache Size: " << prop.l2CacheSize / 1024 << " KB\n";

    // 计算单元
    std::cout << "| Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "| Max Threads per MP: " << prop.maxThreadsPerMultiProcessor << "\n";

    // 线程配置
    std::cout << "| Warp Size: " << prop.warpSize << "\n";
    std::cout << "| Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "| Max Block Dimensions: ["
        << prop.maxThreadsDim[0] << ", "
        << prop.maxThreadsDim[1] << ", "
        << prop.maxThreadsDim[2] << "]\n";
    std::cout << "| Max Grid Dimensions: ["
        << prop.maxGridSize[0] << ", "
        << prop.maxGridSize[1] << ", "
        << prop.maxGridSize[2] << "]\n";

    // 硬件特性
    std::cout << "| Concurrent Kernels: " << boolToStr(prop.concurrentKernels) << "\n";
    std::cout << "| ECC Enabled: " << boolToStr(prop.ECCEnabled) << "\n";
    std::cout << "| Unified Addressing: " << boolToStr(prop.unifiedAddressing) << "\n";
    std::cout << "| Managed Memory: " << boolToStr(prop.managedMemory) << "\n";

    // 高级功能支持
    std::cout << "| Cooperative Launch: " << boolToStr(prop.cooperativeLaunch) << "\n";
    std::cout << "| Compute Preemption: " << boolToStr(prop.computePreemptionSupported) << "\n";
    std::cout << "| Stream Priorities: " << boolToStr(prop.streamPrioritiesSupported) << "\n";

    // 纹理/表面限制
    std::cout << "| Max Texture 1D: " << prop.maxTexture1D << "\n";
    std::cout << "| Max Texture 2D: ["
        << prop.maxTexture2D[0] << ", "
        << prop.maxTexture2D[1] << "]\n";
    std::cout << "| Max Texture 3D: ["
        << prop.maxTexture3D[0] << ", "
        << prop.maxTexture3D[1] << ", "
        << prop.maxTexture3D[2] << "]\n";

    // 内存带宽信息
    std::cout << "| Memory Bus Width: " << prop.memoryBusWidth << "-bit\n";
    std::cout << "| Memory Clock Rate: " << prop.memoryClockRate / 1e6 << " GHz\n";
    std::cout << "| Theoretical Bandwidth: "
        << (2.0 * prop.memoryBusWidth / 8 * prop.memoryClockRate * 1e6) / 1e9
        << " GB/s\n";

    // 其他重要属性
    std::cout << "| Async Engine Count: " << prop.asyncEngineCount << "\n";
    std::cout << "| Pageable Memory Access: " << boolToStr(prop.pageableMemoryAccess) << "\n";
    std::cout << "| TCC Driver: " << boolToStr(prop.tccDriver) << "\n";
    std::cout << "========================================\n\n";
}

int main() {
    int dev_count;
    cudaError_t err = cudaGetDeviceCount(&dev_count);

    // 检查CUDA初始化是否成功
    if (err != cudaSuccess) {
        std::cerr << "CUDA初始化失败: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    if (dev_count == 0) {
        std::cerr << "未找到CUDA设备" << std::endl;
        return -1;
    }

    cudaDeviceProp dev_prop;
    std::cout << "找到 " << dev_count << " 个CUDA设备\n\n";

    for (int i = 0; i < dev_count; i++) {
        cudaGetDeviceProperties(&dev_prop, i);
            printDeviceProperties(dev_prop);
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
