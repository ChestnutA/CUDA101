#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_math.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define NUM_ITERATIONS 10
#define NUM_TESTS 5
constexpr auto BANDWIDTH = 1.5f;
constexpr auto INV_BANDWIDTH_SQ = (1.f / (2.0f * BANDWIDTH * BANDWIDTH));

std::vector<float> readPointsFromCSV(const std::string& fileName) {
    std::vector<float> points;
    std::ifstream data(fileName);
    std::string line;
    while (std::getline(data, line)) {
        std::vector<float> point;
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            points.push_back(stod(cell));
        }
    }
    return points;
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

#define TILE_DIM 64
#define BLOCK_DIM 256

__global__ void NaiveMeanShift(float* shiftedPoints, const float* __restrict__ originalPoints, const unsigned numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float3 newPosition = make_float3(0.0, 0.0, 0.0);
    float totalWeight = 0.0;
    int it = idx * 3;
    if (idx < numPoints) {
        float x = shiftedPoints[it];
        float y = shiftedPoints[it + 1];
        float z = shiftedPoints[it + 2];
        float3 shiftedPoint = make_float3(x, y, z);

        for (int i = 0; i < numPoints; i++) {
            x = originalPoints[3 * i];
            y = originalPoints[3 * i + 1];
            z = originalPoints[3 * i + 2];
            float3 originalPoint = make_float3(x, y, z);
            float3 difference = shiftedPoint - originalPoint;
            float squaredDistance = dot(difference, difference);
            float weight = expf((-squaredDistance) / (2 * powf(BANDWIDTH, 2)));
            newPosition += originalPoint * weight;
            totalWeight += weight;
        }
        newPosition /= totalWeight;
        shiftedPoints[it] = newPosition.x;
        shiftedPoints[it + 1] = newPosition.y;
        shiftedPoints[it + 2] = newPosition.z;
    }
}
__global__ void TilingMeanShift(float* shiftedPoints,
    const float* __restrict__ originalPoints,
    unsigned numPoints) {
    extern __shared__ float3 sharedPoints[]; // 共享内存存储瓦片数据
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= numPoints) return;

    // 读取当前点（AoS布局：XYZ连续存储）
    const int pos = 3 * idx;
    float3 shiftedPoint = make_float3(
        shiftedPoints[pos],
        shiftedPoints[pos + 1],
        shiftedPoints[pos + 2]
    );

    float3 newPosition = make_float3(0.0f, 0.0f, 0.0f);
    float totalWeight = 0.0f;

    const int TILE_SIZE = TILE_DIM;
    const int numTiles = (numPoints + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        const int tileStart = tile * TILE_SIZE;
        const int tileEnd = min(tileStart + TILE_SIZE, numPoints);
        const int tilePointIdx = tileStart + tid;

        // 1. 加载瓦片数据到共享内存
        float3 point = make_float3(0.0f, 0.0f, 0.0f);
        if (tilePointIdx < numPoints) {
            const int tilePos = 3 * tilePointIdx;
            point.x = originalPoints[tilePos];
            point.y = originalPoints[tilePos + 1];
            point.z = originalPoints[tilePos + 2];
        }
        sharedPoints[tid] = point;
        __syncthreads();

        // 2. 处理当前瓦片
        const int validPoints = tileEnd - tileStart;
        for (int s = 0; s < validPoints; ++s) {
            const float3 originalPoint = sharedPoints[s];

            // 计算距离平方
            const float3 diff = shiftedPoint - originalPoint;
            const float squaredDist = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            const float weight = expf(-squaredDist * INV_BANDWIDTH_SQ);

            // 累加加权位置
            newPosition.x += originalPoint.x * weight;
            newPosition.y += originalPoint.y * weight;
            newPosition.z += originalPoint.z * weight;
            totalWeight += weight;
        }
        __syncthreads();
    }

    // 3. 写回结果
    if (totalWeight > 1e-7f) {
        shiftedPoints[pos] = newPosition.x / totalWeight;
        shiftedPoints[pos + 1] = newPosition.y / totalWeight;
        shiftedPoints[pos + 2] = newPosition.z / totalWeight;
    }
}

int main(void)
{

    std::string fileName = "datasets/3D_data_250000.csv";
    std::vector<float> inputPoints = readPointsFromCSV(fileName);

    int numPoints = inputPoints.size() / 3;
    std::cout << "Num points: " << numPoints << std::endl;
    std::cout.precision(9);

    float totalElapsedTime = 0.0;

    thrust::device_vector<float> originalPoints = inputPoints;
    thrust::device_vector<float> shiftedPoints = inputPoints;

    for (int j = 0; j < NUM_TESTS; j++) {
        originalPoints = inputPoints;
        shiftedPoints = inputPoints;

        float* originalPointer = thrust::raw_pointer_cast(&originalPoints[0]);
        float* shiftedPointer = thrust::raw_pointer_cast(&shiftedPoints[0]);

        dim3 tilingGrid((numPoints + BLOCK_DIM - 1) / BLOCK_DIM);
        dim3 tilingBlock(TILE_DIM, BLOCK_DIM / TILE_DIM);

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            TilingMeanShift << <tilingGrid, tilingBlock >> > (shiftedPointer, originalPointer, numPoints);
            cudaDeviceSynchronize();
        }

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float elapsedTime = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
        totalElapsedTime += elapsedTime;
    }
    totalElapsedTime /= NUM_TESTS;
    std::cout << "\nTiling Mean Shift elapsed time: " << totalElapsedTime << std::endl;

    totalElapsedTime = 0.0;

    for (int j = 0; j < NUM_TESTS; j++) {
        originalPoints = inputPoints;
        shiftedPoints = inputPoints;

        float* originalPointer = thrust::raw_pointer_cast(&originalPoints[0]);
        float* shiftedPointer = thrust::raw_pointer_cast(&shiftedPoints[0]);

        dim3 gridDim = dim3(ceil((float)numPoints / BLOCK_DIM));
        dim3 blockDim = dim3(BLOCK_DIM);

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            NaiveMeanShift << <gridDim, blockDim >> > (shiftedPointer, originalPointer, numPoints);
            cudaDeviceSynchronize();
        }

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        float elapsedTime = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
        totalElapsedTime += elapsedTime;
    }
    totalElapsedTime /= NUM_TESTS;
    std::cout << "\nNaive Mean Shift elapsed time: " << totalElapsedTime << std::endl;
    return 0;
}
