#ifndef MEANSHIFT_CU
#define MEANSHIFT_CU

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#include "vector_functions.hpp"
#include "vector_types.h"
#include "helper_math.h"
#include "device_functions.h" 
#include "commonDefines.h"


#define MYASSERT(condition, ERROR) if (!(condition)) { printf("ERROR: %s \n", ERROR); return; }
#define rev_sqrt_two_pi 0.3989422804
#define rev_two_pi rev_sqrt_two_pi*rev_sqrt_two_pi
constexpr int BLOCK_SIZE = 256; // 推荐使用256线程/block

__device__ __host__ float gaussian_kernel(float dist2, float bandwidth) {
	const float rev_bandwidth = 1. / bandwidth;
	const float d2_frac_b2 = dist2 * rev_bandwidth * rev_bandwidth;
	float div = 1. / rev_two_pi * rev_bandwidth;
	float exp_ = div * expf(- 0.5 * d2_frac_b2);
	return exp_;
}

__global__ void cuda_MeanShift_SharedMemory_2D_optimized(float* X, const float* I, const float* originalPoints, const int N, const int dim) {
	__shared__ float tile[2 * TILE_WIDTH][2]; // 扩展共享内存以容纳两倍元素

	int tx = threadIdx.x;
	int row = blockIdx.x * blockDim.x + tx;

	float2 numerator = make_float2(0.0f, 0.0f);
	float denominator = 0.0f;
	int it = row * dim;

	for (int tile_i = 0; tile_i < (N - 1) / (2 * TILE_WIDTH) + 1; ++tile_i) {
		int row_t = tile_i * (2 * TILE_WIDTH) + tx;

		// 加载第一个元素到tile[tx]
		if (row_t < N) {
			tile[tx][0] = originalPoints[row_t * dim];
			tile[tx][1] = originalPoints[row_t * dim + 1];
		}
		else {
			tile[tx][0] = 0.0f;
			tile[tx][1] = 0.0f;
		}

		// 加载第二个元素到tile[tx + TILE_WIDTH]
		int row_t2 = row_t + TILE_WIDTH;
		if (row_t2 < N) {
			tile[tx + TILE_WIDTH][0] = originalPoints[row_t2 * dim];
			tile[tx + TILE_WIDTH][1] = originalPoints[row_t2 * dim + 1];
		}
		else {
			tile[tx + TILE_WIDTH][0] = 0.0f;
			tile[tx + TILE_WIDTH][1] = 0.0f;
		}

		__syncthreads();

		if (row < N) {
			float2 x_i = make_float2(I[it], I[it + 1]);

			// 完全展开循环，处理两倍元素
#pragma unroll
			for (int j = 0; j < 2 * TILE_WIDTH; ++j) {
				float2 x_j = make_float2(tile[j][0], tile[j][1]);
				float2 sub = x_i - x_j;
				float distance2 = sub.x * sub.x + sub.y * sub.y;
				float weight = gaussian_kernel(distance2, BW);
				numerator.x += x_j.x * weight;
				numerator.y += x_j.y * weight;
				denominator += weight;
			}
		}

		__syncthreads();
	}

	if (row < N && denominator != 0.0f) {
		X[it] = numerator.x / denominator;
		X[it + 1] = numerator.y / denominator;
	}
}

extern "C"
void cudaMeanShift_sharedMemory_2D_wrapper_optimized(float* X, const float* I, const float* originalPoints, const int N, const int vecDim, dim3 gridDim, dim3 blockDim) {
	cuda_MeanShift_SharedMemory_2D_optimized << <gridDim, blockDim >> > (X, I, originalPoints, N, vecDim);
}

__global__ void cuda_MeanShift_2D_Brent(float *X, const float *I, const float *originalPoints, int N, int dim) {
    extern __shared__ float2 shared_points[];

    int tx = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tx;

    float2 numerator = make_float2(0.0f, 0.0f);
    float denominator = 0.0f;

    float2 y_i;
    if (row < N) {
        y_i = make_float2(I[row * dim], I[row * dim + 1]);
    }

    const int ELEMENTS_PER_THREAD = 4; // 每个线程处理4个元素
    for (int j_start = 0; j_start < N; j_start += blockDim.x * ELEMENTS_PER_THREAD) {
        int j_end = min(j_start + blockDim.x * ELEMENTS_PER_THREAD, N);

        // 每个线程加载ELEMENTS_PER_THREAD个元素到共享内存
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int j = j_start + tx + i * blockDim.x;
            if (j < j_end) {
                shared_points[tx + i * blockDim.x] = make_float2(originalPoints[j * dim], originalPoints[j * dim + 1]);
            }
        }
        __syncthreads();

        // 处理共享内存中的块
        int num_valid = j_end - j_start;
        for (int k = 0; k < num_valid; k++) {
            float2 x_j = shared_points[k];
            float2 sub = y_i - x_j;
            float distance2 = sub.x * sub.x + sub.y * sub.y;
            float weight = expf(-distance2 / (2.0f * BW * BW));
            numerator.x += x_j.x * weight;
            numerator.y += x_j.y * weight;
            denominator += weight;
        }
        __syncthreads();
    }

    if (row < N && denominator > 1e-7f) {
        X[row * dim] = numerator.x / denominator;
        X[row * dim + 1] = numerator.y / denominator;
    }
}

__global__ void cuda_MeanShift_2D_optimized(float *X, const float *I, const float *originalPoints, const int N, const int dim) {
    extern __shared__ float2 shared_points[]; // 动态共享内存存储数据块

    int tx = threadIdx.x;
    int row = blockIdx.x * blockDim.x + tx;

    float2 numerator = make_float2(0.0f, 0.0f);
    float denominator = 0.0f;

    float2 y_i;
    if (row < N) {
        int it = row * dim;
        y_i = make_float2(I[it], I[it + 1]);
    }

    const int BLOCK_SIZE = blockDim.x;
    for (int j_start = 0; j_start < N; j_start += BLOCK_SIZE) {
        int j_end = min(j_start + BLOCK_SIZE, N);

        // 协作加载当前数据块到共享内存
        int j = j_start + tx;
        if (j < N) {
            shared_points[tx] = make_float2(originalPoints[j * dim], originalPoints[j * dim + 1]);
        } else {
            shared_points[tx] = make_float2(0.0f, 0.0f); // 填充无效数据
        }
        __syncthreads();

        // 处理共享内存中的当前块
        int num_valid = j_end - j_start;
        for (int k = 0; k < num_valid; ++k) {
            float2 x_j = shared_points[k];
            float2 sub = y_i - x_j;
            float distance2 = dot(sub, sub);
            float weight = gaussian_kernel(distance2, BW); // 假设BW为定义好的宏或常量
			numerator += x_j * weight; //accumulating
			denominator += weight;
        }
        __syncthreads();
    }

    if (row < N) {
        int it = row * dim;
        if (denominator > 1e-7f) { // 避免除以零
            X[it] = numerator.x / denominator;
            X[it + 1] = numerator.y / denominator;
        } else {
            X[it] = y_i.x;
            X[it + 1] = y_i.y;
        }
    }
}

extern "C"
void cudaMeanShift_2D_optimized_wrapper(float *X, const float *I, const float *originalPoints, const int N, const int vecDim, dim3 gridDim, dim3 blockDim) {
    size_t sharedMemSize = blockDim.x * sizeof(float2); // 动态共享内存大小
    cuda_MeanShift_2D_optimized<<<gridDim, blockDim, sharedMemSize>>>(X, I, originalPoints, N, vecDim);
}


// template <int BLOCK_SIZE>
// __global__ void cuda_MeanShift_2D_optimized(float *X, const float *I, const float *originalPoints, int N, int dim) {
//     // 每个block处理一个数据点
//     const int row = blockIdx.x;
//     if (row >= N) return;

//     // 使用float2类型优化内存访问
//     const float2 *originalPoints2 = reinterpret_cast<const float2*>(originalPoints);
//     const float2 *I2 = reinterpret_cast<const float2*>(I);
//     float2 *X2 = reinterpret_cast<float2*>(X);

//     // 共享内存分配 (每个线程的中间结果)
//     __shared__ float2 smem_numerator[BLOCK_SIZE];
//     __shared__ float smem_denominator[BLOCK_SIZE];

//     const float2 y_i = I2[row];
//     float2 thread_numerator = make_float2(0.0f, 0.0f);
//     float thread_denominator = 0.0f;

//     const int tid = threadIdx.x;

//     // 每个线程处理多个元素（Brent定理应用）
//     for (int j = tid; j < N; j += BLOCK_SIZE) {
//         const float2 x_j = originalPoints2[j];
//         const float2 diff = make_float2(y_i.x - x_j.x, y_i.y - x_j.y);
//         const float distance2 = diff.x*diff.x + diff.y*diff.y;
//         const float weight = gaussian_kernel(distance2, BW);
        
//         thread_numerator.x += x_j.x * weight;
//         thread_numerator.y += x_j.y * weight;
//         thread_denominator += weight;
//     }

//     // 存储到共享内存
//     smem_numerator[tid] = thread_numerator;
//     smem_denominator[tid] = thread_denominator;
//     __syncthreads();

//     // 归约优化：当s<32时无需同步
//     for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
//         if (tid < s) {
//             smem_numerator[tid].x += smem_numerator[tid + s].x;
//             smem_numerator[tid].y += smem_numerator[tid + s].y;
//             smem_denominator[tid] += smem_denominator[tid + s];
//         }
//         if (s > 32) __syncthreads();
//         else if (s <= 32) __syncwarp();
//     }

//     // 最终结果写入全局内存
//     if (tid == 0) {
//         const float denominator = smem_denominator[0];
//         if (denominator != 0.0f) {
//             X2[row] = make_float2(smem_numerator[0].x / denominator, 
//                                 smem_numerator[0].y / denominator);
//         } else {
//             X2[row] = y_i; // 保持原值以防除以零
//         }
//     }
// }

// extern "C"
// void cudaMeanShift_2D_optimized_wrapper(float *X, const float *I, const float *originalPoints, 
//                              int N, int vecDim, dim3 gridDim, dim3 blockDim) {
//     // 根据block尺寸选择模板实例
//     if (blockDim.x == 512) {
//         cuda_MeanShift_2D_optimized<512><<<N, 512, 0>>>(X, I, originalPoints, N, vecDim);
//     } else if (blockDim.x == 256) {
//         cuda_MeanShift_2D_optimized<256><<<N, 256, 0>>>(X, I, originalPoints, N, vecDim);
//     } else if (blockDim.x == 128) {
//         cuda_MeanShift_2D_optimized<128><<<N, 128, 0>>>(X, I, originalPoints, N, vecDim);
//     } else {
//         // 默认使用256线程/block
//         cuda_MeanShift_2D_optimized<256><<<N, 256, 0>>>(X, I, originalPoints, N, vecDim);
//     }
// }

// template <int BLOCK_SIZE>
// __global__ void cuda_MeanShift_2D_optimized(float* X, const float* I,
// 	const float* originalPoints,
// 	int N, int dim) {
// 	// 共享内存分别存储x/y分量和分母
// 	__shared__ float s_num_x[BLOCK_SIZE];
// 	__shared__ float s_num_y[BLOCK_SIZE];
// 	__shared__ float s_den[BLOCK_SIZE];

// 	const int tid = threadIdx.x;
// 	const int row = blockIdx.x;  // 每个block处理一个点

// 	// 加载当前点的y_i坐标
// 	float y_i_x = 0.0f, y_i_y = 0.0f;
// 	if (row < N) {
// 		y_i_x = I[row * dim];
// 		y_i_y = I[row * dim + 1];
// 	}

// 	// 每个线程的局部累加器
// 	float local_num_x = 0.0f, local_num_y = 0.0f;
// 	float local_den = 0.0f;

// 	// Brent定理：每个线程处理多个点（跨步访问）
// 	for (int j = tid; j < N; j += BLOCK_SIZE) {
// 		const float x_j_x = originalPoints[j * dim];
// 		const float x_j_y = originalPoints[j * dim + 1];

// 		// 计算差值
// 		const float dx = y_i_x - x_j_x;
// 		const float dy = y_i_y - x_j_y;
// 		const float d2 = dx * dx + dy * dy;
// 		const float w = gaussian_kernel(d2, BW);

// 		// 分量累加
// 		local_num_x += x_j_x * w;
// 		local_num_y += x_j_y * w;
// 		local_den += w;
// 	}

// 	// 存储到共享内存
// 	s_num_x[tid] = local_num_x;
// 	s_num_y[tid] = local_num_y;
// 	s_den[tid] = local_den;
// 	__syncthreads();

// 	// 块内归约（支持编译期展开）
// #pragma unroll
// 	for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
// 		if (tid < s) {
// 			s_num_x[tid] += s_num_x[tid + s];
// 			s_num_y[tid] += s_num_y[tid + s];
// 			s_den[tid] += s_den[tid + s];
// 		}
// 		__syncthreads();
// 	}

// 	// Warp级归约（无分支判断）
// 	if (tid < 32) {
// 		volatile float* vs_num_x = s_num_x;
// 		volatile float* vs_num_y = s_num_y;
// 		volatile float* vs_den = s_den;

// 		// 完全展开的warp归约
// 		if (BLOCK_SIZE >= 64) {
// 			vs_num_x[tid] += vs_num_x[tid + 32];
// 			vs_num_y[tid] += vs_num_y[tid + 32];
// 			vs_den[tid] += vs_den[tid + 32];
// 		}
// 		if (BLOCK_SIZE >= 32) {
// 			vs_num_x[tid] += vs_num_x[tid + 16];
// 			vs_num_y[tid] += vs_num_y[tid + 16];
// 			vs_den[tid] += vs_den[tid + 16];
// 		}
// 		if (BLOCK_SIZE >= 16) {
// 			vs_num_x[tid] += vs_num_x[tid + 8];
// 			vs_num_y[tid] += vs_num_y[tid + 8];
// 			vs_den[tid] += vs_den[tid + 8];
// 		}
// 		if (BLOCK_SIZE >= 8) {
// 			vs_num_x[tid] += vs_num_x[tid + 4];
// 			vs_num_y[tid] += vs_num_y[tid + 4];
// 			vs_den[tid] += vs_den[tid + 4];
// 		}
// 		if (BLOCK_SIZE >= 4) {
// 			vs_num_x[tid] += vs_num_x[tid + 2];
// 			vs_num_y[tid] += vs_num_y[tid + 2];
// 			vs_den[tid] += vs_den[tid + 2];
// 		}
// 		if (BLOCK_SIZE >= 2) {
// 			vs_num_x[tid] += vs_num_x[tid + 1];
// 			vs_num_y[tid] += vs_num_y[tid + 1];
// 			vs_den[tid] += vs_den[tid + 1];
// 		}
// 	}

// 	// 写回结果
// 	if (tid == 0 && row < N) {
// 		X[row * dim] = s_num_x[0] / s_den[0];
// 		X[row * dim + 1] = s_num_y[0] / s_den[0];
// 	}
// }

// // 调用封装函数（保持原始接口不变）
// extern "C"
// void cudaMeanShift_2D_optimized_wrapper(float* X, const float* I,
// 	const float* originalPoints,
// 	const int N, const int vecDim,
// 	dim3 gridDim, dim3 blockDim) {
// 	//assert(blockDim.x == BLOCK_SIZE);
// cuda_MeanShift_2D_optimized<BLOCK_SIZE> << <N, BLOCK_SIZE >> > (
// 	X, I, originalPoints, N, vecDim
// 	);
// }

__global__ void cuda_MeanShift_SharedMemory_2D(float *X, const float *I, const float * originalPoints, const int N, const int dim) {

	__shared__ float tile[TILE_WIDTH][2];

	// for each pixel
	int tx = threadIdx.x;
	int row = blockIdx.x*blockDim.x + tx;

	float2 numerator = make_float2(0.0, 0.0);
	float denominator = 0.0;
	int it = row * dim;

	for (int tile_i = 0; tile_i < (N - 1) / TILE_WIDTH + 1; ++tile_i) {
		//loading phase - each thread load something into shared memory
		int row_t = tile_i * TILE_WIDTH + tx;

		int index = row_t * dim;
		if (row_t < N) {
			tile[tx][0] = originalPoints[index];
			tile[tx][1] = originalPoints[index + 1];
		}
		else {
			tile[tx][0] = 0.0;
			tile[tx][1] = 0.0;
		}
		__syncthreads();
		//end of loading into shared memory

		if (row < N) // only the threads inside the bounds do some computation
		{
			float2 x_i = make_float2(I[it], I[it + 1]); //load input point

			//computing phase
			for (int j = 0; j < TILE_WIDTH; ++j) {
				float2 x_j = make_float2(tile[j][0], tile[j][1]); //from shared memory
				float2 sub = x_i - x_j;
				float distance2 = dot(sub, sub);
				float weight = gaussian_kernel(distance2, BW);
				numerator += x_j * weight; //accumulating
				denominator += weight;

			}
		}
		__syncthreads();
		//end of computing phase for tile_ij
	}

	if (row < N) {
		//storing
		numerator /= denominator;
		X[it] = numerator.x;
		X[it + 1] = numerator.y;
	}

}

extern "C"
void cudaMeanShift_sharedMemory_2D_wrapper(float *X, const float *I, const float * originalPoints, const int N, const int vecDim, dim3 gridDim, dim3 blockDim) {
	cuda_MeanShift_SharedMemory_2D <<<gridDim, blockDim >>> (X, I, originalPoints, N, vecDim);
}

__global__ void cuda_MeanShift_2D(float *X, const float *I, const float * originalPoints, const int N, const int dim) {

	// for every pixel
	int tx = threadIdx.x;
	int row = blockIdx.x*blockDim.x + tx;

	float2 numerator = make_float2(0.0, 0.0);
	float denominator = 0.0;

	int it = row * dim;
	float2 y_i;
	if (row < N) {
		y_i = make_float2(I[it], I[it + 1]); //load input point

			//computing mean shift
			for (int j = 0; j < N; ++j) {
				float2 x_j = make_float2(originalPoints[j*dim], originalPoints[j*dim + 1]); //from central gpu memory
				float2 sub = y_i - x_j;
				float distance2 = dot(sub, sub);
				float weight = gaussian_kernel(distance2, BW);
				numerator += x_j * weight; //accumulating
				denominator += weight;
			}

		//storing
		numerator /= denominator;
		X[it] = numerator.x;
		X[it + 1] = numerator.y;
	}

}

extern "C"
void cudaMeanShift_2D_wrapper(float *X, const float *I, const float * originalPoints, const int N, const int vecDim, dim3 gridDim, dim3 blockDim) {
	cuda_MeanShift_2D <<<gridDim, blockDim >>> (X, I, originalPoints, N, vecDim);
}

#endif // !MEANSHIFT_CU
