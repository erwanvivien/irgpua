#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>


template <typename T>
__global__
void kernel_reduce_baseline(const T* __restrict__ buffer, T* __restrict__ total, int size)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size)
        atomicAdd(&total[0], buffer[id]);
}

void baseline_reduce(cuda_tools::host_shared_ptr<int> buffer,
    cuda_tools::host_shared_ptr<int> total)
{
    cudaProfilerStart();
    cudaFuncSetCacheConfig(kernel_reduce_baseline<int>, cudaFuncCachePreferShared);

    constexpr int blocksize = 64;
    const int gridsize = (buffer.size_ + blocksize - 1) / blocksize;

    kernel_reduce_baseline<int><<<gridsize, blocksize>>>(buffer.data_, total.data_, buffer.size_);

    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}

template <int OFFSET>
__device__ void warp_reduce(int tid, int *buffer_shared)
{
    if (tid < OFFSET)
        buffer_shared[tid] += buffer_shared[tid + OFFSET];
}

template <typename T, int BLOCK_SIZE>
    __global__
void reduce_sum(const T* __restrict__ buffer, T* __restrict__ total, int size)
{
    __shared__ T buffer_shared[BLOCK_SIZE << 1];
    const int tid = threadIdx.x;
    const int coord = tid + blockIdx.x * BLOCK_SIZE * 8;

    buffer_shared[tid] = 0;
    buffer_shared[tid + BLOCK_SIZE] = 0;
    if (coord < size)
        buffer_shared[tid] += buffer[coord];
    if (coord + BLOCK_SIZE * 1 < size)
        buffer_shared[tid] += buffer[coord + BLOCK_SIZE * 1];
    if (coord + BLOCK_SIZE * 2 < size)
        buffer_shared[tid] += buffer[coord + BLOCK_SIZE * 2];
    if (coord + BLOCK_SIZE * 3 < size)
        buffer_shared[tid] += buffer[coord + BLOCK_SIZE * 3];
    if (coord + BLOCK_SIZE * 4 < size)
        buffer_shared[tid + BLOCK_SIZE] += buffer[coord + BLOCK_SIZE * 4];
    if (coord + BLOCK_SIZE * 5 < size)
        buffer_shared[tid + BLOCK_SIZE] += buffer[coord + BLOCK_SIZE * 5];
    if (coord + BLOCK_SIZE * 6 < size)
        buffer_shared[tid + BLOCK_SIZE] += buffer[coord + BLOCK_SIZE * 6];
    if (coord + BLOCK_SIZE * 7 < size)
        buffer_shared[tid + BLOCK_SIZE] += buffer[coord + BLOCK_SIZE * 7];

    __syncthreads();

    if constexpr (BLOCK_SIZE >> 0 > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 0;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr (BLOCK_SIZE >> 1 > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 1;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr ((BLOCK_SIZE >> 2) > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 2;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr ((BLOCK_SIZE >> 3) > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 3;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr ((BLOCK_SIZE >> 4) > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 4;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr ((BLOCK_SIZE >> 5) > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 5;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr ((BLOCK_SIZE >> 6) > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 6;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr ((BLOCK_SIZE >> 7) > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 7;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr ((BLOCK_SIZE >> 8) > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 8;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr ((BLOCK_SIZE >> 9) > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 9;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }
    if constexpr ((BLOCK_SIZE >> 10) > 0) {
        constexpr const int BLOCK = BLOCK_SIZE >> 10;
        warp_reduce<BLOCK>(tid, buffer_shared);
        __syncthreads();
    }

    __syncthreads();

    if (tid == 0)
        atomicAdd(total, buffer_shared[0]);
}

template <typename T>
__global__
void kernel_your_reduce(const T* __restrict__ buffer, T* __restrict__ total, int size)
{
    __shared__ T buffer_shared[2048];
    const int id = threadIdx.x + blockIdx.x * blockDim.x * 4;
    const int id_less_size = id < size;

    buffer_shared[threadIdx.x] = buffer[id] * (id < size);
    buffer_shared[threadIdx.x] += buffer[id + blockDim.x] * (id + blockDim.x < size);
    buffer_shared[threadIdx.x + 1024] = buffer[id + blockDim.x * 2] * (id + blockDim.x * 2 < size);
    buffer_shared[threadIdx.x + 1024] += buffer[id + blockDim.x * 3] * (id + blockDim.x * 3 < size);

    __syncthreads();

    for (int start = blockDim.x; start > 0; start >>= 1)
    {
        if (threadIdx.x < start && id_less_size)
        {
            buffer_shared[threadIdx.x] += buffer_shared[threadIdx.x + start];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&total[0], buffer_shared[0]);
    }
}

void your_reduce(cuda_tools::host_shared_ptr<int> buffer,
    cuda_tools::host_shared_ptr<int> total)
{
    cudaProfilerStart();

    constexpr int blocksize = 1024;
    const int gridsize = (buffer.size_ + blocksize - 1) / (blocksize * 16);

	reduce_sum<int, blocksize><<<gridsize, blocksize>>>(buffer.data_, total.data_, buffer.size_);

    cudaDeviceSynchronize();
    kernel_check_error();
    
    cudaProfilerStop();
}
