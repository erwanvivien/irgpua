#include "to_bench.cuh"

#include <cub/cub.cuh>

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>

#include <iostream>

constexpr const int blocksize = 1024;

inline void checkCudaCall(cudaError_t error, const char* file, int line)
{
    if (error)
    {
        std::cout << "CUDA error at " << file << ":" << line << std::endl;
        std::cout << cudaGetErrorName(error) << " :: " << cudaGetErrorString(error) << std::endl;
    }
}
#define CHECK_CUDA_CALL(err) (checkCudaCall(err, __FILE__, __LINE__))

template <typename T>
    __global__
void kernel_scan_baseline(T* buffer, int size)
{
    for (int i = 1; i < size; ++i)
        buffer[i] += buffer[i - 1];
}

void baseline_scan(cuda_tools::host_shared_ptr<int> buffer)
{
    cudaProfilerStart();
    cudaFuncSetCacheConfig(kernel_scan_baseline<int>, cudaFuncCachePreferShared);

    kernel_scan_baseline<int><<<1, 1>>>(buffer.data_, buffer.size_);

    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();
}

enum State {
    NoCompute = 0,
    SelfCompute = 1,
    AllCompute = 2,
};

    template <int STEP>
__device__ void warp_scan(
        int *internal_buffer_1, int *internal_buffer_2, int *internal_buffer_3, int *internal_buffer_4,
        int *internal_buffer_5, int *internal_buffer_6, int *internal_buffer_7, int *internal_buffer_8,
        int tid)
{
    constexpr int left = 1 << STEP;
    if ((tid & left) != 0) {
        int right = tid >> STEP;
        int from = left * right - 1;

        internal_buffer_1[tid] += internal_buffer_1[from];
        internal_buffer_2[tid] += internal_buffer_2[from];
        internal_buffer_3[tid] += internal_buffer_3[from];
        internal_buffer_4[tid] += internal_buffer_4[from];
        internal_buffer_5[tid] += internal_buffer_5[from];
        internal_buffer_6[tid] += internal_buffer_6[from];
        internal_buffer_7[tid] += internal_buffer_7[from];
        internal_buffer_8[tid] += internal_buffer_8[from];
    }

    __syncthreads();
}

    template <typename T, int BLOCK_SIZE>
__global__ void kernel_your_scan(T* buffer, int size, int *counter, int* status, int *internal_sum, int* preceeding_sum)
{
    __shared__ T internal_buffer_1[BLOCK_SIZE];
    __shared__ T internal_buffer_2[BLOCK_SIZE];
    __shared__ T internal_buffer_3[BLOCK_SIZE];
    __shared__ T internal_buffer_4[BLOCK_SIZE];
    __shared__ T internal_buffer_5[BLOCK_SIZE];
    __shared__ T internal_buffer_6[BLOCK_SIZE];
    __shared__ T internal_buffer_7[BLOCK_SIZE];
    __shared__ T internal_buffer_8[BLOCK_SIZE];
    __shared__ int blockIdx_x;

    int tid = threadIdx.x;
    if (tid == 0)
        blockIdx_x = atomicAdd(counter, 1);

    __syncthreads();

    int coord = tid + blockIdx_x * (BLOCK_SIZE << 3);
    internal_buffer_1[tid] = buffer[coord];
    internal_buffer_2[tid] = buffer[coord + BLOCK_SIZE];
    internal_buffer_3[tid] = buffer[coord + BLOCK_SIZE * 2];
    internal_buffer_4[tid] = buffer[coord + BLOCK_SIZE * 3];

    internal_buffer_5[tid] = buffer[coord + BLOCK_SIZE * 4];
    internal_buffer_6[tid] = buffer[coord + BLOCK_SIZE * 5];
    internal_buffer_7[tid] = buffer[coord + BLOCK_SIZE * 6];
    internal_buffer_8[tid] = buffer[coord + BLOCK_SIZE * 7];


    __syncthreads();

    // Cumulative sum
    warp_scan<0>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid); __syncwarp();
    warp_scan<1>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid); __syncwarp();
    warp_scan<2>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid); __syncwarp();
    warp_scan<3>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid); __syncwarp();
    warp_scan<4>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid); __syncwarp();

    if constexpr (32 < BLOCK_SIZE) {
        warp_scan<5>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid);
        __syncthreads();
    }
    if constexpr (64 < BLOCK_SIZE) {
        warp_scan<6>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid);
        __syncthreads();
    }
    if constexpr (128 < BLOCK_SIZE) {
        warp_scan<7>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid);
        __syncthreads();
    }
    if constexpr (256 < BLOCK_SIZE) {
        warp_scan<8>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid);
        __syncthreads();
    }
    if constexpr (512 < BLOCK_SIZE) {
        warp_scan<9>(internal_buffer_1, internal_buffer_2, internal_buffer_3, internal_buffer_4, internal_buffer_5, internal_buffer_6, internal_buffer_7, internal_buffer_8, tid);
        __syncthreads();
    }

    constexpr const int last = BLOCK_SIZE - 1;
    internal_buffer_2[tid] += internal_buffer_1[last];
    __syncthreads();
    internal_buffer_3[tid] += internal_buffer_2[last];
    __syncthreads();
    internal_buffer_4[tid] += internal_buffer_3[last];
    __syncthreads();
    internal_buffer_5[tid] += internal_buffer_4[last];
    __syncthreads();
    internal_buffer_6[tid] += internal_buffer_5[last];
    __syncthreads();
    internal_buffer_7[tid] += internal_buffer_6[last];
    __syncthreads();
    internal_buffer_8[tid] += internal_buffer_7[last];

    int *prefix_sum = preceeding_sum + blockIdx_x;
    int *curr_sum = internal_sum + blockIdx_x;
    int *curr_status = status + blockIdx_x;

    __shared__ int prev_value;

    if (tid == last) {
        int local_prev_value = 0;

        atomicExch(curr_sum, internal_buffer_8[last]);
        __threadfence_system();
        atomicExch(curr_status, SelfCompute);

        if (blockIdx_x != 0) {
            int back = 1;
            while (back <= blockIdx_x)
            {
                int back_status = atomicAdd(curr_status - back, 0);
                if (back_status == NoCompute)
                {
                    continue;
                }

                else if (back_status == SelfCompute) {
                    local_prev_value += atomicAdd(curr_sum - back, 0);
                    back += 1;
                } else {
                    local_prev_value += atomicAdd(prefix_sum - back, 0);
                    break;
                }
            }
        }

        prev_value = local_prev_value;
    }

    __syncthreads();

    internal_buffer_1[tid] += prev_value;
    internal_buffer_2[tid] += prev_value;
    internal_buffer_3[tid] += prev_value;
    internal_buffer_4[tid] += prev_value;
    internal_buffer_5[tid] += prev_value;
    internal_buffer_6[tid] += prev_value;
    internal_buffer_7[tid] += prev_value;
    internal_buffer_8[tid] += prev_value;

    if (tid == last)
    {
        atomicExch(prefix_sum, internal_buffer_8[last]);
        __threadfence_system();
        atomicExch(curr_status, AllCompute);
    }

    buffer[coord] = internal_buffer_1[tid];
    buffer[coord + BLOCK_SIZE] = internal_buffer_2[tid];
    buffer[coord + BLOCK_SIZE * 2] = internal_buffer_3[tid];
    buffer[coord + BLOCK_SIZE * 3] = internal_buffer_4[tid];
    buffer[coord + BLOCK_SIZE * 4] = internal_buffer_5[tid];
    buffer[coord + BLOCK_SIZE * 5] = internal_buffer_6[tid];
    buffer[coord + BLOCK_SIZE * 6] = internal_buffer_7[tid];
    buffer[coord + BLOCK_SIZE * 7] = internal_buffer_8[tid];
}

void your_scan(cuda_tools::host_shared_ptr<int> buffer)
{
    const int gridsize = (buffer.size_ + blocksize - 1) / (blocksize * 8);

    int *counter = NULL;
    CHECK_CUDA_CALL(cudaMalloc(&counter, 1 * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(counter, 0, 1 * sizeof(int)));
    kernel_check_error();

    int *status = NULL;
    CHECK_CUDA_CALL(cudaMalloc(&status, gridsize * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(status, NoCompute, gridsize * sizeof(int)));
    kernel_check_error();

    int *internal_sum = NULL;
    CHECK_CUDA_CALL(cudaMalloc(&internal_sum, gridsize * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(internal_sum, 0, gridsize * sizeof(int)));
    kernel_check_error();

    int *preceeding_sum = NULL;
    CHECK_CUDA_CALL(cudaMalloc(&preceeding_sum, gridsize * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(preceeding_sum, 0, gridsize * sizeof(int)));
    kernel_check_error();

    cudaProfilerStart();
    cudaFuncSetCacheConfig(kernel_your_scan<int, blocksize>, cudaFuncCachePreferShared);

    // printf("%d x %d = %d\n", gridsize, blocksize, buffer.size_);
    kernel_your_scan<int, blocksize><<<gridsize, blocksize>>>(buffer.data_, buffer.size_, counter, status, internal_sum, preceeding_sum);

    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();

    CHECK_CUDA_CALL(cudaFree(counter));
    CHECK_CUDA_CALL(cudaFree(status));
    CHECK_CUDA_CALL(cudaFree(internal_sum));
}

void cub_scan(cuda_tools::host_shared_ptr<int> buffer)
{
    const int gridsize = (buffer.size_ + blocksize - 1) / (blocksize * 2);

    int *counter = NULL;
    CHECK_CUDA_CALL(cudaMalloc(&counter, 1 * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(counter, 0, 1 * sizeof(int)));
    kernel_check_error();

    int *status = NULL;
    CHECK_CUDA_CALL(cudaMalloc(&status, gridsize * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(status, NoCompute, gridsize * sizeof(int)));
    kernel_check_error();

    int *internal_sum = NULL;
    CHECK_CUDA_CALL(cudaMalloc(&internal_sum, gridsize * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(internal_sum, NoCompute, gridsize * sizeof(int)));
    kernel_check_error();

    // printf("%d x %d = %d\n", gridsize, blocksize, buffer.size_);
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    int num_items = buffer.size_;
    int *d_in = buffer.data_;
    int *d_out = NULL;
    CHECK_CUDA_CALL(cudaMalloc(&d_out, num_items * sizeof(int)));
    CHECK_CUDA_CALL(cudaMemset(d_out, 0, num_items * sizeof(int)));
    kernel_check_error();

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum

    cudaProfilerStart();
    cudaFuncSetCacheConfig(kernel_your_scan<int, blocksize>, cudaFuncCachePreferShared);

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

    cudaDeviceSynchronize();
    kernel_check_error();

    cudaProfilerStop();

    CHECK_CUDA_CALL(cudaMemcpy(buffer.data_, d_out, num_items * sizeof(int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA_CALL(cudaFree(counter));
    CHECK_CUDA_CALL(cudaFree(status));
    CHECK_CUDA_CALL(cudaFree(internal_sum));
    CHECK_CUDA_CALL(cudaFree(d_out));
    CHECK_CUDA_CALL(cudaFree(d_temp_storage));
}
