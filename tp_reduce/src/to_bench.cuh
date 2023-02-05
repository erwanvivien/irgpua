#pragma once

#include "cuda_tools/host_shared_ptr.cuh"

void baseline_reduce(cuda_tools::host_shared_ptr<int> buffer,
    cuda_tools::host_shared_ptr<int> total);

void your_reduce(cuda_tools::host_shared_ptr<int> buffer,
    cuda_tools::host_shared_ptr<int> total);