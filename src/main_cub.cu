#include "image.hh"
#include "pipeline.hh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

#include <cub/cub.cuh>

inline void checkCudaCall(cudaError_t error, const char* file, int line)
{
    if (error)
    {
        std::cout << "CUDA error at " << file << ":" << line << std::endl;
        std::cout << cudaGetErrorName(error) << " :: " << cudaGetErrorString(error) << std::endl;
    }
}
#define CHECK_CUDA_CALL(err) (checkCudaCall(err, __FILE__, __LINE__))

struct DifferentFrom
{
    int compare;

    CUB_RUNTIME_FUNCTION __forceinline__
    explicit DifferentFrom(int compare) : compare(compare) {}

    CUB_RUNTIME_FUNCTION __forceinline__
    bool operator()(const int &a) const
    {
        return (a != compare);
    }
};

template <int BLOCK_SIZE>
__global__ void kernel_garbage(int* buffer, int *size)
{
    constexpr const static int corrections[] = {
        1, -5, 3, -8
    };

    int tid = threadIdx.x;
    int coord = tid + blockIdx.x * BLOCK_SIZE;

    if (coord < *size)
        buffer[coord] += corrections[tid % 4];
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("./images"))
        filepaths.emplace_back(dir_entry.path()); 

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    // #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course (wait for last class)
        images[i] = pipeline.get_image(i);

        /// Retrieve image information
        size_t width = images[i].width;
        size_t height = images[i].height;
        int* buffer = &images[i].buffer[0];
        int num_items = images[i].buffer.size();

        /// Prepare CUDA buffer (image input)
        int *d_in = NULL;
        CHECK_CUDA_CALL(cudaMalloc(&d_in, num_items * sizeof(int)));
        CHECK_CUDA_CALL(cudaMemcpy(d_in, buffer, num_items * sizeof(int),
                        cudaMemcpyHostToDevice));

        /// Prepare CUDA buffer (image without -27s)
        int *d_out = NULL;
        CHECK_CUDA_CALL(cudaMalloc(&d_out, num_items * sizeof(int)));

        /// Retrieve the information
        int *d_num_selected_out = NULL;
        CHECK_CUDA_CALL(cudaMalloc(&d_num_selected_out, 1 * sizeof(int)));
        CHECK_CUDA_CALL(cudaMemset(d_num_selected_out, 0, 1 * sizeof(int)));

        /// Create a class with overloaded bool operator
        DifferentFrom select(-27);

        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;

        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                d_num_selected_out, num_items, select);
        // Allocate temporary storage
        CHECK_CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // Run selection (removes -27s)
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                d_num_selected_out, num_items, select);

        /// Move to CPU side the count of item to check
        /// TODO: Remove this (useless)
        int d_num_selected_out_host = 0;
        CHECK_CUDA_CALL(cudaMemcpy(&d_num_selected_out_host,
                    d_num_selected_out, 1 * sizeof(int), cudaMemcpyDeviceToHost));

        assert(d_num_selected_out_host == width * height);

        /// Remove the random garbage from the array
        constexpr const int blocksize = 1024;
        const int gridsize = (width * height + blocksize - 1) / blocksize;
        kernel_garbage<blocksize><<<gridsize, blocksize>>>(d_out, d_num_selected_out);

        /// Retrieve the information back
        CHECK_CUDA_CALL(cudaMemcpy(buffer, d_out, d_num_selected_out_host * sizeof(int), cudaMemcpyDeviceToHost));
        images.resize(d_num_selected_out_host);

        /// Clean everything
        CHECK_CUDA_CALL(cudaFree(d_temp_storage));
        CHECK_CUDA_CALL(cudaFree(d_in));
        CHECK_CUDA_CALL(cudaFree(d_out));
        CHECK_CUDA_CALL(cudaFree(d_num_selected_out));
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        image.to_sort.total = std::reduce(image.buffer.cbegin(), image.buffer.cbegin() + image_size, 0);
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    return 0;
}
