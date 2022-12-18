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

        size_t width = images[i].width;
        size_t height = images[i].height;
        int* buffer = &images[i].buffer[0];

        int num_items = images[i].buffer.size();

        // std::cout << "width: " << width <<
        //     " height: " << height <<
        //     " num_items: " << num_items << std::endl;

        int *d_in = NULL;
        CHECK_CUDA_CALL(cudaMalloc(&d_in, num_items * sizeof(int)));
        CHECK_CUDA_CALL(cudaMemcpy(d_in, buffer, num_items * sizeof(int),
                        cudaMemcpyHostToDevice));

        int *d_out = NULL;
        CHECK_CUDA_CALL(cudaMalloc(&d_out, num_items * sizeof(int)));

        int *d_num_selected_out = NULL;
        CHECK_CUDA_CALL(cudaMalloc(&d_num_selected_out, 1 * sizeof(int)));
        CHECK_CUDA_CALL(cudaMemset(d_num_selected_out, 0, 1 * sizeof(int)));
        DifferentFrom select(-27);

        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;

        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                d_num_selected_out, num_items, select);
        // Allocate temporary storage
        CHECK_CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        // std::cout << "&temp_storage_bytes: " << &temp_storage_bytes << std::endl;
        // std::cout << "temp_storage_bytes: " << temp_storage_bytes << std::endl;

        // Run selection
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                d_num_selected_out, num_items, select);

        int d_num_selected_out_host = 0;
        CHECK_CUDA_CALL(cudaMemcpy(&d_num_selected_out_host,
                    d_num_selected_out, 1 * sizeof(int), cudaMemcpyDeviceToHost));

        // std::cout << "from " << num_items << " to "
        //     << d_num_selected_out_host << std::endl;
        // std::cout << "expected: " << width * height << std::endl;

        assert(d_num_selected_out_host == width * height);

        CHECK_CUDA_CALL(cudaFree(d_temp_storage));
        CHECK_CUDA_CALL(cudaFree(d_in));
        CHECK_CUDA_CALL(cudaFree(d_out));
        CHECK_CUDA_CALL(cudaFree(d_num_selected_out));

        return 0;
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
