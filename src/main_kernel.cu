#include "image.hh"
#include "pipeline.hh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

#include <cub/cub.cuh>

#include <thrust/find.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <thrust/transform.h>
#include <thrust/functional.h>

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

struct ToneMap
{
    int *min_histo;
    int *histo;
    int img_dim;

    CUB_RUNTIME_FUNCTION __forceinline__
    explicit ToneMap(int *min_histo, int *histo, int img_dim) : min_histo(min_histo), histo(histo), img_dim(img_dim) {}

    __host__ __device__ __forceinline__
    int operator()(const int &a) const {
        return std::roundf(((histo[a] - *min_histo) / static_cast<float>(img_dim - *min_histo)) * 255.0f);
    }
};

struct RemoveGarbage
{
    __host__ __device__ __forceinline__
    int4 operator()(const int4 &a) const {
        int4 tmp = a;

        tmp.x += 1;
        tmp.y -= 5;
        tmp.z += 3;
        tmp.w -= 8;

        return tmp;
    }
};

constexpr const long unsigned int expected_total[] = {
    27805567, 185010925, 342970490, 33055988, 390348481,
    91297791, 10825197, 118842538, 72434629, 191735142,
    182802772, 78632198, 491605096, 8109782, 111786760,
    406461934, 80671811, 70004942, 104275727, 30603818,
    6496225, 207334021, 268424419, 432916359, 51973720,
    24489209, 80124196, 29256842, 25803206, 34550754,
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

    /// Prepare streams
    constexpr int STREAM_COUNT = 4;
    cudaStream_t streams[STREAM_COUNT] = { 0 };
    for (int i = 0; i < STREAM_COUNT; i++)
    {
        cudaStreamCreate(streams + i);
    }

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

        int img_dim = width * height;

        /// Retrieve the attached stream
        cudaStream_t stream = streams[i % 4];

        /// Prepare CUDA buffer (image input)
        int *d_in = NULL;
        CHECK_CUDA_CALL(cudaMallocAsync(&d_in, num_items * sizeof(int), stream));
        CHECK_CUDA_CALL(cudaMemcpyAsync(d_in, buffer, num_items * sizeof(int),
                        cudaMemcpyHostToDevice, stream));

        /// Prepare CUDA buffer (image without -27s)
        int *d_out = NULL;
        CHECK_CUDA_CALL(cudaMallocAsync(&d_out, num_items * sizeof(int), stream));

        /// Retrieve the information
        int *d_num_selected_out = NULL;
        CHECK_CUDA_CALL(cudaMallocAsync(&d_num_selected_out, 1 * sizeof(int), stream));
        CHECK_CUDA_CALL(cudaMemsetAsync(d_num_selected_out, 0, 1 * sizeof(int), stream));

        /// Create a class with overloaded bool operator
        DifferentFrom select(-27);

        // Determine temporary device storage requirements
        {
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            int d_num_selected_out_host = 0;

            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                    d_num_selected_out, num_items, select, stream);
            // Allocate temporary storage
            CHECK_CUDA_CALL(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

            // Run selection (removes -27s)
            cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_out,
                    d_num_selected_out, num_items, select, stream);

            /// Move to CPU side the count of item to check
            /// TODO: Remove this (useless)
            CHECK_CUDA_CALL(cudaMemcpyAsync(&d_num_selected_out_host,
                        d_num_selected_out, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream));

            assert(d_num_selected_out_host == img_dim);

            CHECK_CUDA_CALL(cudaFreeAsync(d_temp_storage, stream));
        }

        /// Remove the random garbage values from the array
        {
            auto policy = thrust::cuda::par.on(stream);

            RemoveGarbage garbage;
            int4 *d_garbage = reinterpret_cast<int4 *>(d_out);
            int img_dim_4 = img_dim / 4 + (img_dim % 4 == 0 ? 0 : 1);
            thrust::transform(policy, d_garbage, d_garbage + img_dim_4, d_garbage, garbage);
        }

        /// Compute histogram
        int*     d_histogram = NULL;
        CHECK_CUDA_CALL(cudaMallocAsync(&d_histogram, 256 * sizeof(int), stream));
        CHECK_CUDA_CALL(cudaMemsetAsync(d_histogram, 0, 256 * sizeof(int), stream));

        {
            void*    d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;

            int      num_samples = img_dim;
            int*   d_samples = d_out;
            int num_levels  = 256 + 1;
            int lower_level = 0;
            int upper_level = 256;

            cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                    d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples, stream);

            // Allocate temporary storage
            CHECK_CUDA_CALL(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
            // Compute histograms
            cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                    d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples, stream);

            CHECK_CUDA_CALL(cudaFreeAsync(d_temp_storage, stream));
        }

        /// Replace histogram with cumulative histogram
        {
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;

            int *d_in = d_histogram;
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_in, 256, stream);
            // Allocate temporary storage
            CHECK_CUDA_CALL(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));

            // Run exclusive prefix sum
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_in, 256, stream);

            CHECK_CUDA_CALL(cudaFreeAsync(d_temp_storage, stream));
        }

        /// Apply histogram equalization
        {
            auto policy = thrust::cuda::par.on(stream);

            auto iter = thrust::find_if(policy, d_histogram, d_histogram + 256, DifferentFrom(0));
            ToneMap tonemap(iter, d_histogram, img_dim);
            thrust::transform(policy, d_out, d_out + img_dim, d_out, tonemap);
        }

        /// Compute reduce
        int *reduce_sum = NULL;
        CHECK_CUDA_CALL(cudaMallocAsync(&reduce_sum, 1 * sizeof(int), stream));
        CHECK_CUDA_CALL(cudaMemsetAsync(reduce_sum, 0, 1 * sizeof(int), stream));

        // - First compute the total of each image

        // TODO : make it GPU compatible (aka faster)
        // You can use multiple CPU threads for your GPU version using openmp or not
        // Up to you :)
        {
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;

            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_out, reduce_sum, img_dim, stream);
            // Allocate temporary storage
            CHECK_CUDA_CALL(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream));
            // Run sum-reduction
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_out, reduce_sum, img_dim, stream);

            CHECK_CUDA_CALL(cudaFreeAsync(d_temp_storage, stream));
        }

        /// Not mandatory (we are not using values past img_dim)
        images[i].buffer.resize(img_dim);

        /// Retrieve the image from GPU
        CHECK_CUDA_CALL(cudaMemcpyAsync(buffer, d_out, img_dim * sizeof(int), cudaMemcpyDeviceToHost, stream));
        /// Retrieve the total from GPU
        CHECK_CUDA_CALL(cudaMemcpyAsync(&images[i].to_sort.total, reduce_sum, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream));

        /// Clean everything
        CHECK_CUDA_CALL(cudaFreeAsync(d_in, stream));
        CHECK_CUDA_CALL(cudaFreeAsync(d_out, stream));
        CHECK_CUDA_CALL(cudaFreeAsync(d_num_selected_out, stream));
        CHECK_CUDA_CALL(cudaFreeAsync(reduce_sum, stream));
        CHECK_CUDA_CALL(cudaFreeAsync(d_histogram, stream));
    }

    /// Cleanup streams
    for (int i = 0; i < STREAM_COUNT; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)


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

    for (int i = 0; i < nb_images; ++i) {
        auto &img = images[i];
        if (img.to_sort.total != expected_total[img.to_sort.id]) {
            std::cerr << "Differ computed image " << i << ": (" << img.to_sort.total <<
                ") expected " << expected_total[img.to_sort.id] << std::endl;
        }
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    return 0;
}
