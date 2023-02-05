project(template)
cmake_minimum_required(VERSION 3.18)

include(FindCUDAToolkit)

enable_language(CUDA)

find_package(benchmark REQUIRED)

add_compile_options(-W -Wall)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_FLAGS "--extended-lambda --generate-line-info --expt-relaxed-constexpr")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_FLAGS "--ptxas-options=-v")

add_library(to_bench
  src/to_bench.cu
  src/cuda_tools/host_shared_ptr.cu)

include_directories(src)

add_executable(bench bench/main.cc)
set_property(TARGET bench PROPERTY CXX_STANDARD 20)


target_link_libraries(bench to_bench benchmark::benchmark)

# BUILD_TYPE
if (NOT CMAKE_BUILD_TYPE)
    string(REGEX REPLACE ".*/build_" "" BUILD_NAME ${CMAKE_BINARY_DIR})
    # Default BUILD_TYPE is Release
    set(CMAKE_BUILD_TYPE Release)
    # If folder is named "build_debug" BUILD_TYPE is set to: Debug
    if(${BUILD_NAME} STREQUAL "Debug" OR ${BUILD_NAME} STREQUAL "debug")
        set(CMAKE_BUILD_TYPE Debug)
    endif()
    message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
endif()