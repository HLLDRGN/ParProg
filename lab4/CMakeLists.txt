cmake_minimum_required(VERSION 3.18)
project(MatrixMultiplyCUDA CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# GTX 1080 → Pascal → sm_61
# Если хочешь поддержать несколько архитектур, добавь через точку с запятой:
# set(CMAKE_CUDA_ARCHITECTURES "61;75;86")
set(CMAKE_CUDA_ARCHITECTURES "61")

add_executable(matrix_cuda main_cuda.cu)

set_target_properties(matrix_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_options(matrix_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        $<$<CONFIG:Release>:-O3 --use_fast_math>
        $<$<CONFIG:Debug>:-G -g>
    >
)
