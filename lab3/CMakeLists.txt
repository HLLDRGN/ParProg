cmake_minimum_required(VERSION 3.16)
project(MatrixMultiplyMPI CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)

add_executable(matrix_mul main_mpi.cpp)

target_compile_options(matrix_mul PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -Wall -Wextra>
)

target_link_libraries(matrix_mul PRIVATE MPI::MPI_CXX)
