cmake_minimum_required(VERSION 3.16)
project(MatrixMultiply CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release-сборка по умолчанию
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(matrix_mul main.cpp)

target_compile_options(matrix_mul PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -Wall -Wextra>
)
