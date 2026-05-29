cmake_minimum_required(VERSION 3.16)
project(MatrixMultiplyOMP CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Поиск OpenMP (делаем необязательным и добавляем фолбэк для Homebrew libomp на macOS)
find_package(OpenMP)

# Если find_package не нашёл OpenMP, пробуем подключить Homebrew libomp вручную (macOS)
if(NOT OpenMP_CXX_FOUND)
    message(STATUS "OpenMP not found via find_package — attempting Homebrew libomp fallback")
    set(LIBOMP_PREFIX "${CMAKE_PREFIX_PATH}")
    # common Homebrew locations
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(LIBOMP_PREFIX "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(LIBOMP_PREFIX "/usr/local/opt/libomp")
    endif()

    if(EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib" OR EXISTS "${LIBOMP_PREFIX}/lib/libomp.a")
        message(STATUS "Using libomp from ${LIBOMP_PREFIX}")
        add_compile_options(-Xpreprocessor -fopenmp)
        include_directories(${LIBOMP_PREFIX}/include)
        link_directories(${LIBOMP_PREFIX}/lib)
        set(OpenMP_CXX_FOUND TRUE)
        set(OpenMP_CXX_LIBRARIES "${LIBOMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    endif()
endif()

add_executable(matrix_mul main_omp.cpp)

target_compile_options(matrix_mul PRIVATE
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-g -Wall -Wextra>
)

# Линковка OpenMP: используем найденный импортный target, или явную библиотеку из fallback
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(matrix_mul PRIVATE OpenMP::OpenMP_CXX)
elseif(DEFINED OpenMP_CXX_LIBRARIES)
    target_link_libraries(matrix_mul PRIVATE ${OpenMP_CXX_LIBRARIES})
else()
    message(FATAL_ERROR "OpenMP library not found; please install libomp or provide OpenMP support")
endif()
