#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Вспомогательный макрос: проверка ошибок CUDA
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__         \
                      << "  " << cudaGetErrorString(_e) << std::endl;         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
//  Ядро 1: наивное умножение (один поток — один элемент C)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void matmul_naive(const int* __restrict__ A,
                             const int* __restrict__ B,
                             int*       __restrict__ C,
                             int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    int acc = 0;
    for (int k = 0; k < N; ++k)
        acc += A[row * N + k] * B[k * N + col];
    C[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Ядро 2: тайловое умножение через shared memory
//  TILE_SIZE задаётся через шаблонный параметр, чтобы shared-массивы
//  были константного размера (требование CUDA)
// ─────────────────────────────────────────────────────────────────────────────
template <int TILE>
__global__ void matmul_tiled(const int* __restrict__ A,
                             const int* __restrict__ B,
                             int*       __restrict__ C,
                             int N)
{
    __shared__ int sA[TILE][TILE];
    __shared__ int sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    int acc = 0;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0;
        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = acc;
}

// ─────────────────────────────────────────────────────────────────────────────
//  I/O
// ─────────────────────────────────────────────────────────────────────────────
std::vector<int> load_matrix(const std::string& path, int& N)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    std::vector<std::vector<int>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::vector<int> row;
        int v;
        while (ss >> v) row.push_back(v);
        if (!row.empty()) rows.push_back(row);
    }
    N = (int)rows.size();
    std::vector<int> mat(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i * N + j] = rows[i][j];
    return mat;
}

void save_matrix(const std::vector<int>& mat, int N, const std::string& path)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write: " + path);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            f << mat[i * N + j];
            if (j + 1 < N) f << ' ';
        }
        f << '\n';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Запуск одного варианта ядра + замер времени через cudaEvent
// ─────────────────────────────────────────────────────────────────────────────
struct RunResult { float ms; };

// Наивное ядро
RunResult run_naive(const int* dA, const int* dB, int* dC, int N, int block_size)
{
    dim3 block(block_size, block_size);
    dim3 grid((N + block_size - 1) / block_size,
              (N + block_size - 1) / block_size);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {ms};
}

// Тайловое ядро — инстанциирование для tile=8,16,32
RunResult run_tiled(const int* dA, const int* dB, int* dC, int N, int tile)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    if (tile == 8) {
        dim3 block(8, 8);
        dim3 grid((N + 7) / 8, (N + 7) / 8);
        matmul_tiled<8><<<grid, block>>>(dA, dB, dC, N);
    } else if (tile == 16) {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        matmul_tiled<16><<<grid, block>>>(dA, dB, dC, N);
    } else {
        dim3 block(32, 32);
        dim3 grid((N + 31) / 32, (N + 31) / 32);
        matmul_tiled<32><<<grid, block>>>(dA, dB, dC, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return {ms};
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
//  Использование: ./matrix_cuda [block_size] [kernel]
//    block_size : 8 | 16 | 32  (default 16)
//    kernel     : naive | tiled (default tiled)
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    int  block_size = (argc > 1) ? std::atoi(argv[1]) : 16;
    bool use_tiled  = (argc > 2) ? (std::string(argv[2]) != "naive") : true;

    // Напечатать GPU-информацию
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU             : " << prop.name << std::endl;
    std::cout << "SM count        : " << prop.multiProcessorCount << std::endl;
    std::cout << "Shared mem / SM : " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Kernel          : " << (use_tiled ? "tiled" : "naive") << std::endl;
    std::cout << "Block size      : " << block_size << "x" << block_size << std::endl;

    // Загрузка матриц
    int N = 0;
    auto hA = load_matrix("matrix_a", N);
    int  N2 = 0;
    auto hB = load_matrix("matrix_b", N2);
    if (N != N2) { std::cerr << "Matrix size mismatch\n"; return 1; }

    std::vector<int> hC(N * N, 0);

    // Выделение GPU-памяти
    int *dA, *dB, *dC;
    size_t bytes = (size_t)N * N * sizeof(int);
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    // Запуск
    RunResult res;
    if (use_tiled)
        res = run_tiled(dA, dB, dC, N, block_size);
    else
        res = run_naive(dA, dB, dC, N, block_size);

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // Освобождение памяти
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    // Сохранение результата
    save_matrix(hC, N, "result_matrix");

    // Статистика
    long long flops = 2LL * N * N * N;
    double gflops = (double)flops / (res.ms / 1000.0) / 1e9;
    dim3 grid_dim((N + block_size - 1) / block_size,
                  (N + block_size - 1) / block_size);

    std::cout << "Matrix size     : " << N << "x" << N << std::endl;
    std::cout << "Grid            : " << grid_dim.x << "x" << grid_dim.y << " blocks" << std::endl;
    std::cout << "Time            : " << res.ms << " ms" << std::endl;
    std::cout << "FLOPs           : " << flops << std::endl;
    std::cout << "Performance     : " << gflops << " GFLOPS" << std::endl;

    return 0;
}
