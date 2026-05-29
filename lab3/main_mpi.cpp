#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <mpi.h>

using namespace std;
using Grid = vector<vector<int>>;
using Flat = vector<int>;

// ── I/O ──────────────────────────────────────────────────────────────────────

Grid load_grid(const string& path) {
    ifstream file(path);
    if (!file) { cerr << "Cannot open: " << path << endl; return {}; }
    Grid grid;
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream ss(line);
        vector<int> row;
        int v;
        while (ss >> v) row.push_back(v);
        if (!row.empty()) grid.push_back(row);
    }
    return grid;
}

void save_grid(const Grid& g, const string& path) {
    ofstream f(path);
    if (!f) throw runtime_error("Cannot write: " + path);
    int n = (int)g.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            f << g[i][j];
            if (j + 1 < n) f << ' ';
        }
        f << '\n';
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

// Flatten row-major
Flat flatten(const Grid& g) {
    int n = (int)g.size();
    Flat f(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            f[i * n + j] = g[i][j];
    return f;
}

Grid unflatten(const Flat& f, int n) {
    Grid g(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            g[i][j] = f[i * n + j];
    return g;
}

Grid transpose(const Grid& g) {
    int n = (int)g.size();
    Grid t(n, vector<int>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            t[j][i] = g[i][j];
    return t;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0;
    Flat flatA, flatB, flatBt, flatC;

    // ── Процесс 0: читает матрицы, транспонирует B ──
    if (rank == 0) {
        Grid A = load_grid("matrix_a");
        Grid B = load_grid("matrix_b");
        n = (int)A.size();
        Grid Bt = transpose(B);
        flatA  = flatten(A);
        flatBt = flatten(Bt);
        flatC.resize(n * n, 0);
    }

    // Рассылаем размер всем процессам
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        flatBt.resize(n * n);
        flatC.resize(n * n, 0);
    }

    // Рассылаем транспонированную B всем процессам
    MPI_Bcast(flatBt.data(), n * n, MPI_INT, 0, MPI_COMM_WORLD);

    // Распределяем строки A между процессами
    // rows_per_proc[i] — сколько строк достаётся процессу i
    vector<int> rows_per_proc(size), offsets(size);
    for (int i = 0; i < size; ++i) {
        rows_per_proc[i] = n / size + (i < n % size ? 1 : 0);
        offsets[i] = (i == 0) ? 0 : offsets[i-1] + rows_per_proc[i-1];
    }

    int local_rows = rows_per_proc[rank];
    Flat localA(local_rows * n);

    // Scatter строк A
    vector<int> send_counts(size), send_offsets(size);
    for (int i = 0; i < size; ++i) {
        send_counts[i]  = rows_per_proc[i] * n;
        send_offsets[i] = offsets[i] * n;
    }
    MPI_Scatterv(
        rank == 0 ? flatA.data() : nullptr,
        send_counts.data(), send_offsets.data(), MPI_INT,
        localA.data(), local_rows * n, MPI_INT,
        0, MPI_COMM_WORLD
    );

    // ── Вычисление локальной части C ──
    double t_start = MPI_Wtime();

    Flat localC(local_rows * n, 0);
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < n; ++j) {
            int acc = 0;
            for (int k = 0; k < n; ++k)
                acc += localA[i * n + k] * flatBt[j * n + k];
            localC[i * n + j] = acc;
        }
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;

    // Собираем максимальное время (узкое место)
    double elapsed = 0;
    MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather результата
    MPI_Gatherv(
        localC.data(), local_rows * n, MPI_INT,
        rank == 0 ? flatC.data() : nullptr,
        send_counts.data(), send_offsets.data(), MPI_INT,
        0, MPI_COMM_WORLD
    );

    // ── Процесс 0: сохраняет результат и выводит статистику ──
    if (rank == 0) {
        Grid C = unflatten(flatC, n);
        save_grid(C, "result_matrix");

        long long flops = 2LL * n * n * n;
        double ms = elapsed * 1000.0;
        double gflops = (double)flops / elapsed / 1e9;

        cout << "Matrix size : " << n << "x" << n << endl;
        cout << "MPI procs   : " << size << endl;
        cout << "Time        : " << ms << " ms" << endl;
        cout << "FLOPs       : " << flops << endl;
        cout << "Performance : " << gflops << " GFLOPS" << endl;
    }

    MPI_Finalize();
    return 0;
}
