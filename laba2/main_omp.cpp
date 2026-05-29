#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>

using namespace std;
using Grid = vector<vector<int>>;

Grid load_grid(const string& path) {
    ifstream file(path);
    if (!file) {
        cerr << "Cannot open file: " << path << endl;
        return {};
    }
    Grid grid;
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        istringstream ss(line);
        vector<int> row;
        int val;
        while (ss >> val) row.push_back(val);
        if (!row.empty()) grid.push_back(row);
    }
    return grid;
}

void save_grid(const Grid& grid, const string& path) {
    ofstream file(path);
    if (!file) throw runtime_error("Cannot open file for writing: " + path);
    int rows = static_cast<int>(grid.size());
    if (rows == 0) return;
    int cols = static_cast<int>(grid[0].size());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << grid[i][j];
            if (j + 1 < cols) file << ' ';
        }
        file << '\n';
    }
}

Grid transpose(const Grid& g) {
    int rows = static_cast<int>(g.size());
    int cols = static_cast<int>(g[0].size());
    Grid t(cols, vector<int>(rows));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            t[j][i] = g[i][j];
    return t;
}

// Параллельное умножение через транспонирование B
Grid multiply_parallel(const Grid& A, const Grid& B, int num_threads) {
    int n = static_cast<int>(A.size());
    Grid Bt = transpose(B);
    Grid C(n, vector<int>(n, 0));

    omp_set_num_threads(num_threads);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            int acc = 0;
            const auto& rowA  = A[i];
            const auto& rowBt = Bt[j];
            for (int k = 0; k < n; ++k)
                acc += rowA[k] * rowBt[k];
            C[i][j] = acc;
        }
    }
    return C;
}

int main(int argc, char* argv[]) {
    // Количество потоков можно передать аргументом: ./matrix_mul 4
    int num_threads = omp_get_max_threads();
    if (argc >= 2) num_threads = stoi(argv[1]);

    try {
        Grid A = load_grid("matrix_a");
        Grid B = load_grid("matrix_b");

        int n          = static_cast<int>(A.size());
        long long flops = 2LL * n * n * n;

        cout << "Matrix size : " << n << "x" << n << endl;
        cout << "Threads     : " << num_threads << endl;
        cout << "FLOPs       : " << flops << endl;

        auto t0 = chrono::high_resolution_clock::now();
        Grid C  = multiply_parallel(A, B, num_threads);
        auto t1 = chrono::high_resolution_clock::now();

        save_grid(C, "result_matrix");

        auto ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        cout << "Time        : " << ms << " ms" << endl;

        // Ускорение относительно 1 потока выводится скриптом
        return 0;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
