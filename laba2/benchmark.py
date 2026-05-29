import subprocess
import numpy as np
import os
import sys

SIZES   = [200, 400, 800, 1200, 1600, 2000]
THREADS = [1, 2, 4, 8]
BINARY  = "./build/matrix_mul"

def save_matrix(mat, name):
    np.savetxt(name, mat, fmt="%d", delimiter=" ")

def run(n, t):
    mat_a = np.random.randint(0, 10, (n, n))
    mat_b = np.random.randint(0, 10, (n, n))
    save_matrix(mat_a, "matrix_a")
    save_matrix(mat_b, "matrix_b")

    result = subprocess.run(
        [BINARY, str(t)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return None

    # Верификация
    c_cpp = np.loadtxt("result_matrix", dtype=int)
    c_ref = mat_a @ mat_b
    ok = np.array_equal(c_cpp, c_ref)

    # Парсим время из stdout
    ms = None
    for line in result.stdout.splitlines():
        if line.startswith("Time"):
            ms = int(line.split(":")[1].strip().split()[0])
    return ms, ok

def main():
    if not os.path.exists(BINARY):
        print(f"Binary not found: {BINARY}")
        print("Build first: mkdir build && cd build && cmake .. && cmake --build .")
        sys.exit(1)

    # results[size][threads] = ms
    results = {n: {} for n in SIZES}

    for n in SIZES:
        print(f"\n=== {n}x{n} ===")
        for t in THREADS:
            ret = run(n, t)
            if ret is None:
                continue
            ms, ok = ret
            results[n][t] = ms
            status = "✓" if ok else "✗"
            print(f"  threads={t:2d}  time={ms:6d} ms  verify={status}")

    # Сводная таблица с ускорением
    print("\n" + "=" * 70)
    print("SPEEDUP TABLE (relative to 1 thread)")
    print("=" * 70)
    header = f"{'N':>6} | " + " | ".join(f"T={t:2d} (ms) speedup" for t in THREADS)
    print(header)
    print("-" * 70)
    for n in SIZES:
        base = results[n].get(1)
        row = f"{n:>6} |"
        for t in THREADS:
            ms = results[n].get(t)
            # Guard against missing measurements or zero times (avoid division by zero)
            if ms is None or ms == 0 or base is None or base == 0:
                row += "        N/A       |"
            elif t == 1:
                row += f" {ms:6d} ms  x1.00 |"
            else:
                sp = base / ms
                row += f" {ms:6d} ms  x{sp:.2f}  |"
        print(row)

if __name__ == "__main__":
    main()
