import subprocess
import numpy as np
import os
import sys

SIZES   = [200, 400, 800, 1200, 1600, 2000]
PROCS   = [1, 2, 4, 8]
BINARY  = "./build/matrix_mul"

def save_matrix(mat, name):
    np.savetxt(name, mat, fmt="%d", delimiter=" ")

def run(n, procs):
    mat_a = np.random.randint(0, 10, (n, n))
    mat_b = np.random.randint(0, 10, (n, n))
    save_matrix(mat_a, "matrix_a")
    save_matrix(mat_b, "matrix_b")

    result = subprocess.run(
        ["mpirun", "-np", str(procs), BINARY],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr.strip()}")
        return None

    # Верификация
    c_cpp = np.loadtxt("result_matrix", dtype=int)
    c_ref = mat_a @ mat_b
    ok = np.array_equal(c_cpp, c_ref)

    ms = None
    gflops = None
    for line in result.stdout.splitlines():
        if line.startswith("Time"):
            ms = float(line.split(":")[1].strip().split()[0])
        if line.startswith("Performance"):
            gflops = float(line.split(":")[1].strip().split()[0])
    return ms, gflops, ok

def main():
    if not os.path.exists(BINARY):
        print(f"Binary not found: {BINARY}")
        print("Build: mkdir build && cd build && cmake .. && cmake --build .")
        sys.exit(1)

    # results[n][p] = (ms, gflops)
    results = {n: {} for n in SIZES}

    for n in SIZES:
        print(f"\n=== {n}x{n} ===")
        for p in PROCS:
            ret = run(n, p)
            if ret is None:
                continue
            ms, gflops, ok = ret
            results[n][p] = (ms, gflops)
            status = "✓" if ok else "✗"
            print(f"  procs={p}  time={ms:8.2f} ms  {gflops:.2f} GFLOPS  verify={status}")

    # Сводная таблица speedup
    print("\n" + "=" * 72)
    print("SPEEDUP TABLE (relative to 1 process)")
    print("=" * 72)
    header = f"{'N':>6} | " + " | ".join(f"P={p:2d}  ms      speedup" for p in PROCS)
    print(header)
    print("-" * 72)
    for n in SIZES:
        base_ms = results[n].get(1, (None,))[0]
        row = f"{n:>6} |"
        for p in PROCS:
            entry = results[n].get(p)
            if entry is None:
                row += "         N/A          |"
            else:
                ms = entry[0]
                sp = (base_ms / ms) if base_ms and p > 1 else 1.0
                row += f" {ms:8.2f}  x{sp:.2f}       |"
        print(row)

if __name__ == "__main__":
    main()
