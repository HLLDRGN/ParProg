"""
benchmark_cuda.py — серия экспериментов для л/р №4 (CUDA)

Запуск:
    python benchmark_cuda.py

Требования: бинарь ./build/matrix_cuda должен быть собран.
"""
import subprocess
import numpy as np
import os
import sys
from itertools import product

BINARY = "./build/matrix_cuda"

# Размеры матриц согласно ТЗ
SIZES = [256, 512, 1024, 2048]

# Конфигурации: (block_size, kernel)
# block_size 32 для naive недоступен на некоторых GPU (32*32=1024 тредов — максимум),
# поэтому включаем, но ловим ошибку
CONFIGS = [
    (8,  "naive"),
    (16, "naive"),
    (32, "naive"),
    (8,  "tiled"),
    (16, "tiled"),
    (32, "tiled"),
]


def save_matrix(mat, name):
    np.savetxt(name, mat, fmt="%d", delimiter=" ")


def run_once(n: int, block: int, kernel: str):
    """Генерирует матрицы, запускает бинарь, возвращает (ms, gflops, ok)."""
    mat_a = np.random.randint(0, 10, (n, n))
    mat_b = np.random.randint(0, 10, (n, n))
    save_matrix(mat_a, "matrix_a")
    save_matrix(mat_b, "matrix_b")

    result = subprocess.run(
        [BINARY, str(block), kernel],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return None, None, False, result.stderr.strip()

    ms, gflops = None, None
    for line in result.stdout.splitlines():
        if line.startswith("Time"):
            ms = float(line.split(":")[1].strip().split()[0])
        if line.startswith("Performance"):
            gflops = float(line.split(":")[1].strip().split()[0])

    # Верификация
    try:
        c_cuda = np.loadtxt("result_matrix", dtype=int)
        c_ref  = mat_a @ mat_b
        ok = np.array_equal(c_cuda, c_ref)
    except Exception:
        ok = False

    return ms, gflops, ok, ""


def main():
    if not os.path.exists(BINARY):
        print(f"Бинарь не найден: {BINARY}")
        print("Собери: mkdir build && cd build && cmake .. && cmake --build . -j4")
        sys.exit(1)

    # results[n][(block, kernel)] = (ms, gflops)
    results = {n: {} for n in SIZES}

    for n in SIZES:
        print(f"\n{'='*68}")
        print(f"  Матрица {n}×{n}")
        print(f"{'='*68}")
        for block, kernel in CONFIGS:
            ms, gflops, ok, err = run_once(n, block, kernel)
            key = (block, kernel)
            if ms is None:
                print(f"  block={block:2d}  kernel={kernel:<5}  ERROR: {err[:60]}")
                continue
            results[n][key] = (ms, gflops)
            status = "✓" if ok else "✗ MISMATCH"
            print(f"  block={block:2d}  kernel={kernel:<5}  "
                  f"time={ms:9.2f} ms  {gflops:6.2f} GFLOPS  verify={status}")

    # ── Сводная таблица ускорений (tiled-16 как базовый) ──────────────────────
    print("\n" + "=" * 72)
    print("СВОДНАЯ ТАБЛИЦА (время, мс)")
    print("=" * 72)
    col_keys = CONFIGS
    header = f"{'N':>6} | " + " | ".join(
        f"b={b} {k:<5}" for b, k in col_keys
    )
    print(header)
    print("-" * 72)
    for n in SIZES:
        row = f"{n:>6} |"
        for key in col_keys:
            entry = results[n].get(key)
            if entry is None:
                row += "    N/A   |"
            else:
                row += f" {entry[0]:8.2f} |"
        print(row)

    # ── Таблица ускорений относительно naive-16 ───────────────────────────────
    print("\n" + "=" * 72)
    print("УСКОРЕНИЕ relative to naive block=16")
    print("=" * 72)
    print(f"{'N':>6} | " + " | ".join(f"b={b} {k:<5}" for b, k in col_keys))
    print("-" * 72)
    for n in SIZES:
        base = results[n].get((16, "naive"), (None,))[0]
        row = f"{n:>6} |"
        for key in col_keys:
            entry = results[n].get(key)
            if entry is None or base is None:
                row += "    N/A   |"
            else:
                sp = base / entry[0]
                row += f"   x{sp:.2f}   |"
        print(row)


if __name__ == "__main__":
    main()
