"""
verify_cuda.py — верификация результата CUDA через NumPy
Использование: python verify_cuda.py
"""
import numpy as np
import sys


def read(path):
    return np.loadtxt(path, dtype=int)


def main():
    try:
        A = read("matrix_a")
        B = read("matrix_b")
        C_cuda = read("result_matrix")
    except OSError as e:
        print(f"Ошибка чтения файла: {e}")
        sys.exit(1)

    C_ref = A @ B

    if C_ref.shape != C_cuda.shape:
        print(f"ОШИБКА: размеры не совпадают — ref {C_ref.shape}, cuda {C_cuda.shape}")
        sys.exit(1)

    if np.array_equal(C_ref, C_cuda):
        print("VERIFICATION: PASSED — результат CUDA совпадает с NumPy.")
    else:
        diff = C_ref - C_cuda
        print("ОШИБКА: результаты не совпали.")
        print(f"  Максимальное отклонение : {np.abs(diff).max()}")
        print(f"  Элементов с ошибкой     : {np.count_nonzero(diff)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
