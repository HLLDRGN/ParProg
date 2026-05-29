"""
generate_matrix.py — генерация тестовых матриц заданного размера
Использование: python generate_matrix.py [N]
По умолчанию N=512
"""
import numpy as np
import sys


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    mat_a = np.random.randint(0, 10, (n, n))
    mat_b = np.random.randint(0, 10, (n, n))
    np.savetxt("matrix_a", mat_a, fmt="%d", delimiter=" ")
    np.savetxt("matrix_b", mat_b, fmt="%d", delimiter=" ")
    print(f"Сгенерированы matrix_a и matrix_b размером {n}×{n}")


if __name__ == "__main__":
    main()
