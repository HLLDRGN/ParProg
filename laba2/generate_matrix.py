import numpy as np
import sys

def random_matrix(n: int, low=0, high=10) -> np.ndarray:
    return np.random.randint(low, high, size=(n, n))

def save_matrix(matrix: np.ndarray, filename: str) -> None:
    np.savetxt(filename, matrix, fmt="%d", delimiter=" ")

if __name__ == "__main__":
    # Использование: python generate_matrix.py 400
    # По умолчанию 100x100
    n = int(sys.argv[1]) if len(sys.argv) >= 2 else 100

    matrix_a = random_matrix(n)
    matrix_b = random_matrix(n)
    save_matrix(matrix_a, "matrix_a")
    save_matrix(matrix_b, "matrix_b")
    print(f"Generated {n}x{n} matrices -> matrix_a, matrix_b")
