import numpy as np

def random_matrix_100x100(low=0, high=10):
    # матрица 100x100 из целых в [low, high)
    return np.random.randint(low, high, size=(100, 100))
def save_matrix_to_file(matrix: np.ndarray, filename: str) -> None:
  
    np.savetxt(filename, matrix, fmt="%d", delimiter=" ")

if __name__ == "__main__":
    matrix_a = random_matrix_100x100()
    matrix_b = random_matrix_100x100()
    save_matrix_to_file(matrix_a, filename="matrix_a")
    save_matrix_to_file(matrix_b, filename="matrix_b")