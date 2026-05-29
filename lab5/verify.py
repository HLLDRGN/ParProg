import numpy as np

def read_matrix_from_file(filename: str) -> np.ndarray:
    return np.loadtxt(filename, dtype=int)

def verify_cpp_result(a_file: str, b_file: str, c_file: str) -> None:
    A = read_matrix_from_file(a_file)
    B = read_matrix_from_file(b_file)
    C_cpp = read_matrix_from_file(c_file)

    C_py = A @ B

    if C_py.shape != C_cpp.shape:
        print("Размеры не совпали")
        print("C++:", C_cpp.shape, "Python:", C_py.shape)
        return

    if np.array_equal(C_py, C_cpp):
        print("ОК: результат C++ совпадает с NumPy.")
    else:
        print("ОШИБКА: результаты не совпали.")
        print("Разность (Python - C++):")
        print(C_py - C_cpp)

if __name__ == "__main__":
    verify_cpp_result("matrix_a", "matrix_b", "result_matrix")
