import numpy as np


# Функция для нормального уравнения с использованием Numpy
def normal_equation_with_np(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Функция для нормального уравнения без Numpy

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def matrix_multiply(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Matrices cannot be multiplied")

    result = [[0] * cols_B for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return result


def gauss_jordan(matrix):
    n = len(matrix)
    # Создаем расширенную матрицу с единичной матрицей
    augmented = [row + [1 if i == j else 0 for i in range(n)] for j, row in enumerate(matrix)]

    # Приводим матрицу к верхнему треугольному виду
    for i in range(n):
        # Находим строку с максимальным элементом в столбце
        max_row = max(range(i, n), key=lambda r: abs(augmented[r][i]))
        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

        # Делаем диагональный элемент равным 1
        divisor = augmented[i][i]
        augmented[i] = [x / divisor for x in augmented[i]]

        # Обнуляем остальные элементы в столбце
        for j in range(n):
            if j != i:
                factor = augmented[j][i]
                augmented[j] = [augmented[j][k] - factor * augmented[i][k] for k in range(2 * n)]

    # Отделяем правую часть расширенной матрицы (обратную матрицу)
    return [row[n:] for row in augmented]


def normal_equation(X, y):
    # Транспонируем матрицу X
    X_transpose = transpose(X)

    # Умножаем X.T на X
    X_transpose_X = matrix_multiply(X_transpose, X)

    # Находим обратную матрицу для X.T * X
    X_transpose_X_inv = gauss_jordan(X_transpose_X)

    # Умножаем (X.T * X)^-1 на X.T
    X_transpose_X_inv_X_transpose = matrix_multiply(X_transpose_X_inv, X_transpose)

    # Умножаем результат на y
    theta = matrix_multiply(X_transpose_X_inv_X_transpose, [[yi] for yi in y])

    return [row[0] for row in theta]  # Извлекаем значения из списка списков
