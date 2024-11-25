import numpy as np


def computeCost_elementwise(X, y, theta):
    """
    Функция для расчёта функции стоимости регресии нескольких переменных

    :param X: список (m x n) - входные признаки (с добавлением столбца единиц для свободного члена)
    :param y: список (m x 1) - выходные признаки
    :param theta: список (1 x n) - веса
    :return: число - результат расчёта функции стоимости
    """
    m = len(y)
    cost = 0

    for i in range(m):
        prediction = 0
        for j in range(len(X[0])):
            prediction += theta[j] * X[i][j]

        error = prediction - y[i]
        cost += error ** 2

    return cost / (2 * m)


def computeCost_vectorized(X, y, theta):
    """
    Функция для расчёта функции стоимости регресии нескольких переменных в векторном виде

    :param X: список (m x n) - входные признаки
    :param y: список (m x 1) - выходные признаки
    :param theta: список (1 x n) - веса
    :return: число - результат расчёта функции стоимости
    """
    m = len(y)
    predictions = np.dot(X, theta)  # Векторное произведение для всех примеров
    errors = predictions - y
    cost = (1 / (2 * m)) * np.dot(errors.T, errors)  # Сумма квадратов ошибок
    return cost
