from computeCost import computeCost_elementwise, computeCost_vectorized
import numpy as np
import matplotlib.pyplot as plt


def gradientDescent_elementwise(X, y, theta, alpha, iterations):
    """
    Функция для выполнения градиентного спуска для линейной регрессии с несколькими переменными

    :param X: список (m x n) - входные признаки (с добавлением столбца единиц для свободного члена)
    :param y: список (m x 1) - выходные признаки
    :param theta: список (1 x n) - начальные веса
    :param alpha: скорость обучения
    :param iterations: количество итераций градиентного спуска
    :return: обновленные параметры theta, история стоимости (cost_history), история параметров (theta_history)
    """
    m = len(y)  # количество примеров
    cost_history = []  # история стоимости
    theta_history = []  # история значений theta

    for _ in range(iterations):
        # Прогнозирование (предсказания) для всех примеров
        predictions = [sum(theta[j] * X[i][j] for j in range(len(theta))) for i in range(m)]

        # Вычисляем ошибки
        errors = [predictions[i] - y[i] for i in range(m)]

        # Обновление каждого параметра theta
        for j in range(len(theta)):
            theta[j] -= (alpha / m) * sum(errors[i] * X[i][j] for i in range(m))

        # Сохраняем историю значений theta и стоимости
        theta_history.append(theta[:])
        cost_history.append(computeCost_elementwise(X, y, theta))  # Считаем стоимость на каждой итерации

    # Визуализация изменения стоимости
    plot_j(iterations, cost_history)

    return theta, cost_history, theta_history



def gradientDescent_vectorized(X, y, theta, alpha, iterations):
    """
    Функция для выполнения градиентного спуска для линейной регрессии с несколькими переменными.

    :param X: Матрица входных данных (m x n), где m - количество примеров, n - количество признаков.
    :param y: Вектор выходных данных (m x 1), где m - количество примеров.
    :param theta: Начальные параметры (веса) модели (n x 1).
    :param alpha: Скорость обучения.
    :param iterations: Количество итераций.

    :return: Обновленные параметры theta, история стоимости, история значений theta.
    """
    m = len(y)  # количество примеров
    cost_history = []  # история стоимости
    theta_history = []  # история значений theta

    # Главный цикл градиентного спуска
    for _ in range(iterations):
        # Прогнозирование (предсказания) для всех примеров
        predictions = np.dot(X, theta)  # X.dot(theta) — вычисляем предсказания

        # Вычисляем ошибки
        errors = predictions - y

        # Градиент для всех параметров (градиенты для каждого признака)
        gradients = (1 / m) * np.dot(X.T, errors)

        # Обновление каждого параметра theta
        theta -= alpha * gradients

        # Сохраняем историю значений theta и стоимости
        theta_history.append(theta.copy())
        cost_history.append(computeCost_vectorized(X, y, theta))  # Считаем стоимость на каждой итерации

    # Визуализация изменения стоимости
    plot_j(iterations, cost_history)

    return theta, cost_history, theta_history


def plot_j(num_iterations, cost_history):
    plt.plot(range(num_iterations), cost_history, color='blue')
    plt.xlabel("Количество итераций")
    plt.ylabel("Функция потерь")
    plt.title("График функции потерь при градиентном спуске")
    plt.grid(True)
    plt.show()
