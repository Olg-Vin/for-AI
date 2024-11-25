from first import scale_by_max
import numpy as np
from computeCost import computeCost_elementwise, computeCost_vectorized
from gradientDescent import gradientDescent_elementwise, gradientDescent_vectorized, plot_j
from normalEquation import normal_equation, normal_equation_with_np

data = []
with open('../ex1data2.txt', 'r') as file:
    for line in file:
        data.append([float(num) for num in line.split(',')])

# Нормализуем входные данные
scale_data = scale_by_max(data)

# Преобразуем данные в нужный формат
X = [[1, row[0], row[1]] for row in scale_data]
y = [row[2] for row in scale_data]

print(f"расчёты без numpy \n")

# Входные данные
theta = [0, 0, 0]
iterations = 40000
alpha = 0.01

# 1. Градиентный спуск
theta_gd, cost_history, theta_history = gradientDescent_elementwise(X, y, theta, alpha, iterations)
print(f"Параметры theta после градиентного спуска: {theta_gd}")
gd_cost = computeCost_elementwise(X, y, theta_gd)
print(f"Функция стоимости для градиентного спуска: {gd_cost}")

# 2. Нормальное уравнение
theta_ne = normal_equation(X, y)
print(f"Параметры theta после нормального уравнения: {theta_ne}")
ne_cost = computeCost_elementwise(X, y, theta_ne)
print(f"Функция стоимости для нормального уравнения: {ne_cost}")

# Сравнение результатов
if abs(gd_cost - ne_cost) < 1e-6:
    print("Оба метода дали одинаково точные результаты!")
else:
    print("Методы дают разные результаты. Надо анализировать, почему.")

print()
print(f"расчёты с numpy \n")

# Входные данные
X = np.array(X)
y = np.array(y)

theta = [0, 0, 0]
iterations = 4500
alpha = 0.015

# 1. Градиентный спуск
theta_gd, cost_history, theta_history = gradientDescent_vectorized(X, y, theta, alpha, iterations)
print(f"Параметры theta после градиентного спуска: {theta_gd}")
gd_cost = computeCost_vectorized(X, y, theta_gd)
print(f"Функция стоимости для градиентного спуска: {gd_cost}")

# 2. Нормальное уравнение
theta_ne = normal_equation_with_np(X, y)
print(f"Параметры theta после нормального уравнения: {theta_ne}")
ne_cost = computeCost_vectorized(X, y, theta_ne)
print(f"Функция стоимости для нормального уравнения: {ne_cost}")

# Сравнение результатов
if abs(gd_cost - ne_cost) < 1e-6:
    print("Оба метода дали одинаково точные результаты!")
else:
    print("Методы дают разные результаты. Надо анализировать, почему.")


