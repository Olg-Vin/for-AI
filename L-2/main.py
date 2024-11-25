from first import scale_by_max
from computeCost import computeCost_elementwise, computeCost_vectorized
from gradientDescent import gradientDescent_elementwise, gradientDescent_vectorized, plot_j

data = []
with open('../ex1data2.txt', 'r') as file:
    for line in file:
        data.append([float(num) for num in line.split(',')])


# Нормализуем входные данные
scale_data = scale_by_max(data)

# Преобразуем данные в нужный формат
X = [[1, row[0], row[1]] for row in scale_data]
y = [row[2] for row in scale_data]

# Входнфе данные
theta = [0, 0, 0]
iterations = 2000
alpha = 0.001

# Считаем градиент
initial_cost = computeCost_elementwise(X, y, theta)
theta, cost_history, theta_history = gradientDescent_elementwise(X, y, theta, alpha, iterations)

