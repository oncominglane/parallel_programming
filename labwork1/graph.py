
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Загрузка данных
data_seq = np.loadtxt("sequential_transfer_eq.txt")
data_par = np.loadtxt("parallel_transfer_eq.txt")

# Размеры сетки
K, M = data_seq.shape
X = 1.0
T = 1.0
x = np.linspace(0, X, M)
t = np.linspace(0, T, K)
T_grid, X_grid = np.meshgrid(t, x, indexing='ij')

# Построение
fig = plt.figure(figsize=(14, 6))

# Последовательное решение
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X_grid, T_grid, data_seq, cmap='viridis')
ax1.set_title("Последовательное решение")
ax1.set_xlabel("x")
ax1.set_ylabel("t")
ax1.set_zlabel("u(x, t)")

# Параллельное решение
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X_grid, T_grid, data_par, cmap='plasma')
ax2.set_title("Параллельное решение")
ax2.set_xlabel("x")
ax2.set_ylabel("t")
ax2.set_zlabel("u(x, t)")

plt.tight_layout()
plt.show()
