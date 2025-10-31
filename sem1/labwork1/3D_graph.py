

import numpy as np
import matplotlib.pyplot as plt

# Загрузка времени
with open("timings.txt", "r") as f:
    t_seq, t_par = map(float, f.readline().split())

speedup = t_seq / t_par

print(f"\n### Последовательная: {t_seq:.6f} сек")
print(f"### Параллельная (4 процесса): {t_par:.6f} сек")
print(f"### Ускорение: {speedup:.2f}x")

# Графики
data_seq = np.loadtxt("sequential_transfer_eq.txt")
data_par = np.loadtxt("parallel_transfer_eq.txt")

K, M = data_seq.shape
X = 1.0
T = 1.0
x = np.linspace(0, X, M)
t = np.linspace(0, T, K)
T_grid, X_grid = np.meshgrid(t, x, indexing='ij')

fig = plt.figure(figsize=(14, 6))

# Последовательное решение
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X_grid, T_grid, data_seq, cmap='viridis')
ax1.set_title("Последовательное решение")
ax1.set_xlabel("x")
ax1.set_ylabel("t")
ax1.set_zlabel("u(x, t)")

# Параллельное решение
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X_grid, T_grid, data_par, cmap='plasma')
ax2.set_title("Параллельное решение")
ax2.set_xlabel("x")
ax2.set_ylabel("t")
ax2.set_zlabel("u(x, t)")

plt.tight_layout()
plt.show()
