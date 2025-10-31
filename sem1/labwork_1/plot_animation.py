import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Загрузка данных
data = np.loadtxt("sequential_transfer_eq.txt")  # или "solution_parallel.txt"

K, M = data.shape
X = 1.0
x = np.linspace(0, X, M)

# Создание фигуры и осей
fig, ax = plt.subplots()
line, = ax.plot(x, data[0, :])
ax.set_xlim(0, 1)
ax.set_ylim(np.min(data), np.max(data))
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.set_title("Анимация распространения волны")

# Функция обновления графика
def update(frame):
    line.set_ydata(data[frame, :])
    ax.set_title(f"timestep {frame}/{K}")
    return line,

# Создание анимации
ani = animation.FuncAnimation(fig, update, frames=K, interval=100, blit=True)

plt.show()
