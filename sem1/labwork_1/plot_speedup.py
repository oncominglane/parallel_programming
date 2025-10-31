import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv("results.csv", skiprows=1, names=["M", "K", "P", "T_seq", "T_par"])

# Преобразование типов
df = df.astype({
    "M": int,
    "K": int,
    "P": int,
    "T_seq": float,
    "T_par": float
})

# Вычисление метрик
df["speedup"] = df["T_seq"] / df["T_par"]
df["efficiency"] = df["speedup"] / df["P"]

# Уникальные значения P
ps = sorted(df["P"].unique())

# Создание общей фигуры
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# График ускорения
for p in ps:
    sub = df[df["P"] == p]
    axs[0].plot(sub["M"], sub["speedup"], marker='o', label=f"{p} process")

axs[0].set_title("speedup")
axs[0].set_xlabel("task size M")
axs[0].set_ylabel("speedup = T_seq / T_par")
axs[0].grid(True)
axs[0].legend()

# График эффективности
for p in ps:
    sub = df[df["P"] == p]
    axs[1].plot(sub["M"], sub["efficiency"], marker='s', label=f"{p} process")

axs[1].set_title("Efficiency")
axs[1].set_xlabel("task size M")
axs[1].set_ylabel("efficiency = speedup / P")
axs[1].grid(True)
axs[1].legend()

# Финальная настройка
plt.tight_layout()
plt.savefig("plot_speedup_efficiency_combined.png")
plt.show()
