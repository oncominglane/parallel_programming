import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
df["speedup"] = df["T_seq"] / df["T_par"]

plt.plot(df["M"], df["speedup"], marker='o')
plt.xlabel("Размер задачи M")
plt.ylabel("Ускорение (T_seq / T_par)")
plt.title("Зависимость ускорения от размера задачи")
plt.grid(True)
plt.savefig("speedup_plot.png")
plt.show()
