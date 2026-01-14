# plot_time_vs_N.py
#
# CSV: algo,N,p,repeat,time_sec
#
# algo:
#   bvp_seq      - последовательная (Thomas)
#   bvp_red_seq  - последовательная с редукцией
#   bvp_red_omp  - параллельная редукция (p = 4)
#
# Строит три графика:
#   1) time_vs_N.png       : x = N, y = среднее время time_sec
#   2) speedup_vs_N.png    : x = N, y = ускорение S(N) = T_seq(N) / T_algo(N)
#   3) efficiency_vs_N.png : x = N, y = эффективность E(N) = S(N) / p
#
# Использование:
#   python3 plot_time_vs_N.py [csv_path] [out_time_png] [out_speedup_png] [out_eff_png]

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/bvp_results.csv"
out_time = sys.argv[2] if len(sys.argv) > 2 else "results/time_vs_N.png"
out_speedup = sys.argv[3] if len(sys.argv) > 3 else "results/speedup_vs_N.png"
out_eff = sys.argv[4] if len(sys.argv) > 4 else "results/efficiency_vs_N.png"

if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    sys.exit(1)

print(f"Reading CSV: {csv_path}")

# --- читаем CSV ---
df = pd.read_csv(csv_path, comment="#")

# типы
df["N"] = pd.to_numeric(df["N"], errors="coerce")
df["p"] = pd.to_numeric(df["p"], errors="coerce")
df["repeat"] = pd.to_numeric(df["repeat"], errors="coerce")
df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")

df = df.dropna(subset=["algo", "N", "p", "repeat", "time_sec"])

# --- усредняем по повторам ---
g = df.groupby(["algo", "N", "p"], as_index=False)["time_sec"].mean()

print("Algorithms found:", sorted(g["algo"].unique()))
print("N values:", sorted(g["N"].unique()))

# ================= ГРАФИК ВРЕМЕНИ vs N =================

plt.figure(figsize=(8, 5))
for algo, sub in g.groupby("algo"):
    sub = sub.sort_values("N")
    # если для алгоритма есть несколько p (теоретически), берём p с макс. ускорением
    # но в нашей лабе для каждого algo только одно p
    plt.plot(sub["N"], sub["time_sec"], marker="o", label=algo)

plt.xlabel("Число точек N")
plt.ylabel("Время исполнения, сек")
plt.title("Время исполнения vs число точек")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(out_time, dpi=150)
print(f"Saved time vs N plot to {out_time}")

# ================= УСКОРЕНИЕ И ЭФФЕКТИВНОСТЬ =================

# Базовый алгоритм: bvp_seq (последовательная с прогонкой)
baseline_algo = "bvp_seq"

# таблица базовых времен T1(N) для bvp_seq, p=1
base = g[(g["algo"] == baseline_algo) & (g["p"] == 1)].copy()
if base.empty:
    print(f"No baseline rows for algo='{baseline_algo}', p=1. Cannot compute speedup.")
    sys.exit(1)

T1_by_N = dict(zip(base["N"], base["time_sec"]))

# считаем S(N, algo) и E(N, algo)
speedup_vals = []
eff_vals = []

for _, row in g.iterrows():
    algo = row["algo"]
    N = row["N"]
    p = row["p"]
    t = row["time_sec"]

    if N not in T1_by_N or t <= 0 or p <= 0:
        speedup_vals.append(float("nan"))
        eff_vals.append(float("nan"))
        continue

    T1 = T1_by_N[N]
    S = T1 / t
    E = S / p

    speedup_vals.append(S)
    eff_vals.append(E)

g["speedup"] = speedup_vals
g["efficiency"] = eff_vals

# ================= ГРАФИК УСКОРЕНИЯ vs N =================

plt.figure(figsize=(8, 5))

for algo, sub in g.groupby("algo"):
    if algo == baseline_algo:
        # базовый алгоритм имеет S=1, можно не рисовать
        continue
    sub = sub.sort_values("N")
    plt.plot(sub["N"], sub["speedup"], marker="o", label=algo)

plt.xlabel("Число точек N")
plt.ylabel("Ускорение S(N) = T_seq(N) / T_algo(N)")
plt.title("Ускорение vs число точек (относительно bvp_seq)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(out_speedup, dpi=150)
print(f"Saved speedup vs N plot to {out_speedup}")

# ================= ГРАФИК ЭФФЕКТИВНОСТИ vs N =================

plt.figure(figsize=(8, 5))

for algo, sub in g.groupby("algo"):
    if sub["p"].iloc[0] <= 1:
        continue
    sub = sub.sort_values("N")
    plt.plot(sub["N"], sub["efficiency"], marker="o", label=algo)

plt.xlabel("Число точек N")
plt.ylabel("Эффективность E(N) = S(N) / p")
plt.title("Эффективность vs число точек")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(out_eff, dpi=150)
print(f"Saved efficiency vs N plot to {out_eff}")
