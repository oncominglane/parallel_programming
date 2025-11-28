# plot_speedup.py
#
# CSV: algo,N,p,repeat,time_sec
# Строит для каждого алгоритма графики:
#   - ускорения S(p) = T1 / Tp
#   - эффективности E(p) = S(p) / p
#
# T1 берётся как среднее время для *_seq при p=1
# внутри одного семейства (etalon, var2zh, var3z, var1e).

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/lab2_results.csv"
out_speedup = sys.argv[2] if len(sys.argv) > 2 else "results/speedup_vs_p.png"
out_eff = sys.argv[3] if len(sys.argv) > 3 else "results/efficiency_vs_p.png"

if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    sys.exit(1)

# Читаем CSV
df = pd.read_csv(csv_path, comment="#")

# Чистим типы
df["N"] = pd.to_numeric(df["N"], errors="coerce")
df["p"] = pd.to_numeric(df["p"], errors="coerce")
df["repeat"] = pd.to_numeric(df["repeat"], errors="coerce")
df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
df = df.dropna(subset=["algo", "N", "p", "repeat", "time_sec"])

# Усредняем по повторам
g = df.groupby(["algo", "N", "p"], as_index=False)["time_sec"].mean()

# Выделяем "семейство" алгоритма: часть до первого "_"
# etalon_seq -> etalon, var2zh_omp -> var2zh, var1e_mpi -> var1e и т.д.
g["family"] = g["algo"].str.extract(r"^([^_]+)")

# Строим таблицу базовых времен T1 для каждого (family, N)
# Берём *_seq при p=1, если есть
baselines = {}  # (family, N) -> T1

for (family, N), sub in g.groupby(["family", "N"]):
    # Кандидаты: *_seq при p=1
    seq_name = f"{family}_seq"
    sub_seq = sub[(sub["algo"] == seq_name) & (sub["p"] == 1)]
    if not sub_seq.empty:
        T1 = sub_seq["time_sec"].iloc[0]
        baselines[(family, N)] = T1
    else:
        # fallback: минимальное время при p=1 для этого семейства
        sub_p1 = sub[sub["p"] == 1]
        if not sub_p1.empty:
            T1 = sub_p1["time_sec"].min()
            baselines[(family, N)] = T1

# Считаем S(p) и E(p)
speedup = []
eff = []

for idx, row in g.iterrows():
    family = row["family"]
    N = row["N"]
    p = row["p"]
    t = row["time_sec"]
    key = (family, N)

    if p <= 0 or key not in baselines:
        speedup.append(float("nan"))
        eff.append(float("nan"))
        continue

    T1 = baselines[key]
    S = T1 / t if t > 0 else float("nan")
    E = S / p if p > 0 else float("nan")

    speedup.append(S)
    eff.append(E)

g["speedup"] = speedup
g["efficiency"] = eff

# Для построения возьмём максимальный N (обычно единственный)
if g["N"].nunique() > 1:
    N_plot = g["N"].max()
else:
    N_plot = g["N"].iloc[0]

gN = g[g["N"] == N_plot].copy()

# Убираем записи, где ускорение не посчитано
gN = gN[~gN["speedup"].isna()]

# --- ГРАФИК УСКОРЕНИЯ ---

plt.figure(figsize=(8, 5))
for algo, sub in gN.groupby("algo"):
    # Не рисуем чисто последовательные *_seq
    if algo.endswith("_seq"):
        continue
    sub = sub.sort_values("p")
    plt.plot(sub["p"], sub["speedup"], marker="o", label=algo)

plt.xlabel("Число исполнителей p")
plt.ylabel("speedup S(p) = T₁ / Tₚ")
plt.title(f"speedup (N = {int(N_plot)})")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(out_speedup, dpi=150)
print(f"Saved speedup plot to {out_speedup}")

# --- ГРАФИК ЭФФЕКТИВНОСТИ ---

plt.figure(figsize=(8, 5))
for algo, sub in gN.groupby("algo"):
    if algo.endswith("_seq"):
        continue
    sub = sub.sort_values("p")
    plt.plot(sub["p"], sub["efficiency"], marker="o", label=algo)

plt.xlabel("Число исполнителей p")
plt.ylabel("efficiency E(p) = S(p) / p")
plt.title(f"efficiency (N = {int(N_plot)})")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(out_eff, dpi=150)
print(f"Saved efficiency plot to {out_eff}")
