# plot_runtime.py
# Reads results.csv with columns: algo,N,repeat,time_sec
# Plots mean runtime vs N for each algorithm.
import pandas as pd
import matplotlib.pyplot as plt

import sys
csv_path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "runtime_vs_N.png"

df = pd.read_csv(csv_path)
# Aggregate: mean time for each (algo, N)
g = df.groupby(["algo","N"], as_index=False)["time_sec"].mean()

# Unique algos
algos = g["algo"].unique()

plt.figure(figsize=(7,5))
for algo in algos:
    sub = g[g["algo"] == algo].sort_values("N")
    plt.plot(sub["N"], sub["time_sec"], marker="o", label=algo)

plt.xlabel("Matrix size N (N x N)")
plt.ylabel("Mean runtime (s)")
plt.title("Runtime vs Matrix Size")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
