# plot_runtime.py
# CSV: algo,N,repeat,time_sec (иногда запятые бывают внутри algo)
import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/runtime_vs_N.png"

# Разделитель: запятая, НЕ находящаяся внутри круглых скобок
SEP = r',(?![^()]*\))'

# Читаем CSV; names на случай, если заголовок дублится
df = pd.read_csv(
    csv_path,
    engine='python',
    sep=SEP,
    header=0,
    names=['algo','N','repeat','time_sec'],
    comment='#'
)

# Убираем возможные строковые заголовки, пустые строки и приводим типы
df = df[df['algo'] != 'algo']
df['N'] = pd.to_numeric(df['N'], errors='coerce')
df['repeat'] = pd.to_numeric(df['repeat'], errors='coerce')
df['time_sec'] = pd.to_numeric(df['time_sec'], errors='coerce')
df = df.dropna(subset=['N','repeat','time_sec'])

# Группируем по среднему времени
g = df.groupby(['algo','N'], as_index=False)['time_sec'].mean()

plt.figure(figsize=(8,5))
for algo, sub in g.groupby('algo'):
    sub = sub.sort_values('N')
    plt.plot(sub['N'], sub['time_sec'], marker='o', label=algo)

plt.xlabel('Matrix size N (N x N)')
plt.ylabel('Mean runtime (s)')
plt.title('Runtime vs Matrix Size')
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
