#!/usr/bin/env bash
# Сравнение параллельных (OpenMP) версий: omp_blocked, omp_strassen и omp_hybrid.
# Использование:
#   bash run_matrix_bench_omp.sh [OUT_CSV]
#
# Переменные окружения:
#   THREADS="1 2 4 8"      # список значений OMP_NUM_THREADS
#   TILES="32 64 128"      # размеры тайла для omp_blocked
#   LEAF=64                # порог листа для Strassen (переключение на обычное умножение)
#   CUT=256                # минимальный размер, при котором ещё создаём OpenMP-задачи (Strassen)
#   HYBRID_LEAF=128        # порог листа для гибрида
#   HYBRID_T=64            # тайл листа для гибрида
#   HYBRID_CUT=256         # порог создания задач для гибрида
#   APPEND=1               # не удалять OUT_CSV перед запуском
#   CXX, CXXFLAGS          # переопределить компилятор/флаги

set -euo pipefail

# ---------- PATHS ----------
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$ROOT/src"
INC="$SRC"
BUILD="$ROOT/build"
RESULTS="$ROOT/results"
mkdir -p "$BUILD" "$RESULTS"

# ---------- COMPILER / FLAGS ----------
CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:--O3 -march=native -std=c++17 -fopenmp}

# ---------- PARAMS ----------
THREADS=${THREADS:-"8"}       # "1 2 4 8"
TILES=${TILES:-"64"}          # "32 64 128"
LEAF=${LEAF:-64}              # порог листа для Strassen
CUT=${CUT:-256}               # порог создания задач (Strassen)
HYBRID_LEAF=${HYBRID_LEAF:-128}
HYBRID_T=${HYBRID_T:-64}
HYBRID_CUT=${HYBRID_CUT:-256}
OUT=${1:-"$RESULTS/results_omp.csv"}

# ---------- BUILD ----------
$CXX $CXXFLAGS -I"$INC" "$SRC/blocked_omp_mm.cpp"  -o "$BUILD/blocked_omp_mm"
$CXX $CXXFLAGS -I"$INC" "$SRC/strassen_omp_mm.cpp" -o "$BUILD/strassen_omp_mm"
$CXX $CXXFLAGS -I"$INC" "$SRC/hybrid_omp_mm.cpp"   -o "$BUILD/hybrid_omp_mm"

# ---------- CSV ----------
if [ "${APPEND:-0}" != "1" ]; then rm -f "$OUT"; fi

echo "Writing to: $OUT"
echo "THREADS:     $THREADS"
echo "TILES:       $TILES"
echo "LEAF:        $LEAF"
echo "CUT:         $CUT"
echo "HYBRID_LEAF: $HYBRID_LEAF"
echo "HYBRID_T:    $HYBRID_T"
echo "HYBRID_CUT:  $HYBRID_CUT"

# ---------- OMP STABILITY ----------
export OMP_DYNAMIC=${OMP_DYNAMIC:-false}
export OMP_PROC_BIND=${OMP_PROC_BIND:-true}
export OMP_PLACES=${OMP_PLACES:-cores}

# ---------- RUN ----------
for P in $THREADS; do
  export OMP_NUM_THREADS="$P"
  echo "== OMP_NUM_THREADS=$P =="

  # blocked + OMP с перебором размера тайла
  for T in $TILES; do
    "$BUILD/blocked_omp_mm"  "$OUT" "$T"
  done

  # Strassen + tasks
  "$BUILD/strassen_omp_mm"  "$OUT" "$LEAF" "$CUT"

  # Hybrid (верх — Strassen с задачами, лист — блочное ядро)
  "$BUILD/hybrid_omp_mm"    "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"
done

# защита CSV от запятой в метках (если вдруг встретится)
sed -i 's/omp_blocked(T=\([0-9]\+\),p=\([0-9]\+\))/omp_blocked(T=\1;p=\2)/g' "$OUT" || true

echo "Done. Results saved to $OUT"