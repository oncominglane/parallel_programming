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

CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:--O3 -march=native -std=c++17 -fopenmp}

THREADS=${THREADS:-"8"}
TILES=${TILES:-"64"}
LEAF=${LEAF:-64}
CUT=${CUT:-256}
HYBRID_LEAF=${HYBRID_LEAF:-128}
HYBRID_T=${HYBRID_T:-64}
HYBRID_CUT=${HYBRID_CUT:-256}
OUT=${1:-results_omp.csv}

# --- сборка ---
mkdir -p build
$CXX $CXXFLAGS omp_blocked_mm.cpp -o build/omp_blocked_mm

# Страссен: поддержим оба названия файла (strassen_omp.cpp или omp_strassen_mm.cpp)
if [ -f strassen_omp.cpp ]; then
  $CXX $CXXFLAGS strassen_omp.cpp -o build/omp_strassen_mm
else
  $CXX $CXXFLAGS omp_strassen_mm.cpp -o build/omp_strassen_mm
fi

# Гибрид (Strassen + tasks, leaf = блочное ядро с упаковкой)
$CXX $CXXFLAGS omp_hybrid_mm.cpp -o build/omp_hybrid_mm

# --- CSV: подготовка один раз ---
if [ "${APPEND:-0}" != "1" ]; then rm -f "$OUT"; fi

echo "Writing to: $OUT"
echo "THREADS:      $THREADS"
echo "TILES:        $TILES"
echo "LEAF:         $LEAF"
echo "CUT:          $CUT"
echo "HYBRID_LEAF:  $HYBRID_LEAF"
echo "HYBRID_T:     $HYBRID_T"
echo "HYBRID_CUT:   $HYBRID_CUT"

# --- стабильность замеров ---
export OMP_DYNAMIC=${OMP_DYNAMIC:-false}
export OMP_PROC_BIND=${OMP_PROC_BIND:-true}
export OMP_PLACES=${OMP_PLACES:-cores}

# --- запуски ---
for P in $THREADS; do
  export OMP_NUM_THREADS="$P"
  echo "== OMP_NUM_THREADS=$P =="

  # Блочная + OMP с перебором тайлов
  for T in $TILES; do
    ./build/omp_blocked_mm "$OUT" "$T"
  done

  # Strassen + OMP tasks
  ./build/omp_strassen_mm "$OUT" "$LEAF" "$CUT"

  # Hybrid + OMP tasks (Strassen верх, leaf = блочный с упаковкой)
  ./build/omp_hybrid_mm "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"
done

# --- защита CSV от запятой внутри меток (если код ещё не экранирует) ---
sed -i 's/omp_blocked(T=\([0-9]\+\),p=\([0-9]\+\))/omp_blocked(T=\1;p=\2)/g' "$OUT"

echo "Done. Results saved to $OUT"
