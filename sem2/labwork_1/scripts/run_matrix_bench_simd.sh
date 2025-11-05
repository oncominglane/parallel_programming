#!/usr/bin/env bash
# Сравнение SIMD-реализаций: simd_blocked, simd_strassen и simd_hybrid.
# Использование:
#   bash run_matrix_bench_simd.sh [OUT_CSV]
# Переменные окружения:
#   TILES="32 64 128"   # набор T для simd_blocked
#   LEAF=64             # порог листа для simd_strassen
#   HYBRID_LEAF=128     # порог листа для simd_hybrid
#   HYBRID_T=64         # тайл листа для simd_hybrid
#   APPEND=1            # не удалять CSV перед запуском
#   CXX, CXXFLAGS       # переопределить компилятор/флаги

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
CXXFLAGS_BASE=${CXXFLAGS_BASE:--O3 -march=native -std=c++17}
# SIMD таргет (AVX2/FMA). При желании добавь: -ffast-math -funroll-loops
CXXFLAGS_SIMD="${CXXFLAGS_BASE} -mavx2 -mfma"

# ---------- PARAMS ----------
TILES=${TILES:-"64"}          # набор T для blocked_simd
LEAF=${LEAF:-64}              # порог листа для strassen_simd
HYBRID_LEAF=${HYBRID_LEAF:-128}
HYBRID_T=${HYBRID_T:-64}
OUT=${1:-"$RESULTS/results_simd.csv"}

# ---------- BUILD ----------
$CXX $CXXFLAGS_SIMD -I"$INC" "$SRC/blocked_simd_mm.cpp"  -o "$BUILD/blocked_simd_mm"
$CXX $CXXFLAGS_SIMD -I"$INC" "$SRC/strassen_simd_mm.cpp" -o "$BUILD/strassen_simd_mm"
$CXX $CXXFLAGS_SIMD -I"$INC" "$SRC/hybrid_simd_mm.cpp"   -o "$BUILD/hybrid_simd_mm"

# ---------- RUN ----------
if [ "${APPEND:-0}" != "1" ]; then rm -f "$OUT"; fi

echo "Writing to: $OUT"
echo "TILES:        $TILES"
echo "LEAF:         $LEAF"
echo "HYBRID_LEAF:  $HYBRID_LEAF"
echo "HYBRID_T:     $HYBRID_T"

# blocked_simd: перебор размеров тайла
for T in $TILES; do
  "$BUILD/blocked_simd_mm" "$OUT" "$T"
done

# strassen_simd: фиксированный leaf
"$BUILD/strassen_simd_mm" "$OUT" "$LEAF"

# hybrid_simd: Strassen + блочный лист с упаковкой + SIMD
"$BUILD/hybrid_simd_mm" "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# подстрахуем CSV, если вдруг в метках снова окажутся запятые
sed -i 's/simd_blocked(T=\([0-9]\+\),/simd_blocked(T=\1;/g' "$OUT" || true

echo "Done. Results saved to $OUT"