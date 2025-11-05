#!/usr/bin/env bash
# Сравнение последовательных реализаций: naive, transpose, blocked, strassen, hybrid.
# Компилирует из src/, бинарники в build/, результаты в results/.

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
CXXFLAGS=${CXXFLAGS:--O3 -march=native -std=c++17}

# ---------- OUTPUT ----------
OUT=${1:-"$RESULTS/results_seq.csv"}

# ---------- PARAMS ----------
TILES=${TILES:-"64"}          # список T для blocked
LEAF=${LEAF:-64}              # leaf для strassen
HYBRID_LEAF=${HYBRID_LEAF:-128}  # leaf для гибрида
HYBRID_T=${HYBRID_T:-64}         # T для гибрида

# ---------- BUILD ----------
$CXX $CXXFLAGS -I"$INC" "$SRC/naive_mm.cpp"      -o "$BUILD/naive_mm"
$CXX $CXXFLAGS -I"$INC" "$SRC/transpose_mm.cpp"  -o "$BUILD/transpose_mm"
$CXX $CXXFLAGS -I"$INC" "$SRC/blocked_mm.cpp"    -o "$BUILD/blocked_mm"
$CXX $CXXFLAGS -I"$INC" "$SRC/strassen_mm.cpp"   -o "$BUILD/strassen_mm"
$CXX $CXXFLAGS -I"$INC" "$SRC/hybrid_mm.cpp"     -o "$BUILD/hybrid_mm"

# ---------- RUN ----------
rm -f "$OUT"

# naive / transpose
"$BUILD/naive_mm"     "$OUT"
"$BUILD/transpose_mm" "$OUT"

# blocked с перебором тайла
for T in $TILES; do
  "$BUILD/blocked_mm" "$OUT" "$T"
done

# strassen
"$BUILD/strassen_mm" "$OUT" "$LEAF"

# hybrid (Strassen сверху + блочный лист с локальной упаковкой)
"$BUILD/hybrid_mm"   "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

echo "Wrote results to $OUT"
echo "Params: TILES={$TILES}  STRASSEN_LEAF=$LEAF  HYBRID_LEAF=$HYBRID_LEAF  HYBRID_T=$HYBRID_T"