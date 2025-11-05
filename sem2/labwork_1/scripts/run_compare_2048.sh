#!/usr/bin/env bash
# Сравнение блочного, Штрассена и ГИБРИДНОГО (seq, OMP, SIMD) ТОЛЬКО для N=2048.
# Использование:
#   bash run_compare_2048.sh [OUT_CSV]
#
# Параметры через переменные окружения:
#   REPEAT=3        # число повторов в новом config.h
#   RNG=42          # сид для генерации матриц
#   TILE=64         # тайл для blocked (seq/OMP/SIMD)
#   LEAF=64         # порог листа для Strassen (seq/SIMD)
#   CUT=256         # порог создания tasks для OMP-Strassen
#   THREADS=8       # OMP_NUM_THREADS
#   HYBRID_LEAF=128 # порог листа для гибрида (seq/OMP/SIMD)
#   HYBRID_T=64     # тайл листа для гибрида (seq/OMP/SIMD)
#   HYBRID_CUT=256  # порог создания tasks для OMP-гибрида
#
# Скрипт временно переписывает config.h -> только SIZES={2048} и REPEAT, затем восстанавливает.

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
CXXFLAGS_OMP="${CXXFLAGS_BASE} -fopenmp"
CXXFLAGS_SIMD="${CXXFLAGS_BASE} -mavx2 -mfma"
CXXFLAGS_OMPSIMD="${CXXFLAGS_BASE} -fopenmp -mavx2 -mfma"

# ---------- PARAMS ----------
OUT=${1:-"$RESULTS/results_2048.csv"}
N=2048
REPEAT=${REPEAT:-3}
RNG=${RNG:-42}
TILE=${TILE:-64}
LEAF=${LEAF:-64}
CUT=${CUT:-256}
THREADS=${THREADS:-8}

HYBRID_LEAF=${HYBRID_LEAF:-128}
HYBRID_T=${HYBRID_T:-64}
HYBRID_CUT=${HYBRID_CUT:-256}

# ---------- config.h override ----------
CFG="$SRC/config.h"
if [ ! -f "$CFG" ]; then
  echo "ERROR: $CFG not found." >&2
  exit 1
fi
cp -f "$CFG" "$CFG.bak"
cat > "$CFG" <<EOF
#pragma once
#include <vector>
static const std::vector<int> SIZES = { ${N} };
static const int REPEAT = ${REPEAT};
static const unsigned int RNG_SEED = ${RNG};
EOF

# ---------- BUILD ----------
# seq
$CXX $CXXFLAGS_BASE   -I"$INC" "$SRC/blocked_mm.cpp"        -o "$BUILD/blocked_mm"
$CXX $CXXFLAGS_BASE   -I"$INC" "$SRC/strassen_mm.cpp"       -o "$BUILD/strassen_mm"
$CXX $CXXFLAGS_BASE   -I"$INC" "$SRC/hybrid_mm.cpp"         -o "$BUILD/hybrid_mm"

# OMP
$CXX $CXXFLAGS_OMP    -I"$INC" "$SRC/blocked_omp_mm.cpp"    -o "$BUILD/blocked_omp_mm"
$CXX $CXXFLAGS_OMP    -I"$INC" "$SRC/strassen_omp_mm.cpp"   -o "$BUILD/strassen_omp_mm"
$CXX $CXXFLAGS_OMP    -I"$INC" "$SRC/hybrid_omp_mm.cpp"     -o "$BUILD/hybrid_omp_mm"

# SIMD
$CXX $CXXFLAGS_SIMD   -I"$INC" "$SRC/blocked_simd_mm.cpp"   -o "$BUILD/blocked_simd_mm"
$CXX $CXXFLAGS_SIMD   -I"$INC" "$SRC/strassen_simd_mm.cpp"  -o "$BUILD/strassen_simd_mm"
$CXX $CXXFLAGS_SIMD   -I"$INC" "$SRC/hybrid_simd_mm.cpp"    -o "$BUILD/hybrid_simd_mm"

# OMP + SIMD
$CXX $CXXFLAGS_OMPSIMD -I"$INC" "$SRC/hybrid_ompsimd_mm.cpp" -o "$BUILD/hybrid_ompsimd_mm"

# ---------- RUN ----------
rm -f "$OUT"

# seq
"$BUILD/blocked_mm"   "$OUT" "$TILE"
"$BUILD/strassen_mm"  "$OUT" "$LEAF"
"$BUILD/hybrid_mm"    "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# OMP env (стабильность)
export OMP_DYNAMIC=false
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS="$THREADS"

# OMP
"$BUILD/blocked_omp_mm"   "$OUT" "$TILE"
"$BUILD/strassen_omp_mm"  "$OUT" "$LEAF" "$CUT"
"$BUILD/hybrid_omp_mm"    "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"

# SIMD
"$BUILD/blocked_simd_mm"  "$OUT" "$TILE"
"$BUILD/strassen_simd_mm" "$OUT" "$LEAF"
"$BUILD/hybrid_simd_mm"   "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# OMP+SIMD
"$BUILD/hybrid_ompsimd_mm" "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"

# ---------- SUMMARY ----------
echo
echo "=== Summary (N=${N}, average over REPEAT=${REPEAT}) ==="
printf "%-60s %12s\n" "algo" "avg_time_s"
echo "------------------------------------------------------------ ------------"
LC_ALL=C awk -F, -v N="$N" '
  NR>1 && $2==N { t=$4+0.0; sum[$1]+=t; cnt[$1]++ }
  END { for (a in sum) printf "%-60s %12.6f\n", a, sum[a]/cnt[a]; }
' "$OUT" | sort -k2,2n

echo
echo "Raw data saved to: $OUT"

# ---------- restore config.h ----------
mv -f "$CFG.bak" "$CFG"