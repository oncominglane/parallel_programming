#!/usr/bin/env bash
# Сравнение гибридов: seq, OMP, SIMD и OMP+SIMD (для всех SIZES из config.h)
# Использование:
#   bash run_matrix_bench_hybrids.sh [OUT_CSV]
#
# Переменные:
#   HYBRID_LEAF=128   # порог листа
#   HYBRID_T=64       # тайл в листе
#   HYBRID_CUT=256    # порог создания задач (OMP/OMP+SIMD)
#   THREADS=8         # OMP_NUM_THREADS
#   APPEND=0/1        # очищать CSV или дописывать
#   CXX, CXXFLAGS     # переопределить компилятор/флаги

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

# База для всех
CXXFLAGS_BASE=${CXXFLAGS_BASE:--O3 -march=native -std=c++17}
# Для OMP
CXXFLAGS_OMP="${CXXFLAGS_BASE} -fopenmp"
# Для SIMD (AVX2+FMA); при желании добавь -ffast-math -funroll-loops
CXXFLAGS_SIMD="${CXXFLAGS_BASE} -mavx2 -mfma"
# OMP+SIMD
CXXFLAGS_OMPSIMD="${CXXFLAGS_BASE} -fopenmp -mavx2 -mfma"

# ---------- PARAMS ----------
OUT=${1:-"$RESULTS/results_hybrids.csv"}
HYBRID_LEAF=${HYBRID_LEAF:-128}   # порог листа
HYBRID_T=${HYBRID_T:-64}          # тайл в листе
HYBRID_CUT=${HYBRID_CUT:-256}     # порог создания задач (OMP/OMP+SIMD)
THREADS=${THREADS:-8}             # OMP_NUM_THREADS

# ---------- BUILD ----------
$CXX $CXXFLAGS_BASE    -I"$INC" "$SRC/hybrid_mm.cpp"          -o "$BUILD/hybrid_mm"
$CXX $CXXFLAGS_OMP     -I"$INC" "$SRC/hybrid_omp_mm.cpp"      -o "$BUILD/hybrid_omp_mm"
$CXX $CXXFLAGS_SIMD    -I"$INC" "$SRC/hybrid_simd_mm.cpp"     -o "$BUILD/hybrid_simd_mm"
$CXX $CXXFLAGS_OMPSIMD -I"$INC" "$SRC/hybrid_ompsimd_mm.cpp"  -o "$BUILD/hybrid_ompsimd_mm"

# ---------- RUN ----------
if [ "${APPEND:-0}" != "1" ]; then rm -f "$OUT"; fi

echo "Writing to: $OUT"
echo "HYBRID_LEAF=$HYBRID_LEAF  HYBRID_T=$HYBRID_T  HYBRID_CUT=$HYBRID_CUT  THREADS=$THREADS"

# seq
"$BUILD/hybrid_mm" "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# OMP env
export OMP_DYNAMIC=false
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS="$THREADS"

# OMP
"$BUILD/hybrid_omp_mm" "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"

# SIMD
"$BUILD/hybrid_simd_mm" "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# OMP+SIMD
"$BUILD/hybrid_ompsimd_mm" "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"

echo "Done. Results saved to $OUT"