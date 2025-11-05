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


CXX=${CXX:-g++}

# База для всех
CXXFLAGS_BASE=${CXXFLAGS_BASE:--O3 -march=native -std=c++17}

# Для OMP
CXXFLAGS_OMP="${CXXFLAGS_BASE} -fopenmp"

# Для SIMD (AVX2+FMA); при желании добавь -ffast-math -funroll-loops
CXXFLAGS_SIMD="${CXXFLAGS_BASE} -mavx2 -mfma"
CXXFLAGS_OMPSIMD="${CXXFLAGS_SIMD} -fopenmp"

OUT=${1:-results_hybrids.csv}
HYBRID_LEAF=${HYBRID_LEAF:-128}
HYBRID_T=${HYBRID_T:-64}
HYBRID_CUT=${HYBRID_CUT:-256}
THREADS=${THREADS:-8}

mkdir -p build

# seq
$CXX $CXXFLAGS_BASE   hybrid_mm.cpp         -o build/hybrid_mm

# OMP
$CXX $CXXFLAGS_OMP    omp_hybrid_mm.cpp     -o build/omp_hybrid_mm

# SIMD (AVX2/FMA принудительно)
$CXX $CXXFLAGS_SIMD   simd_hybrid_mm.cpp    -o build/simd_hybrid_mm

# OMP+SIMD
$CXX $CXXFLAGS_OMPSIMD hybrid_ompsimd_mm.cpp -o build/hybrid_ompsimd_mm

if [ "${APPEND:-0}" != "1" ]; then rm -f "$OUT"; fi

echo "Writing to: $OUT"
echo "HYBRID_LEAF=$HYBRID_LEAF  HYBRID_T=$HYBRID_T  HYBRID_CUT=$HYBRID_CUT  THREADS=$THREADS"

# seq
./build/hybrid_mm "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# OMP env
export OMP_DYNAMIC=false
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS="$THREADS"

# OMP
./build/omp_hybrid_mm "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"

# SIMD
./build/simd_hybrid_mm "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# OMP+SIMD
./build/hybrid_ompsimd_mm "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"

echo "Done. Results saved to $OUT"
