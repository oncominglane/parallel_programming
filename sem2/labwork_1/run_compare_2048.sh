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

# -------------------- настройки --------------------
CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:--O3 -march=native -std=c++17 -fopenmp}

OUT=${1:-results_2048.csv}
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

# -------------------- backup/override config.h --------------------
if [ ! -f config.h ]; then
  echo "ERROR: config.h not found in current directory." >&2
  exit 1
fi

cp -f config.h config.h.bak
cat > config.h <<EOF
#pragma once
#include <vector>
static const std::vector<int> SIZES = { ${N} };
static const int REPEAT = ${REPEAT};
static const unsigned int RNG_SEED = ${RNG};
EOF

# -------------------- сборка --------------------
mkdir -p build

# последовательные
$CXX $CXXFLAGS blocked_mm.cpp    -o build/blocked_mm
$CXX $CXXFLAGS strassen_mm.cpp   -o build/strassen_mm
$CXX $CXXFLAGS hybrid_mm.cpp     -o build/hybrid_mm

# OMP
$CXX $CXXFLAGS omp_blocked_mm.cpp -o build/omp_blocked_mm
if [ -f strassen_omp.cpp ]; then
  $CXX $CXXFLAGS strassen_omp.cpp -o build/omp_strassen_mm
else
  $CXX $CXXFLAGS omp_strassen_mm.cpp -o build/omp_strassen_mm
fi
$CXX $CXXFLAGS hybrid_omp_mm.cpp  -o build/hybrid_omp_mm

# SIMD (без OpenMP) — пробуем отключить OpenMP флагом, если не поддерживается — просто без него
$CXX $CXXFLAGS -fopenmp=0 simd_blocked_mm.cpp   -o build/simd_blocked_mm 2>/dev/null || \
$CXX -O3 -march=native -std=c++17 simd_blocked_mm.cpp   -o build/simd_blocked_mm
$CXX $CXXFLAGS -fopenmp=0 simd_strassen_mm.cpp  -o build/simd_strassen_mm 2>/dev/null || \
$CXX -O3 -march=native -std=c++17 simd_strassen_mm.cpp  -o build/simd_strassen_mm
$CXX $CXXFLAGS -fopenmp=0 hybrid_simd_mm.cpp    -o build/hybrid_simd_mm 2>/dev/null || \
$CXX -O3 -march=native -std=c++17 hybrid_simd_mm.cpp    -o build/hybrid_simd_mm

# OMP+SIMD (нужны и -fopenmp, и AVX2/FMA для SIMD-листа)
$CXX $CXXFLAGS -mavx2 -mfma hybrid_ompsimd_mm.cpp -o build/hybrid_ompsimd_mm 2>/dev/null || \
$CXX -O3 -march=native -std=c++17 -fopenmp -mavx2 -mfma hybrid_ompsimd_mm.cpp -o build/hybrid_ompsimd_mm

# -------------------- запуски --------------------
rm -f "$OUT"

# seq: blocked, strassen, HYBRID
./build/blocked_mm   "$OUT" "$TILE"
./build/strassen_mm  "$OUT" "$LEAF"
./build/hybrid_mm    "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# OMP env
export OMP_DYNAMIC=false
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS="$THREADS"

# omp: blocked, strassen, HYBRID
./build/omp_blocked_mm   "$OUT" "$TILE"
./build/omp_strassen_mm  "$OUT" "$LEAF" "$CUT"
./build/hybrid_omp_mm    "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"

# simd: blocked, strassen, HYBRID
./build/simd_blocked_mm  "$OUT" "$TILE"
./build/simd_strassen_mm "$OUT" "$LEAF"
./build/hybrid_simd_mm   "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# omp+simd: гибрид c задачами + SIMD-лист
./build/hybrid_ompsimd_mm "$OUT" "$HYBRID_LEAF" "$HYBRID_T" "$HYBRID_CUT"

# sanitize CSV (на случай запятых внутри algo у omp_blocked)
sed -i 's/(T=\([0-9]\+\),p=/\(T=\1;p=/g' "$OUT"

# -------------------- печать сводки по N=2048 --------------------
echo
echo "=== Summary (N=${N}, average over REPEAT=${REPEAT}) ==="
printf "%-55s %12s\n" "algo" "avg_time_s"
echo "------------------------------------------------------- ------------"
awk -F, -v N="$N" '
  NR>1 && $2==N { sum[$1]+=$4; cnt[$1]++ }
  END { for (a in sum) printf "%-55s %12.6f\n", a, sum[a]/cnt[a]; }
' "$OUT" | sort -k2,2n

echo
echo "Raw data saved to: $OUT"

# -------------------- restore config.h --------------------
mv -f config.h.bak config.h