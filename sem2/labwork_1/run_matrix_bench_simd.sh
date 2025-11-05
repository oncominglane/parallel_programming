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

CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:--O3 -march=native -std=c++17}

TILES=${TILES:-"64"}
LEAF=${LEAF:-64}
HYBRID_LEAF=${HYBRID_LEAF:-128}
HYBRID_T=${HYBRID_T:-64}
OUT=${1:-results_simd.csv}

mkdir -p build
$CXX $CXXFLAGS simd_blocked_mm.cpp   -o build/simd_blocked_mm
$CXX $CXXFLAGS simd_strassen_mm.cpp  -o build/simd_strassen_mm
$CXX $CXXFLAGS simd_hybrid_mm.cpp    -o build/simd_hybrid_mm

if [ "${APPEND:-0}" != "1" ]; then rm -f "$OUT"; fi

echo "Writing to: $OUT"
echo "TILES:        $TILES"
echo "LEAF:         $LEAF"
echo "HYBRID_LEAF:  $HYBRID_LEAF"
echo "HYBRID_T:     $HYBRID_T"

# simd_blocked: перебор размеров тайла
for T in $TILES; do
  ./build/simd_blocked_mm "$OUT" "$T"
done

# simd_strassen: фиксированный leaf
./build/simd_strassen_mm "$OUT" "$LEAF"

# simd_hybrid: Strassen + блочный лист с упаковкой + SIMD
./build/simd_hybrid_mm "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

# защитим CSV на случай меток с запятыми (на всякий случай; у нас и так без запятых)
sed -i 's/simd_blocked(T=\([0-9]\+\),/simd_blocked(T=\1;/g' "$OUT"

echo "Done. Results saved to $OUT"
