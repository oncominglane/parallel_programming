#!/usr/bin/env bash
set -euo pipefail

CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:--O3 -march=native -std=c++17}

# Build directory
mkdir -p build

# Compile
$CXX $CXXFLAGS naive_mm.cpp       -o build/naive_mm
$CXX $CXXFLAGS transpose_mm.cpp   -o build/transpose_mm
$CXX $CXXFLAGS blocked_mm.cpp     -o build/blocked_mm
$CXX $CXXFLAGS strassen_mm.cpp    -o build/strassen_mm
$CXX $CXXFLAGS hybrid_mm.cpp      -o build/hybrid_mm

# Output CSV
OUT=${1:-results.csv}
rm -f "$OUT"

# Params
TILES=${TILES:-"64"}      # список T для blocked
LEAF=${LEAF:-64}          # leaf для strassen
HYBRID_LEAF=${HYBRID_LEAF:-128}  # leaf для гибрида
HYBRID_T=${HYBRID_T:-64}         # T для гибрида

# Run binaries; they append to the same CSV and add header if needed
./build/naive_mm "$OUT"
./build/transpose_mm "$OUT"

for T in $TILES; do
  ./build/blocked_mm "$OUT" "$T"
done

./build/strassen_mm "$OUT" "$LEAF"

# Hybrid (Strassen + blocked leaf with local B packing)
./build/hybrid_mm "$OUT" "$HYBRID_LEAF" "$HYBRID_T"

echo "Wrote results to $OUT"
echo "Params: TILES={$TILES}  STRASSEN_LEAF=$LEAF  HYBRID_LEAF=$HYBRID_LEAF  HYBRID_T=$HYBRID_T"
