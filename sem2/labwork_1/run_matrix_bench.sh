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

# Output CSV
OUT=${1:-results.csv}
rm -f "$OUT"

# Run binaries; they append to the same CSV and add header if needed
./build/naive_mm "$OUT"
./build/transpose_mm "$OUT"

TILES=${TILES:-"64"}
for T in $TILES; do
  ./build/blocked_mm "$OUT" "$T"
done

LEAF=${LEAF:-64}
./build/strassen_mm "$OUT" "$LEAF"

echo "Wrote results to $OUT"


