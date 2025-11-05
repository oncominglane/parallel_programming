#!/usr/bin/env bash
set -euo pipefail

CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:--O3 -march=native -std=c++17}

# Build directory
mkdir -p build

# Compile
$CXX $CXXFLAGS naive_mm.cpp   -o build/naive_mm
$CXX $CXXFLAGS transpose_mm.cpp -o build/transpose_mm

# Output CSV
OUT=${1:-results.csv}
rm -f "$OUT"

# Run both binaries; they append to the same CSV and add header if needed
./build/naive_mm "$OUT"
./build/transpose_mm "$OUT"

echo "Wrote results to $OUT"


