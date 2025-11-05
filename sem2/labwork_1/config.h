#pragma once
#include <vector>
#include <cstdint>

// List of matrix sizes to benchmark (square matrices N x N).
static const std::vector<int> SIZES = {
    64, 128, 256, 384, 512, 768, 1024, 2048
};

// Number of times to repeat each measurement (the program will report each run).
static const int REPEAT = 1;

// RNG seed for reproducible matrix generation.
static const std::uint32_t RNG_SEED = 42;
