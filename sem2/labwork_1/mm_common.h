#pragma once
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <algorithm>

using Clock = std::chrono::high_resolution_clock;
using dsec  = std::chrono::duration<double>;

inline void ensure_csv_header(const std::string& path, const std::string& header) {
    namespace fs = std::filesystem;
    if (!fs::exists(path)) {
        std::ofstream out(path, std::ios::out);
        out << header << "\n";
    }
}

inline void fill_matrix(std::vector<double>& M, int N, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : M) x = dist(gen);
}

inline bool almost_equal(const std::vector<double>& A, const std::vector<double>& B, double eps=1e-9) {
    if (A.size() != B.size()) return false;
    for (size_t i = 0; i < A.size(); ++i) {
        if (std::abs(A[i] - B[i]) > eps) return false;
    }
    return true;
}
