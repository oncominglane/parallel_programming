/*
 * Алгоритм: наивное умножение + OpenMP (параллелим по строкам i).
 * Доступ к B остаётся по столбцам (stride=N), поэтому ускорение хуже, чем у блочного.
 *
 * Сложность: Θ(N^3) по времени, Θ(N^2) по памяти; ускорение пропорционально числу ядер,
 * но ограничено пропускной способностью памяти и неудачной локальностью по B.
 * Сборка: g++ -O3 -march=native -std=c++17 -fopenmp omp_naive_mm.cpp -o omp_naive_mm
 */
#include "config.h"
#include "mm_common.h"
#include <omp.h>

static void mmul_naive_omp(const std::vector<double>& A,
                           const std::vector<double>& B,
                           std::vector<double>& C,
                           int N)
{
    std::fill(C.begin(), C.end(), 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i*N + k] * B[k*N + j]; // B по столбцам — кэш-невыгодно
            }
            C[i*N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");

    ensure_csv_header(out_path, "algo,N,repeat,time_sec");

    for (int N : SIZES) {
        std::vector<double> A(N*N), B(N*N), C(N*N);
        fill_matrix(A, N, RNG_SEED + 1);
        fill_matrix(B, N, RNG_SEED + 2);

        for (int rep = 1; rep <= REPEAT; ++rep) {
            double t0 = omp_get_wtime();
            mmul_naive_omp(A, B, C, N);
            double t1 = omp_get_wtime();
            double secs = t1 - t0;

            std::ofstream out(out_path, std::ios::app);
            out << "omp_naive(p=" << omp_get_max_threads() << "),"
                << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}

