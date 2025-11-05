/*
 * Алгоритм: блочное умножение + OpenMP.
 * Идея параллелизма: каждая (ii,jj)-плитка C принадлежит ровно одному потоку,
 * он последовательно накапливает вклад по всем kk-блокам → нет гонок.
 *
 * Сложность: Θ(N^3) по времени, Θ(N^2) по памяти; снижаем константу за счёт кэша и распараллеливания.
 * Параметры: argv[1]=csv путь (опц.), argv[2]=T (tile, по умолчанию 64).
 * Сборка: g++ -O3 -march=native -std=c++17 -fopenmp omp_blocked_mm.cpp -o omp_blocked_mm
 */

#include "config.h"
#include "mm_common.h"
#include <omp.h>
#include <algorithm>
#include <cstdio>

static void mmul_blocked_omp(const std::vector<double>& A,
                             const std::vector<double>& B,
                             std::vector<double>& C,
                             int N, int T)
{
    std::fill(C.begin(), C.end(), 0.0);

    // Параллелим по (ii, jj): каждая плитка C обновляется одним потоком.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < N; ii += T) {
        for (int jj = 0; jj < N; jj += T) {

            for (int kk = 0; kk < N; kk += T) {
                int iimax = std::min(ii + T, N);
                int jjmax = std::min(jj + T, N);
                int kkmax = std::min(kk + T, N);

                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        const double aik = A[i*N + k];
                        // j внутренний → читаем B[k, j] построчно (плотно)
                        for (int j = jj; j < jjmax; ++j) {
                            C[i*N + j] += aik * B[k*N + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");
    int T = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : 64;

    ensure_csv_header(out_path, "algo,N,repeat,time_sec");

    for (int N : SIZES) {
        std::vector<double> A(N*N), B(N*N), C(N*N);
        fill_matrix(A, N, RNG_SEED + 1);
        fill_matrix(B, N, RNG_SEED + 2);

        for (int rep = 1; rep <= REPEAT; ++rep) {
            double t0 = omp_get_wtime();
            mmul_blocked_omp(A, B, C, N, T);
            double t1 = omp_get_wtime();
            double secs = t1 - t0;

            std::ofstream out(out_path, std::ios::app);
            out << "omp_blocked(T=" << T << ",p=" << omp_get_max_threads() << "),"
                << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
