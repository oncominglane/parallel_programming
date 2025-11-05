/*
 * Алгоритм: блочное (tile/blocked) умножение матриц.
 *   Делим матрицы на квадраты T×T и обновляем C-подматрицы по частям.
 *   Порядок циклов: ii, kk, jj (по блокам) и i, k, j внутри.
 *   Для фиксированных i,k внутренний цикл по j читает B[k, j] последовательно,
 *   а A[i, k] переиспользуется из регистра/кэша. Это существенно снижает
 *   промахи кэша по сравнению с наивной i-j-k схемой.
 *
 * Сложность:
 *   Время:   Θ(N^3) (число операций не меняется).
 *   Память:  Θ(N^2) для A, B, C; доп. память не требуется (кроме регистров/кэша).
 *   Константа во времени снижается за счёт лучшей локальности (T подбирается эмпирически).
 *
 * Параметры:
 *   Размер блока T задаётся вторым аргументом командной строки (по умолчанию 64).
 *   CSV-вывод совместим с предыдущими бенчами: algo,N,repeat,time_sec.
 */
#include "config.h"
#include "mm_common.h"

static void mmul_blocked(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C,
                         int N, int T)
{
    std::fill(C.begin(), C.end(), 0.0);
    for (int ii = 0; ii < N; ii += T) {
        int iimax = std::min(ii + T, N);
        for (int kk = 0; kk < N; kk += T) {
            int kkmax = std::min(kk + T, N);
            for (int jj = 0; jj < N; jj += T) {
                int jjmax = std::min(jj + T, N);

                for (int i = ii; i < iimax; ++i) {
                    for (int k = kk; k < kkmax; ++k) {
                        const double aik = A[i*N + k];
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
            auto t0 = Clock::now();
            mmul_blocked(A, B, C, N, T);
            auto t1 = Clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            std::ofstream out(out_path, std::ios::app);
            out << "blocked(T=" << T << ")," << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}