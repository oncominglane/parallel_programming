// transpose_mm.cpp

/*
 * Алгоритм: умножение с предварительным транспонированием B.
 *   1) Строим Bt = B^T (O(N^2)).
 *   2) Затем C[i,j] = dot( A[i,:], Bt[j,:] ) — обе строки читаются последовательно,
 *      что улучшает локальность и векторизацию (по сравнению с наивным вариантом).
 *   Хранение: row-major в std::vector<double> длиной N*N; отдельный буфер Bt.
 *   Примечание: в измерение времени включена стоимость транспонирования.
 *
 * Сложность:
 *   Время:   Θ(N^3) (умножение) + Θ(N^2) (transpose) → асимптотически Θ(N^3).
 *   Память:  Θ(N^2) доп. под Bt (итого A, B, C, Bt — константно больше, но всё ~N^2).
 */


#include "config.h" 
#include "mm_common.h"

// Transpose B so that we read contiguous data when computing dot products.
// We INCLUDE the transpose time in the measurement by default (conservative).
static void transpose(const std::vector<double>& B, std::vector<double>& Bt, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            Bt[j*N + i] = B[i*N + j]; //транспонируем B
}

static void mmul_row_row(const std::vector<double>& A,
                         const std::vector<double>& Bt,
                         std::vector<double>& C,
                         int N)
{
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < N; ++i) {
        const double* arow = &A[i*N]; // указатель на начало iй строки матрицы А
        for (int j = 0; j < N; ++j) {
            const double* brow = &Bt[j*N];  // указатель на начало jй строки матрицы Bt (т е jго столбца B)
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += arow[k] * brow[k];
            }
            C[i*N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");

    ensure_csv_header(out_path, "algo,N,repeat,time_sec");

    for (int N : SIZES) {
        std::vector<double> A(N*N), B(N*N), C(N*N), Bt(N*N);
        fill_matrix(A, N, RNG_SEED + 1);
        fill_matrix(B, N, RNG_SEED + 2);

        for (int rep = 1; rep <= REPEAT; ++rep) {
            auto t0 = Clock::now();
            transpose(B, Bt, N);           // учитываем время на транспонирование
            mmul_row_row(A, Bt, C, N);
            auto t1 = Clock::now();
            double secs = std::chrono::duration_cast<dsec>(t1 - t0).count();

            std::ofstream out(out_path, std::ios::app);
            out << "transpose" << "," << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
