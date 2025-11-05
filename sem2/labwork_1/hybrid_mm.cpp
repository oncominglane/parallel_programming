/*
 * Алгоритм: гибрид Strassen + блочный leaf с локальной упаковкой плит B.
 *
 * Идея:
 *   • Верхние уровни — классический Штрассен (7 подзадач).
 *   • База рекурсии (n <= leaf) — блочное умножение:
 *       - делим на тайлы T×T;
 *       - для каждой пары (kk, jj) упаковываем плиту B[kk:kk+T, jj:jj+T] подряд в буфер Bpack
 *         (локальная «транспозиция/копия»), потом умножаем куски A×Bpack и накапливаем в C.
 *     Это даёт хорошую локальность и для A/C (как у блочного) и для B (как у transpose),
 *     но без глобальной транспозиции B.
 *
 * Сложность:
 *   Время:  Θ(N^log2 7) ≈ Θ(N^2.807) (доминирует верхняя часть Страссена);
 *           leaf-ядро — Θ(n^3), но с более выгодной константой за счёт блока+упаковки.
 *   Память: Θ(N^2) + временные буферы на уровне (квадранты + M1..M7 + суммирования).
 *
 * Параметры запуска:
 *   ./hybrid_mm [results.csv] [leaf] [T]
 *     leaf — порог для перехода на блочный leaf (по умолчанию 128)
 *     T    — размер тайла в leaf (по умолчанию 64)
 *
 * CSV-вывод: "algo,N,repeat,time_sec" с algo = "hybrid(strassen+blocked;leaf=...;T=...)".
 *
 * Сборка:
 *   g++ -O3 -march=native -std=c++17 hybrid_mm.cpp -o hybrid_mm
 */

#include "config.h"
#include "mm_common.h"
#include <algorithm>
#include <vector>
#include <cassert>

// --------- утилиты копирования подблоков (для Strassen) ---------

static void copy_block(const std::vector<double>& src, int N, int r0, int c0,
                       std::vector<double>& dst, int n)
{
    for (int i = 0; i < n; ++i) {
        const double* srow = &src[(r0 + i)*N + c0];
        std::copy(srow, srow + n, &dst[i*n]);
    }
}

static void write_block(const std::vector<double>& src, int n,
                        std::vector<double>& dst, int N, int r0, int c0)
{
    for (int i = 0; i < n; ++i) {
        const double* srow = &src[i*n];
        double* drow = &dst[(r0 + i)*N + c0];
        std::copy(srow, srow + n, drow);
    }
}

// --------- простые поэлементные операции (для Strassen) ---------

static void add(const std::vector<double>& A, const std::vector<double>& B,
                std::vector<double>& C, int n)
{
    for (int i = 0; i < n*n; ++i) C[i] = A[i] + B[i];
}

static void sub(const std::vector<double>& A, const std::vector<double>& B,
                std::vector<double>& C, int n)
{
    for (int i = 0; i < n*n; ++i) C[i] = A[i] - B[i];
}

static void add_inplace(std::vector<double>& A, const std::vector<double>& B, int n)
{
    for (int i = 0; i < n*n; ++i) A[i] += B[i];
}

static void sub_inplace(std::vector<double>& A, const std::vector<double>& B, int n)
{
    for (int i = 0; i < n*n; ++i) A[i] -= B[i];
}

// --------- leaf-ядро: блочное умножение с локальной упаковкой плит B ---------

// Упаковать плиту B[k0:k0+kspan, j0:j0+jspan] (из матрицы n×n) в буфер row-major.
// Размер буфера должен быть >= kspan*jspan.
static void pack_B_tile(const std::vector<double>& B, int n,
                        int k0, int j0, int kspan, int jspan,
                        std::vector<double>& Bpack)
{
    for (int k = 0; k < kspan; ++k) {
        const double* src = &B[(k0 + k)*n + j0];
        double*       dst = &Bpack[k * jspan];
        std::copy(src, src + jspan, dst); // копируем подряд сегмент строки B[k0+k, j0..j0+jspan)
    }
}

// Блочное умножение C += A×B (все n×n), с упаковкой плит B[jj..jj+T).
static void blocked_packed_leaf(const std::vector<double>& A,
                                const std::vector<double>& B,
                                std::vector<double>& C,
                                int n, int T)
{
    std::fill(C.begin(), C.end(), 0.0);

    std::vector<double> Bpack; // переиспользуемый буфер для текущей плиты B (макс. T*T)
    Bpack.reserve((size_t)T * (size_t)T);

    for (int ii = 0; ii < n; ii += T) {
        int iimax = std::min(ii + T, n);
        for (int kk = 0; kk < n; kk += T) {
            int kkmax = std::min(kk + T, n);
            int kspan = kkmax - kk;

            for (int jj = 0; jj < n; jj += T) {
                int jjmax = std::min(jj + T, n);
                int jspan = jjmax - jj;

                // упаковать плиту B[kk:kk+kspan, jj:jj+jspan] подряд
                Bpack.resize((size_t)kspan * (size_t)jspan);
                pack_B_tile(B, n, kk, jj, kspan, jspan, Bpack);

                // умножить A[ii:iimax, kk:kkmax] на упакованную плиту B и накапливать в C[ii:iimax, jj:jjmax]
                for (int i = ii; i < iimax; ++i) {
                    for (int k = 0; k < kspan; ++k) {
                        const double aik = A[i*n + (kk + k)];
                        const double* brow = &Bpack[(size_t)k * jspan]; // плотный ряд длиной jspan
                        double*       crow = &C[i*n + jj];
                        for (int j = 0; j < jspan; ++j) {
                            crow[j] += aik * brow[j];
                        }
                    }
                }
            }
        }
    }
}

// --------- рекурсивный гибрид Strassen + блочный leaf ---------

static void strassen_hybrid_rec(const std::vector<double>& A,
                                const std::vector<double>& B,
                                std::vector<double>& C,
                                int n, int leaf, int T)
{
    if (n <= leaf) { blocked_packed_leaf(A, B, C, n, T); return; }

    const int m = n / 2;

    // разбиение на квадранты (в плотные буферы m×m)
    std::vector<double> A11(m*m), A12(m*m), A21(m*m), A22(m*m);
    std::vector<double> B11(m*m), B12(m*m), B21(m*m), B22(m*m);
    copy_block(A, n, 0, 0, A11, m);
    copy_block(A, n, 0, m, A12, m);
    copy_block(A, n, m, 0, A21, m);
    copy_block(A, n, m, m, A22, m);
    copy_block(B, n, 0, 0, B11, m);
    copy_block(B, n, 0, m, B12, m);
    copy_block(B, n, m, 0, B21, m);
    copy_block(B, n, m, m, B22, m);

    // временные буферы
    std::vector<double> T1(m*m), T2(m*m);
    std::vector<double> M1(m*m), M2(m*m), M3(m*m), M4(m*m), M5(m*m), M6(m*m), M7(m*m);

    // 7 умножений Штрассена, в рекурсивных вызовах — тот же гибрид
    // M1 = (A11 + A22) * (B11 + B22)
    add(A11, A22, T1, m);
    add(B11, B22, T2, m);
    strassen_hybrid_rec(T1, T2, M1, m, leaf, T);

    // M2 = (A21 + A22) * B11
    add(A21, A22, T1, m);
    strassen_hybrid_rec(T1, B11, M2, m, leaf, T);

    // M3 = A11 * (B12 - B22)
    sub(B12, B22, T2, m);
    strassen_hybrid_rec(A11, T2, M3, m, leaf, T);

    // M4 = A22 * (B21 - B11)
    sub(B21, B11, T2, m);
    strassen_hybrid_rec(A22, T2, M4, m, leaf, T);

    // M5 = (A11 + A12) * B22
    add(A11, A12, T1, m);
    strassen_hybrid_rec(T1, B22, M5, m, leaf, T);

    // M6 = (A21 - A11) * (B11 + B12)
    sub(A21, A11, T1, m);
    add(B11, B12, T2, m);
    strassen_hybrid_rec(T1, T2, M6, m, leaf, T);

    // M7 = (A12 - A22) * (B21 + B22)
    sub(A12, A22, T1, m);
    add(B21, B22, T2, m);
    strassen_hybrid_rec(T1, T2, M7, m, leaf, T);

    // сборка результата
    std::vector<double> C11 = M1;  add_inplace(C11, M4, m);  sub_inplace(C11, M5, m);  add_inplace(C11, M7, m);
    std::vector<double> C12 = M3;  add_inplace(C12, M5, m);
    std::vector<double> C21 = M2;  add_inplace(C21, M4, m);
    std::vector<double> C22 = M1;  sub_inplace(C22, M2, m);  add_inplace(C22, M3, m);  add_inplace(C22, M6, m);

    write_block(C11, m, C, n, 0, 0);
    write_block(C12, m, C, n, 0, m);
    write_block(C21, m, C, n, m, 0);
    write_block(C22, m, C, n, m, m);
}

static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");
    int leaf = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : 128; // по умолчанию leaf чуть больше
    int T    = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 64;

    ensure_csv_header(out_path, "algo,N,repeat,time_sec");

    for (int N : SIZES) {
        std::vector<double> A(N*N), B(N*N);
        fill_matrix(A, N, RNG_SEED + 1);
        fill_matrix(B, N, RNG_SEED + 2);

        // паддинг до степени 2
        int P = next_pow2(N);
        std::vector<double> Ap(P*P, 0.0), Bp(P*P, 0.0), Cp(P*P, 0.0);
        for (int i = 0; i < N; ++i) {
            std::copy(A.begin() + i*N, A.begin() + (i+1)*N, Ap.begin() + i*P);
            std::copy(B.begin() + i*N, B.begin() + (i+1)*N, Bp.begin() + i*P);
        }

        for (int rep = 1; rep <= REPEAT; ++rep) {
            std::fill(Cp.begin(), Cp.end(), 0.0);
            auto t0 = Clock::now();
            strassen_hybrid_rec(Ap, Bp, Cp, P, leaf, T);
            auto t1 = Clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            std::ofstream out(out_path, std::ios::app);
            out << "hybrid(strassen+blocked;leaf=" << leaf << ";T=" << T << "),"
                << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
