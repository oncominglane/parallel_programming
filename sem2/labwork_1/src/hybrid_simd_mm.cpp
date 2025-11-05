/*
 * SIMD-гибрид: Strassen (верхние уровни) + блочный leaf с локальной упаковкой плит B и AVX2/SSE2.
 *
 * Идея:
 *   • Верх: классический Штрассен (7 подзадач, последовательно).
 *   • Лист (n <= leaf): блочное умножение:
 *       - делим на тайлы T×T;
 *       - для каждой пары блоков (kk, jj) упаковываем плиту B[kk:kk+T, jj:jj+T] подряд (row-major) в Bpack;
 *       - умножаем A×Bpack и накапливаем в C, векторизуя внутренний цикл по j (AVX2/SSE2).
 *
 * Сложность:
 *   Время:  Θ(N^log2 7) ≈ Θ(N^2.807) (доминирует верхняя часть Страссена);
 *           лист — Θ(n^3), но с меньшей константой за счёт блоков + упаковки + SIMD.
 *   Память: Θ(N^2) + временные буферы на уровне (квадранты, M1..M7, суммы/разности) + Bpack (<= T*T).
 *
 * Параметры запуска:
 *   ./hybrid_simd_mm [results.csv] [leaf=128] [T=64]
 *
 * CSV: "algo,N,repeat,time_sec" с algo = "simd_hybrid(strassen+blocked;leaf=..;T=..;W=..)".
 *
 * Сборка (x86 AVX2):
 *   g++ -O3 -march=native -std=c++17 hybrid_simd_mm.cpp -o hybrid_simd_mm
 *   # при необходимости явно: -mavx2 -mfma
 */

#include "config.h"
#include "mm_common.h"

#include <algorithm>
#include <vector>
#include <cassert>
#include <immintrin.h>

// ---------------- SIMD helpers ----------------
#if defined(__AVX2__)
  static inline __m256d fmadd_pd(__m256d a, __m256d b, __m256d c) {
  #if defined(__FMA__)
    return _mm256_fmadd_pd(a,b,c);
  #else
    return _mm256_add_pd(_mm256_mul_pd(a,b), c);
  #endif
  }
  constexpr int VLEN = 4;      // 4 doubles = 256 бит
  constexpr int SIMD_BITS = 256;
#elif defined(__SSE2__)
  static inline __m128d fmadd_pd(__m128d a, __m128d b, __m128d c) {
    return _mm_add_pd(_mm_mul_pd(a,b), c);
  }
  constexpr int VLEN = 2;      // 2 doubles = 128 бит
  constexpr int SIMD_BITS = 128;
#else
  constexpr int VLEN = 1;      // без SIMD
  constexpr int SIMD_BITS = 0;
#endif

// --------- копирование подблоков (для Strassen) ---------
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

// --------- поэлементные операции (для Strassen) ---------
static void add(const std::vector<double>& A, const std::vector<double>& B,
                std::vector<double>& C, int n)
{ for (int i = 0; i < n*n; ++i) C[i] = A[i] + B[i]; }
static void sub(const std::vector<double>& A, const std::vector<double>& B,
                std::vector<double>& C, int n)
{ for (int i = 0; i < n*n; ++i) C[i] = A[i] - B[i]; }
static void add_inplace(std::vector<double>& A, const std::vector<double>& B, int n)
{ for (int i = 0; i < n*n; ++i) A[i] += B[i]; }
static void sub_inplace(std::vector<double>& A, const std::vector<double>& B, int n)
{ for (int i = 0; i < n*n; ++i) A[i] -= B[i]; }

// --------- упаковка плит B (локальная «транспозиция/копия») ---------
static void pack_B_tile(const std::vector<double>& B, int n,
                        int k0, int j0, int kspan, int jspan,
                        std::vector<double>& Bpack)
{
    for (int k = 0; k < kspan; ++k) {
        const double* src = &B[(k0 + k)*n + j0];
        double*       dst = &Bpack[(size_t)k * jspan]; // ряд длиной jspan
        std::copy(src, src + jspan, dst);
    }
}

// --------- лист: блочное умножение с упаковкой B + SIMD по j ---------
static void blocked_packed_leaf_simd(const std::vector<double>& A,
                                     const std::vector<double>& B,
                                     std::vector<double>& C,
                                     int n, int T)
{
    std::fill(C.begin(), C.end(), 0.0);

    // выровненный буфер не обязателен, но полезен; можно оставить std::vector
    std::vector<double> Bpack;
    Bpack.reserve((size_t)T * (size_t)T);

    for (int ii = 0; ii < n; ii += T) {
        int iimax = std::min(ii + T, n);
        for (int kk = 0; kk < n; kk += T) {
            int kkmax = std::min(kk + T, n);
            int kspan = kkmax - kk;

            for (int jj = 0; jj < n; jj += T) {
                int jjmax = std::min(jj + T, n);
                int jspan = jjmax - jj;

                // плотно упакуем плиту B (kspan × jspan) в row-major
                Bpack.resize((size_t)kspan * (size_t)jspan);
                pack_B_tile(B, n, kk, jj, kspan, jspan, Bpack);

                // обрабатываем строки по 2 штуки (микроядро 2×VLEN)
                int i = ii;
                for (; i + 1 < iimax; i += 2) {

                    int j = 0;
                #if defined(__AVX2__)
                    for (; j + 8 <= jspan; j += 8) {
                        // 2×8 аккумулятора в регистрах для двух строк C
                        __m256d acc00 = _mm256_setzero_pd(), acc01 = _mm256_setzero_pd();
                        __m256d acc10 = _mm256_setzero_pd(), acc11 = _mm256_setzero_pd();

                        for (int k = 0; k < kspan; ++k) {
                            const double a0 = A[i    *n + (kk + k)];
                            const double a1 = A[(i+1)*n + (kk + k)];
                            __m256d A0 = _mm256_set1_pd(a0);
                            __m256d A1 = _mm256_set1_pd(a1);

                            const double* brow = &Bpack[(size_t)k * jspan + j];
                            __m256d B0 = _mm256_loadu_pd(brow + 0);
                            __m256d B1 = _mm256_loadu_pd(brow + 4);

                            acc00 = _mm256_fmadd_pd(A0, B0, acc00);
                            acc01 = _mm256_fmadd_pd(A0, B1, acc01);
                            acc10 = _mm256_fmadd_pd(A1, B0, acc10);
                            acc11 = _mm256_fmadd_pd(A1, B1, acc11);
                        }

                        double* c0 = &C[i    *n + (jj + j)];
                        double* c1 = &C[(i+1)*n + (jj + j)];
                        __m256d c00 = _mm256_loadu_pd(c0 + 0);
                        __m256d c01 = _mm256_loadu_pd(c0 + 4);
                        __m256d c10 = _mm256_loadu_pd(c1 + 0);
                        __m256d c11 = _mm256_loadu_pd(c1 + 4);

                        _mm256_storeu_pd(c0 + 0, _mm256_add_pd(c00, acc00));
                        _mm256_storeu_pd(c0 + 4, _mm256_add_pd(c01, acc01));
                        _mm256_storeu_pd(c1 + 0, _mm256_add_pd(c10, acc10));
                        _mm256_storeu_pd(c1 + 4, _mm256_add_pd(c11, acc11));
                    }
                #elif defined(__SSE2__)
                    for (; j + 4 <= jspan; j += 4) {
                        __m128d acc00 = _mm_setzero_pd(), acc01 = _mm_setzero_pd();
                        __m128d acc10 = _mm_setzero_pd(), acc11 = _mm_setzero_pd();

                        for (int k = 0; k < kspan; ++k) {
                            const double a0 = A[i    *n + (kk + k)];
                            const double a1 = A[(i+1)*n + (kk + k)];
                            __m128d A0 = _mm_set1_pd(a0);
                            __m128d A1 = _mm_set1_pd(a1);

                            const double* brow = &Bpack[(size_t)k * jspan + j];
                            __m128d B0 = _mm_loadu_pd(brow + 0);
                            __m128d B1 = _mm_loadu_pd(brow + 2);

                            acc00 = _mm_add_pd(_mm_mul_pd(A0, B0), acc00);
                            acc01 = _mm_add_pd(_mm_mul_pd(A0, B1), acc01);
                            acc10 = _mm_add_pd(_mm_mul_pd(A1, B0), acc10);
                            acc11 = _mm_add_pd(_mm_mul_pd(A1, B1), acc11);
                        }

                        double* c0 = &C[i    *n + (jj + j)];
                        double* c1 = &C[(i+1)*n + (jj + j)];
                        __m128d c00 = _mm_loadu_pd(c0 + 0);
                        __m128d c01 = _mm_loadu_pd(c0 + 2);
                        __m128d c10 = _mm_loadu_pd(c1 + 0);
                        __m128d c11 = _mm_loadu_pd(c1 + 2);

                        _mm_storeu_pd(c0 + 0, _mm_add_pd(c00, acc00));
                        _mm_storeu_pd(c0 + 2, _mm_add_pd(c01, acc01));
                        _mm_storeu_pd(c1 + 0, _mm_add_pd(c10, acc10));
                        _mm_storeu_pd(c1 + 2, _mm_add_pd(c11, acc11));
                    }
                #endif
                    // хвост по j
                    for (; j < jspan; ++j) {
                        double s0 = 0.0, s1 = 0.0;
                        for (int k = 0; k < kspan; ++k) {
                            const double b = Bpack[(size_t)k * jspan + j];
                            s0 += A[i    *n + (kk + k)] * b;
                            s1 += A[(i+1)*n + (kk + k)] * b;
                        }
                        C[i    *n + (jj + j)] += s0;
                        C[(i+1)*n + (jj + j)] += s1;
                    }
                }

                // одиночная «висячая» строка
                for (; i < iimax; ++i) {
                    for (int j = 0; j < jspan; ++j) {
                        double sum = 0.0;
                        for (int k = 0; k < kspan; ++k)
                            sum += A[i*n + (kk + k)] * Bpack[(size_t)k * jspan + j];
                        C[i*n + (jj + j)] += sum;
                    }
                }
            }
        }
    }
}


// --------- рекурсивная часть: Strassen + SIMD лист ---------
static void strassen_hybrid_rec_simd(const std::vector<double>& A,
                                     const std::vector<double>& B,
                                     std::vector<double>& C,
                                     int n, int leaf, int T)
{
    if (n <= leaf) { blocked_packed_leaf_simd(A, B, C, n, T); return; }

    const int m = n / 2;

    // Квадранты (плотные m×m буферы)
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

    // Временные/результирующие буферы
    std::vector<double> T1(m*m), T2(m*m);
    std::vector<double> M1(m*m), M2(m*m), M3(m*m), M4(m*m), M5(m*m), M6(m*m), M7(m*m);

    // 7 умножений Штрассена (последовательно)
    add(A11, A22, T1, m);  add(B11, B22, T2, m);
    strassen_hybrid_rec_simd(T1, T2, M1, m, leaf, T);

    add(A21, A22, T1, m);
    strassen_hybrid_rec_simd(T1, B11, M2, m, leaf, T);

    sub(B12, B22, T2, m);
    strassen_hybrid_rec_simd(A11, T2, M3, m, leaf, T);

    sub(B21, B11, T2, m);
    strassen_hybrid_rec_simd(A22, T2, M4, m, leaf, T);

    add(A11, A12, T1, m);
    strassen_hybrid_rec_simd(T1, B22, M5, m, leaf, T);

    sub(A21, A11, T1, m);  add(B11, B12, T2, m);
    strassen_hybrid_rec_simd(T1, T2, M6, m, leaf, T);

    sub(A12, A22, T1, m);  add(B21, B22, T2, m);
    strassen_hybrid_rec_simd(T1, T2, M7, m, leaf, T);

    // Сборка C
    std::vector<double> C11 = M1;  add_inplace(C11, M4, m);  sub_inplace(C11, M5, m);  add_inplace(C11, M7, m);
    std::vector<double> C12 = M3;  add_inplace(C12, M5, m);
    std::vector<double> C21 = M2;  add_inplace(C21, M4, m);
    std::vector<double> C22 = M1;  sub_inplace(C22, M2, m);  add_inplace(C22, M3, m);  add_inplace(C22, M6, m);

    write_block(C11, m, C, n, 0, 0);
    write_block(C12, m, C, n, 0, m);
    write_block(C21, m, C, n, m, 0);
    write_block(C22, m, C, n, m, m);
}

// ближайшая степень двойки ≥ n
static int next_pow2(int n) { int p=1; while (p<n) p<<=1; return p; }

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");
    int leaf = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : 128;
    int T    = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 64;

    ensure_csv_header(out_path, "algo,N,repeat,time_sec");

    for (int N : SIZES) {
        std::vector<double> A(N*N), B(N*N);
        fill_matrix(A, N, RNG_SEED + 1);
        fill_matrix(B, N, RNG_SEED + 2);

        int P = next_pow2(N);
        std::vector<double> Ap(P*P,0.0), Bp(P*P,0.0), Cp(P*P,0.0);
        for (int i = 0; i < N; ++i) {
            std::copy(A.begin()+i*N, A.begin()+(i+1)*N, Ap.begin()+i*P);
            std::copy(B.begin()+i*N, B.begin()+(i+1)*N, Bp.begin()+i*P);
        }

        for (int rep = 1; rep <= REPEAT; ++rep) {
            std::fill(Cp.begin(), Cp.end(), 0.0);
            auto t0 = Clock::now();
            strassen_hybrid_rec_simd(Ap, Bp, Cp, P, leaf, T);
            auto t1 = Clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            std::ofstream out(out_path, std::ios::app);
            out << "simd_hybrid(strassen+blocked;leaf=" << leaf
                << ";T=" << T << ";W=" << SIMD_BITS << "),"
                << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
