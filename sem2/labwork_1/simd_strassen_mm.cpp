/*
 * Алгоритм: Strassen с SIMD-листом.
 *   Рекурсивные уровни — как в классическом Штрассене (7 умножений).
 *   База рекурсии (n <= leaf): векторизованное умножение по оси j (без блоков).
 *
 * Сложность:
 *   Асимптотически Θ(N^2.807), лист — Θ(n^3) с SIMD-ускорением.
 *   Память: Θ(N^2) + временные буферы на уровне (копируем квадранты).
 *
 * Параметры:
 *   argv[1] — CSV
 *   argv[2] — leaf (порог для перехода на SIMD-лист, default 64)
 *
 * Сборка: g++ -O3 -march=native -std=c++17 strassen_simd_mm.cpp -o strassen_simd_mm
 */
#include "config.h"
#include "mm_common.h"
#include <immintrin.h>
#include <algorithm>

// --- SIMD-лист: векторизация по j ---
#if defined(__AVX2__)
  #define HAVE_AVX2 1
  static inline __m256d fmadd_pd(__m256d a, __m256d b, __m256d c) {
  #if defined(__FMA__)
    return _mm256_fmadd_pd(a,b,c);
  #else
    return _mm256_add_pd(_mm256_mul_pd(a,b), c);
  #endif
  }
  constexpr int VLEN = 4;
#elif defined(__SSE2__)
  static inline __m128d fmadd_pd(__m128d a, __m128d b, __m128d c) {
    return _mm_add_pd(_mm_mul_pd(a,b), c);
  }
  constexpr int VLEN = 2;
#else
  constexpr int VLEN = 1;
#endif

static void mmul_simd_leaf(const std::vector<double>& A,
                           const std::vector<double>& B,
                           std::vector<double>& C, int n)
{
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j + VLEN <= n; j += VLEN) {
        #if defined(__AVX2__)
            __m256d sumv = _mm256_setzero_pd();
        #elif defined(__SSE2__)
            __m128d sumv = _mm_setzero_pd();
        #else
            double sumv = 0.0; // не используется для VLEN>1
        #endif
            for (int k = 0; k < n; ++k) {
                double aik = A[i*n + k];
            #if defined(__AVX2__)
                __m256d a = _mm256_set1_pd(aik);
                __m256d b = _mm256_loadu_pd(&B[k*n + j]);
                sumv = fmadd_pd(a, b, sumv);
            #elif defined(__SSE2__)
                __m128d a = _mm_set1_pd(aik);
                __m128d b = _mm_loadu_pd(&B[k*n + j]);
                sumv = fmadd_pd(a, b, sumv);
            #else
                (void)aik;
            #endif
            }
        #if defined(__AVX2__)
            _mm256_storeu_pd(&C[i*n + j], sumv);
        #elif defined(__SSE2__)
            _mm_storeu_pd(&C[i*n + j], sumv);
        #endif
        }
        // хвост
        for (int j = (n/VLEN)*VLEN; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) sum += A[i*n + k] * B[k*n + j];
            C[i*n + j] = sum;
        }
    }
}

// простые операции для Страссена
static void add(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n) {
    for (int i = 0; i < n*n; ++i) C[i] = A[i] + B[i];
}
static void sub(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n) {
    for (int i = 0; i < n*n; ++i) C[i] = A[i] - B[i];
}
static void add_inplace(std::vector<double>& A, const std::vector<double>& B, int n) {
    for (int i = 0; i < n*n; ++i) A[i] += B[i];
}
static void sub_inplace(std::vector<double>& A, const std::vector<double>& B, int n) {
    for (int i = 0; i < n*n; ++i) A[i] -= B[i];
}

static void copy_block(const std::vector<double>& src, int N, int r0, int c0,
                       std::vector<double>& dst, int n) {
    for (int i = 0; i < n; ++i) {
        std::copy(&src[(r0+i)*N + c0], &src[(r0+i)*N + c0] + n, &dst[i*n]);
    }
}
static void write_block(const std::vector<double>& src, int n,
                        std::vector<double>& dst, int N, int r0, int c0) {
    for (int i = 0; i < n; ++i) {
        std::copy(&src[i*n], &src[i*n] + n, &dst[(r0+i)*N + c0]);
    }
}

static void strassen_rec_simd(const std::vector<double>& A,
                              const std::vector<double>& B,
                              std::vector<double>& C, int n, int leaf)
{
    if (n <= leaf) { mmul_simd_leaf(A, B, C, n); return; }

    int m = n/2;
    std::vector<double> A11(m*m),A12(m*m),A21(m*m),A22(m*m);
    std::vector<double> B11(m*m),B12(m*m),B21(m*m),B22(m*m);
    copy_block(A,n,0,0,A11,m); copy_block(A,n,0,m,A12,m);
    copy_block(A,n,m,0,A21,m); copy_block(A,n,m,m,A22,m);
    copy_block(B,n,0,0,B11,m); copy_block(B,n,0,m,B12,m);
    copy_block(B,n,m,0,B21,m); copy_block(B,n,m,m,B22,m);

    std::vector<double> M1(m*m),M2(m*m),M3(m*m),M4(m*m),M5(m*m),M6(m*m),M7(m*m);
    std::vector<double> S1(m*m),S2(m*m),S3(m*m),S4(m*m),S5(m*m),S6(m*m),S7(m*m);
    std::vector<double> T1(m*m),T2(m*m),T3(m*m),T4(m*m),T5(m*m),T6(m*m),T7(m*m);

    add(A11,A22,S1,m); add(B11,B22,T1,m); strassen_rec_simd(S1,T1,M1,m,leaf);
    add(A21,A22,S2,m);                    strassen_rec_simd(S2,B11,M2,m,leaf);
    sub(B12,B22,T3,m);                    strassen_rec_simd(A11,T3,M3,m,leaf);
    sub(B21,B11,T4,m);                    strassen_rec_simd(A22,T4,M4,m,leaf);
    add(A11,A12,S5,m);                    strassen_rec_simd(S5,B22,M5,m,leaf);
    sub(A21,A11,S6,m); add(B11,B12,T6,m); strassen_rec_simd(S6,T6,M6,m,leaf);
    sub(A12,A22,S7,m); add(B21,B22,T7,m); strassen_rec_simd(S7,T7,M7,m,leaf);

    std::vector<double> C11 = M1; add_inplace(C11,M4,m); sub_inplace(C11,M5,m); add_inplace(C11,M7,m);
    std::vector<double> C12 = M3; add_inplace(C12,M5,m);
    std::vector<double> C21 = M2; add_inplace(C21,M4,m);
    std::vector<double> C22 = M1; sub_inplace(C22,M2,m); add_inplace(C22,M3,m); add_inplace(C22,M6,m);

    write_block(C11,m,C,n,0,0);
    write_block(C12,m,C,n,0,m);
    write_block(C21,m,C,n,m,0);
    write_block(C22,m,C,n,m,m);
}

static int next_pow2(int n){ int p=1; while(p<n) p<<=1; return p; }

int main(int argc, char** argv){
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");
    int leaf = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : 64;

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

        for (int rep=1; rep<=REPEAT; ++rep) {
            std::fill(Cp.begin(), Cp.end(), 0.0);
            auto t0 = Clock::now();
            strassen_rec_simd(Ap, Bp, Cp, P, leaf);
            auto t1 = Clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            std::ofstream out(out_path, std::ios::app);
            out << "simd_strassen(leaf=" << leaf << "),"
                << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
