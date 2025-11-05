/*
 * Алгоритм: блочное умножение + SIMD (AVX2) по оси j.
 *   Внутри тайла T×T для фиксированных (i,k) накапливаем векторную сумму по j:
 *     load C[i, j..j+VLEN), sum += aik * B[k, j..j+VLEN)
 *   Используем _mm256_{loadu,set1,fmadd,storeu}_pd (VLEN=4 для double).
 *
 * Сложность:
 *   Время: Θ(N^3) (как и у блочного), но меньше константа за счёт SIMD.
 *   Память: Θ(N^2). Доп. буферы не требуются.
 *
 * Параметры:
 *   argv[1] — путь к CSV (опц., default "results.csv")
 *   argv[2] — размер тайла T (опц., default 64)
 *
 * Сборка: g++ -O3 -march=native -std=c++17 simd_blocked_mm.cpp -o simd_blocked_mm
 *  (на x86 с AVX2/AVX-512 -march=native подтянет нужные ISA; при желании можно добавить -mavx2 -mfma)
 */
#include "config.h"
#include "mm_common.h"
#include <immintrin.h>    // AVX/SSE
#include <algorithm>
#include <cstdio>

#if defined(__AVX2__)
  static inline __m256d fmadd_pd(__m256d a, __m256d b, __m256d c) {
  #if defined(__FMA__)
    return _mm256_fmadd_pd(a,b,c);
  #else
    return _mm256_add_pd(_mm256_mul_pd(a,b), c);
  #endif
  }
  constexpr int VLEN = 4;     // 4 doubles = 256 бит
  constexpr int SIMD_BITS = 256;
#elif defined(__SSE2__)
  // Фолбэк на SSE2 (128 бит, 2 double)
  static inline __m128d fmadd_pd(__m128d a, __m128d b, __m128d c) {
    return _mm_add_pd(_mm_mul_pd(a,b), c);
  }
  constexpr int VLEN = 2;
  constexpr int SIMD_BITS = 128;
#else
  constexpr int VLEN = 1;     // без SIMD
  constexpr int SIMD_BITS = 0;
#endif

static void mmul_blocked_simd(const std::vector<double>& A,
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

                    #if defined(__AVX2__)
                        __m256d a = _mm256_set1_pd(aik);
                        int j = jj;
                        for (; j + 4 <= jjmax; j += 4) {
                            __m256d c = _mm256_loadu_pd(&C[i*N + j]);
                            __m256d b = _mm256_loadu_pd(&B[k*N + j]);
                            c = _mm256_fmadd_pd(a, b, c);  // если нет FMA, компилятор заменит на mul+add
                            _mm256_storeu_pd(&C[i*N + j], c);
                        }
                        for (; j < jjmax; ++j)
                            C[i*N + j] += aik * B[k*N + j];

                    #elif defined(__SSE2__)
                        __m128d a = _mm_set1_pd(aik);
                        int j = jj;
                        for (; j + 2 <= jjmax; j += 2) {
                            __m128d c = _mm_loadu_pd(&C[i*N + j]);
                            __m128d b = _mm_loadu_pd(&B[k*N + j]);
                            c = _mm_add_pd(_mm_mul_pd(a,b), c);
                            _mm_storeu_pd(&C[i*N + j], c);
                        }
                        for (; j < jjmax; ++j)
                            C[i*N + j] += aik * B[k*N + j];

                    #else
                        for (int j = jj; j < jjmax; ++j)
                            C[i*N + j] += aik * B[k*N + j];
                    #endif
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
            mmul_blocked_simd(A, B, C, N, T);
            auto t1 = Clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            std::ofstream out(out_path, std::ios::app);
            out << "simd_blocked(T=" << T << ";W=" << SIMD_BITS << "),"
                << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
