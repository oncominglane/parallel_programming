/*
 * OMP-гибрид: Strassen (верхние уровни) + блочный leaf с локальной упаковкой плит B.
 * Параллелизм: на каждом уровне рекурсии создаём задачи для M1..M7 (OpenMP tasks).
 * Лист (n <= leaf) — последовательный блочный kernel с упаковкой B-плит (без глобального transpose).
 *
 * Параметры запуска:
 *   ./hybrid_omp_mm [results.csv] [leaf=128] [T=64] [cut=256]
 *     leaf — порог перехода на leaf-ядро (блочное)
 *     T    — размер тайла в leaf
 *     cut  — минимальный размер подзадачи, при котором ещё создаём omp task
 *
 * Сборка:
 *   g++ -O3 -march=native -std=c++17 -fopenmp hybrid_omp_mm.cpp -o hybrid_omp_mm
 *
 * CSV: "algo,N,repeat,time_sec" с algo = "omp_hybrid(strassen+blocked;leaf=...;T=...;cut=...;p=...)".
 */

#include "config.h"
#include "mm_common.h"
#include <omp.h>
#include <algorithm>
#include <vector>
#include <cassert>

// ---- утилиты копирования подблоков (для Strassen) ----
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

// ---- поэлементные операции (для Strassen) ----
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

// ---- leaf-ядро: блочное умножение с локальной упаковкой плит B (последовательное) ----
static void pack_B_tile(const std::vector<double>& B, int n,
                        int k0, int j0, int kspan, int jspan,
                        std::vector<double>& Bpack)
{
    for (int k = 0; k < kspan; ++k) {
        const double* src = &B[(k0 + k)*n + j0];
        double*       dst = &Bpack[k * jspan];
        std::copy(src, src + jspan, dst);
    }
}

static void blocked_packed_leaf(const std::vector<double>& A,
                                const std::vector<double>& B,
                                std::vector<double>& C,
                                int n, int T)
{
    std::fill(C.begin(), C.end(), 0.0);
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

                Bpack.resize((size_t)kspan * (size_t)jspan);
                pack_B_tile(B, n, kk, jj, kspan, jspan, Bpack);

                for (int i = ii; i < iimax; ++i) {
                    double* crow = &C[i*n + jj];
                    for (int k = 0; k < kspan; ++k) {
                        const double aik = A[i*n + (kk + k)];
                        const double* brow = &Bpack[(size_t)k * jspan];
                        for (int j = 0; j < jspan; ++j)
                            crow[j] += aik * brow[j];
                    }
                }
            }
        }
    }
}

// ---- рекурсивный гибрид: Strassen + tasks, leaf = блочное ядро ----
static void strassen_hybrid_rec_omp(const std::vector<double>& A,
                                    const std::vector<double>& B,
                                    std::vector<double>& C,
                                    int n, int leaf, int T, int cut)
{
    if (n <= leaf) { blocked_packed_leaf(A, B, C, n, T); return; }

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

    // Выходные и временные буферы
    std::vector<double> M1(m*m), M2(m*m), M3(m*m), M4(m*m), M5(m*m), M6(m*m), M7(m*m);
    std::vector<double> S1(m*m), S2(m*m), S3(m*m), S4(m*m), S5(m*m), S6(m*m), S7(m*m);
    std::vector<double> T1(m*m), T2(m*m), T3(m*m), T4(m*m), T5(m*m), T6(m*m), T7(m*m);

    const bool make_tasks = (n > cut);

    #pragma omp taskgroup
    {
        // M1 = (A11 + A22) * (B11 + B22)
        add(A11, A22, S1, m);  add(B11, B22, T1, m);
        #pragma omp task firstprivate(m,leaf,T,cut) if(make_tasks) shared(M1,S1,T1)
        { strassen_hybrid_rec_omp(S1, T1, M1, m, leaf, T, cut); }

        // M2 = (A21 + A22) * B11
        add(A21, A22, S2, m);
        #pragma omp task firstprivate(m,leaf,T,cut) if(make_tasks) shared(M2,S2,B11)
        { strassen_hybrid_rec_omp(S2, B11, M2, m, leaf, T, cut); }

        // M3 = A11 * (B12 - B22)
        sub(B12, B22, T3, m);
        #pragma omp task firstprivate(m,leaf,T,cut) if(make_tasks) shared(M3,A11,T3)
        { strassen_hybrid_rec_omp(A11, T3, M3, m, leaf, T, cut); }

        // M4 = A22 * (B21 - B11)
        sub(B21, B11, T4, m);
        #pragma omp task firstprivate(m,leaf,T,cut) if(make_tasks) shared(M4,A22,T4)
        { strassen_hybrid_rec_omp(A22, T4, M4, m, leaf, T, cut); }

        // M5 = (A11 + A12) * B22
        add(A11, A12, S5, m);
        #pragma omp task firstprivate(m,leaf,T,cut) if(make_tasks) shared(M5,S5,B22)
        { strassen_hybrid_rec_omp(S5, B22, M5, m, leaf, T, cut); }

        // M6 = (A21 - A11) * (B11 + B12)
        sub(A21, A11, S6, m);  add(B11, B12, T6, m);
        #pragma omp task firstprivate(m,leaf,T,cut) if(make_tasks) shared(M6,S6,T6)
        { strassen_hybrid_rec_omp(S6, T6, M6, m, leaf, T, cut); }

        // M7 = (A12 - A22) * (B21 + B22)
        sub(A12, A22, S7, m);  add(B21, B22, T7, m);
        #pragma omp task firstprivate(m,leaf,T,cut) if(make_tasks) shared(M7,S7,T7)
        { strassen_hybrid_rec_omp(S7, T7, M7, m, leaf, T, cut); }
    } // ждём все задачи

    // Сборка C из M1..M7
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
static int next_pow2(int n){ int p=1; while(p<n) p<<=1; return p; }

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");
    int leaf = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : 128;
    int T    = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 64;
    int cut  = (argc >= 5) ? std::max(1, std::stoi(argv[4])) : 256;

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

            double t0 = omp_get_wtime();
            #pragma omp parallel
            {
                #pragma omp single nowait
                strassen_hybrid_rec_omp(Ap, Bp, Cp, P, leaf, T, cut);
            }
            double t1 = omp_get_wtime();
            double secs = t1 - t0;

            std::ofstream out(out_path, std::ios::app);
            out << "omp_hybrid(strassen+blocked;leaf=" << leaf
                << ";T=" << T << ";cut=" << cut
                << ";p=" << omp_get_max_threads() << "),"
                << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
