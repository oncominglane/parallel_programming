/*
 * Алгоритм: Strassen + OpenMP tasks.
 *
 * Параллелизм: на каждом уровне рекурсии 7 независимых умножений M1..M7
 *              запускаются как OpenMP-задачи (если размер подзадачи > cut).
 * База рекурсии: при n <= leaf используем обычное наивное умножение (i-k-j).
 *
 * Сложность: асимптотически Θ(N^log2 7) ≈ Θ(N^2.807); память Θ(N^2) + временные буферы.
 * Замечания:
 *   - Для произвольных N делаем паддинг до ближайшей степени 2 и обрезаем.
 *   - Чтобы CSV не «ломался», в метке algo нет запятых (используем ';').
 *
 * Сборка:
 *   g++ -O3 -march=native -std=c++17 -fopenmp strassen_omp.cpp -o strassen_omp
 *
 * Запуск:
 *   OMP_NUM_THREADS=8 ./strassen_omp results.csv 64 256
 *   #                  ^out_path      ^leaf ^cut
 */

#include "config.h"
#include "mm_common.h"
#include <omp.h>
#include <cassert>
#include <algorithm>

// --------- базовые операции над плотными квадратными матрицами n×n ---------

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

static void mul_naive(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, int n)
{
    std::fill(C.begin(), C.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const double aik = A[i*n + k];
            for (int j = 0; j < n; ++j) {
                C[i*n + j] += aik * B[k*n + j];
            }
        }
    }
}

// скопировать подблок src(N×N)[r0:r0+n, c0:c0+n) -> dst(n×n)
static void copy_block(const std::vector<double>& src, int N, int r0, int c0,
                       std::vector<double>& dst, int n)
{
    for (int i = 0; i < n; ++i) {
        const double* srow = &src[(r0 + i)*N + c0];
        std::copy(srow, srow + n, &dst[i*n]);
    }
}

// dst(N×N)[r0:r0+n, c0:c0+n) = src(n×n)
static void write_block(const std::vector<double>& src, int n,
                        std::vector<double>& dst, int N, int r0, int c0)
{
    for (int i = 0; i < n; ++i) {
        std::copy(&src[i*n], &src[i*n] + n, &dst[(r0 + i)*N + c0]);
    }
}

// --------- рекурсивная часть: при больших n создаём OpenMP-задачи ---------

static void strassen_rec_omp(const std::vector<double>& A,
                             const std::vector<double>& B,
                             std::vector<double>& C,
                             int n, int leaf, int cut)
{
    if (n <= leaf) { mul_naive(A, B, C, n); return; }

    const int m = n / 2;

    // Разбиение A и B на квадранты (в отдельные плотные буферы m×m).
    std::vector<double> A11(m*m), A12(m*m), A21(m*m), A22(m*m);
    std::vector<double> B11(m*m), B12(m*m), B21(m*m), B22(m*m);
    copy_block(A, n, 0, 0, A11, m);  copy_block(A, n, 0, m, A12, m);
    copy_block(A, n, m, 0, A21, m);  copy_block(A, n, m, m, A22, m);
    copy_block(B, n, 0, 0, B11, m);  copy_block(B, n, 0, m, B12, m);
    copy_block(B, n, m, 0, B21, m);  copy_block(B, n, m, m, B22, m);

    // Выходные блоки C
    std::vector<double> C11(m*m), C12(m*m), C21(m*m), C22(m*m);

    // Временные суммы/разности для аргументов M1..M7 (каждой задаче — свои буферы)
    std::vector<double> M1(m*m), M2(m*m), M3(m*m), M4(m*m), M5(m*m), M6(m*m), M7(m*m);
    std::vector<double> S1(m*m), S2(m*m), S3(m*m), S4(m*m), S5(m*m), S6(m*m), S7(m*m);
    std::vector<double> T1(m*m), T2(m*m), T3(m*m), T4(m*m), T5(m*m), T6(m*m), T7(m*m);

    // Создаём задачи только если подзадача достаточно крупная
    const bool make_tasks = (n > cut);

    #pragma omp taskgroup
    {
        // M1 = (A11 + A22) * (B11 + B22)
        add(A11, A22, S1, m);
        add(B11, B22, T1, m);
        #pragma omp task firstprivate(m,leaf,cut) if(make_tasks) shared(M1,S1,T1)
        { strassen_rec_omp(S1, T1, M1, m, leaf, cut); }

        // M2 = (A21 + A22) * B11
        add(A21, A22, S2, m);
        #pragma omp task firstprivate(m,leaf,cut) if(make_tasks) shared(M2,S2,B11)
        { strassen_rec_omp(S2, B11, M2, m, leaf, cut); }

        // M3 = A11 * (B12 - B22)
        sub(B12, B22, T3, m);
        #pragma omp task firstprivate(m,leaf,cut) if(make_tasks) shared(M3,A11,T3)
        { strassen_rec_omp(A11, T3, M3, m, leaf, cut); }

        // M4 = A22 * (B21 - B11)
        sub(B21, B11, T4, m);
        #pragma omp task firstprivate(m,leaf,cut) if(make_tasks) shared(M4,A22,T4)
        { strassen_rec_omp(A22, T4, M4, m, leaf, cut); }

        // M5 = (A11 + A12) * B22
        add(A11, A12, S5, m);
        #pragma omp task firstprivate(m,leaf,cut) if(make_tasks) shared(M5,S5,B22)
        { strassen_rec_omp(S5, B22, M5, m, leaf, cut); }

        // M6 = (A21 - A11) * (B11 + B12)
        sub(A21, A11, S6, m);
        add(B11, B12, T6, m);
        #pragma omp task firstprivate(m,leaf,cut) if(make_tasks) shared(M6,S6,T6)
        { strassen_rec_omp(S6, T6, M6, m, leaf, cut); }

        // M7 = (A12 - A22) * (B21 + B22)
        sub(A12, A22, S7, m);
        add(B21, B22, T7, m);
        #pragma omp task firstprivate(m,leaf,cut) if(make_tasks) shared(M7,S7,T7)
        { strassen_rec_omp(S7, T7, M7, m, leaf, cut); }
    } // taskgroup автоматически ждёт все задачи

    // Сборка блоков C
    // C11 = M1 + M4 - M5 + M7
    C11 = M1; add_inplace(C11, M4, m); sub_inplace(C11, M5, m); add_inplace(C11, M7, m);
    // C12 = M3 + M5
    C12 = M3; add_inplace(C12, M5, m);
    // C21 = M2 + M4
    C21 = M2; add_inplace(C21, M4, m);
    // C22 = M1 - M2 + M3 + M6
    C22 = M1; sub_inplace(C22, M2, m); add_inplace(C22, M3, m); add_inplace(C22, M6, m);

    // Запись в результирующую матрицу C
    write_block(C11, m, C, n, 0, 0);
    write_block(C12, m, C, n, 0, m);
    write_block(C21, m, C, n, m, 0);
    write_block(C22, m, C, n, m, m);
}

// ближайшая степень двойки ≥ n
static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");
    int LEAF = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : 64;   // порог «лист»
    int CUT  = (argc >= 4) ? std::max(1, std::stoi(argv[3])) : 256;  // порог создания tasks

    ensure_csv_header(out_path, "algo,N,repeat,time_sec");

    for (int N : SIZES) {
        // Инициализация входов N×N
        std::vector<double> A(N*N), B(N*N);
        fill_matrix(A, N, RNG_SEED + 1);
        fill_matrix(B, N, RNG_SEED + 2);

        // Паддинг до степени 2
        int P = next_pow2(N);
        std::vector<double> Ap(P*P, 0.0), Bp(P*P, 0.0), Cp(P*P, 0.0);
        for (int i = 0; i < N; ++i) {
            std::copy(A.begin() + i*N, A.begin() + (i+1)*N, Ap.begin() + i*P);
            std::copy(B.begin() + i*N, B.begin() + (i+1)*N, Bp.begin() + i*P);
        }

        for (int rep = 1; rep <= REPEAT; ++rep) {
            std::fill(Cp.begin(), Cp.end(), 0.0);

            double t0 = omp_get_wtime();
            // один пул потоков на весь вызов; дальнейшая рекурсия порождает задачи
            #pragma omp parallel
            {
                #pragma omp single nowait
                strassen_rec_omp(Ap, Bp, Cp, P, LEAF, CUT);
            }
            double t1 = omp_get_wtime();
            double secs = t1 - t0;

            std::ofstream out(out_path, std::ios::app);
            out << "omp_strassen(leaf=" << LEAF
                << ";cut=" << CUT
                << ";p=" << omp_get_max_threads() << "),"
                << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
