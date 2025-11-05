/*
 * Алгоритм: умножение Страссена (Strassen).
 *   Делим A и B на квадранты (N = 2^k): A11,A12,A21,A22 и B11,B12,B21,B22.
 *   Вычисляем 7 произведений:
 *     M1 = (A11 + A22) * (B11 + B22)
 *     M2 = (A21 + A22) * B11
 *     M3 = A11 * (B12 - B22)
 *     M4 = A22 * (B21 - B11)
 *     M5 = (A11 + A12) * B22
 *     M6 = (A21 - A11) * (B11 + B12)
 *     M7 = (A12 - A22) * (B21 + B22)
 *   И собираем C из:
 *     C11 = M1 + M4 - M5 + M7
 *     C12 = M3 + M5
 *     C21 = M2 + M4
 *     C22 = M1 - M2 + M3 + M6
 *
 * Особенности реализации:
 *   - Рекурсивно; при размере <= LEAF используем обычное наивное умножение.
 *   - Для произвольных N делаем паддинг до ближайшей степени 2, затем обрезаем.
 *
 * Сложность:
 *   Время: Θ(N^log2 7) ≈ Θ(N^2.807) асимптотически (без учёта копирований).
 *   Память: Θ(N^2) + временные буферы O((N/2)^2) на каждом уровне рекурсии.
 *
 * CSV-вывод: algo,N,repeat,time_sec (algo = "strassen(leaf=...)").
 */
#include "config.h"
#include "mm_common.h"
#include <cassert>

// ----- базовые операции над квадратными матрицами (контейнером выступает std::vector<double>) -----

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

// копируем квадратный блок (r0..r0+n-1, c0..c0+n-1) из src(N×N) в dst(n×n)
static void copy_block(const std::vector<double>& src, int N, int r0, int c0,
                       std::vector<double>& dst, int n)
{
    for (int i = 0; i < n; ++i) {
        const double* srow = &src[(r0 + i)*N + c0];
        double* drow = &dst[i*n];
        std::copy(srow, srow + n, drow);
    }
}

// кладём квадратный блок src(n×n) в позицию (r0,c0) матрицы dst(N×N)
static void write_block(const std::vector<double>& src, int n,
                        std::vector<double>& dst, int N, int r0, int c0)
{
    for (int i = 0; i < n; ++i) {
        const double* srow = &src[i*n];
        double* drow = &dst[(r0 + i)*N + c0];
        std::copy(srow, srow + n, drow);
    }
}

// рекурсивная часть Страссена: перемножает n×n, результат в C
static void strassen_rec(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C,
                         int n, int leaf)
{
    if (n <= leaf) { mul_naive(A, B, C, n); return; }

    const int m = n / 2;

    // разбиение на квадранты (копированием в плотные буферы m×m)
    std::vector<double> A11(m*m), A12(m*m), A21(m*m), A22(m*m);
    std::vector<double> B11(m*m), B12(m*m), B21(m*m), B22(m*m);
    // верх-лево: (0,0); верх-право: (0,m); низ-лево: (m,0); низ-право: (m,m)
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

    // M1 = (A11 + A22) * (B11 + B22)
    add(A11, A22, T1, m);
    add(B11, B22, T2, m);
    strassen_rec(T1, T2, M1, m, leaf);

    // M2 = (A21 + A22) * B11
    add(A21, A22, T1, m);
    strassen_rec(T1, B11, M2, m, leaf);

    // M3 = A11 * (B12 - B22)
    sub(B12, B22, T2, m);
    strassen_rec(A11, T2, M3, m, leaf);

    // M4 = A22 * (B21 - B11)
    sub(B21, B11, T2, m);
    strassen_rec(A22, T2, M4, m, leaf);

    // M5 = (A11 + A12) * B22
    add(A11, A12, T1, m);
    strassen_rec(T1, B22, M5, m, leaf);

    // M6 = (A21 - A11) * (B11 + B12)
    sub(A21, A11, T1, m);
    add(B11, B12, T2, m);
    strassen_rec(T1, T2, M6, m, leaf);

    // M7 = (A12 - A22) * (B21 + B22)
    sub(A12, A22, T1, m);
    add(B21, B22, T2, m);
    strassen_rec(T1, T2, M7, m, leaf);

    // C11 = M1 + M4 - M5 + M7
    std::vector<double> C11 = M1;  add_inplace(C11, M4, m);  sub_inplace(C11, M5, m);  add_inplace(C11, M7, m);
    // C12 = M3 + M5
    std::vector<double> C12 = M3;  add_inplace(C12, M5, m);
    // C21 = M2 + M4
    std::vector<double> C21 = M2;  add_inplace(C21, M4, m);
    // C22 = M1 - M2 + M3 + M6
    std::vector<double> C22 = M1;  sub_inplace(C22, M2, m);  add_inplace(C22, M3, m);  add_inplace(C22, M6, m);

    // собрать квадранты в C
    write_block(C11, m, C, n, 0, 0);
    write_block(C12, m, C, n, 0, m);
    write_block(C21, m, C, n, m, 0);
    write_block(C22, m, C, n, m, m);
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");
    int LEAF = (argc >= 3) ? std::max(1, std::stoi(argv[2])) : 64; // порог перехода на наивный

    ensure_csv_header(out_path, "algo,N,repeat,time_sec");

    for (int N : SIZES) {
        // инициализация исходных матриц N×N
        std::vector<double> A(N*N), B(N*N);
        fill_matrix(A, N, RNG_SEED + 1);
        fill_matrix(B, N, RNG_SEED + 2);

        // паддинг до степени 2
        int P = next_pow2(N);
        std::vector<double> Ap(P*P, 0.0), Bp(P*P, 0.0), Cp(P*P, 0.0);
        // копирование в левый верхний угол
        for (int i = 0; i < N; ++i) {
            std::copy(A.begin() + i*N, A.begin() + (i+1)*N, Ap.begin() + i*P);
            std::copy(B.begin() + i*N, B.begin() + (i+1)*N, Bp.begin() + i*P);
        }

        for (int rep = 1; rep <= REPEAT; ++rep) {
            std::fill(Cp.begin(), Cp.end(), 0.0);
            auto t0 = Clock::now();
            strassen_rec(Ap, Bp, Cp, P, LEAF);
            auto t1 = Clock::now();
            double secs = std::chrono::duration<double>(t1 - t0).count();

            // (результат при желании можно обрезать до N×N — сейчас это не требуется для бенчмарка)

            std::ofstream out(out_path, std::ios::app);
            out << "strassen(leaf=" << LEAF << ")," << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
