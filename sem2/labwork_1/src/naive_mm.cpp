// naive_mm.cpp

/*
 * Алгоритм: наивное умножение матриц (i-j-k).
 *   Для каждого элемента C[i,j] вычисляем сумму:
 *     C[i,j] = Σ_{k=0..N-1} A[i,k] * B[k,j].
 *   Хранение: row-major в std::vector<double> длиной N*N.
 *   Доступ к данным: A читается по строке (хорошо для кэша),
 *                    B — по столбцу (stride = N; кэш-невыгодно).
 *
 * Сложность:
 *   Время:   Θ(N^3) ≈ 2·N^3 FLOPs.
 *   Память:  Θ(N^2) для A, B, C; доп. буферов нет.
 */


#include "config.h"
#include "mm_common.h"

// Classic i-j-k triple loop. B is accessed by columns (strided), i.e., cache-unfriendly.
static void mmul_naive(const std::vector<double>& A,
                       const std::vector<double>& B,
                       std::vector<double>& C,
                       int N) // все - одномерные массивы длиной N*N
{
    std::fill(C.begin(), C.end(), 0.0); // обнулим матрицу результатов
    for (int i = 0; i < N; ++i) { 
        for (int j = 0; j < N; ++j) {
            // формула C[i,j] = ∑k​ A[i,k]*B[k,j]
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i*N + k] * B[k*N + j]; 
            // при проходе по B перепрыгиваем в разные кэш-линии, элементы не подряд - плохо 
            }
            C[i*N + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    std::string out_path = (argc >= 2) ? argv[1] : std::string("results.csv");
    // создадим файл если его нет и запишем строку заголовок
    ensure_csv_header(out_path, "algo,N,repeat,time_sec"); 

    for (int N : SIZES) {
        // выделяем плоские буферы
        std::vector<double> A(N*N), B(N*N), C(N*N);
        // заполняем случ числами с воспроизводимостью
        fill_matrix(A, N, RNG_SEED + 1);
        fill_matrix(B, N, RNG_SEED + 2);

        for (int rep = 1; rep <= REPEAT; ++rep) {
            auto t0 = Clock::now();
            mmul_naive(A, B, C, N); 
            auto t1 = Clock::now();
            // чистое время умножения
            double secs = std::chrono::duration_cast<dsec>(t1 - t0).count();

            std::ofstream out(out_path, std::ios::app);
            out << "naive" << "," << N << "," << rep << "," << secs << "\n";
        }
    }
    return 0;
}
