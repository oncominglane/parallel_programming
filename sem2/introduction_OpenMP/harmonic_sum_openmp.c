// omp_harmonic_deterministic.c
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc != 2) { printf("Usage: %s N\n", argv[0]); return 1; }

    long long N = atoll(argv[1]);
    int T = omp_get_max_threads();
    double *partials = (double*)calloc(T, sizeof(double));
    int used = 0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp single
        used = omp_get_num_threads();          // сколько потоков реально запущено

        double s = 0.0;
        int nt = omp_get_num_threads();
        for (long long i = tid + 1; i <= N; i += nt)
            s += 1.0 / (double)i;
        partials[tid] = s;                     // каждый пишет свою часть в ячейку с индексом tid
    } // параллельная область закончена

    double total = 0.0;
    for (int t = 0; t < used; ++t)            // детерминированный порядок суммирования
        total += partials[t];

    printf("Partial harmonic sum for N = %lld is: %.12f\n", N, total);
    free(partials);
    return 0;
}

// Вариант B (ручная сводка): каждый поток пишет свою сумму в partials[tid],
// а потом один поток складывает их в фиксированном порядке 0,1,2,…. 
// Порядок операций жёстко задан → при неизменном числе потоков получается 
// бит-в-бит один и тот же результат.

// g++ -std=c++17 -O2 -fopenmp sem2/introduction_OpenMP/harmonic_sum_openmp.c -o sem2/introduction_OpenMP/omp
// OMP_NUM_THREADS=8 ./sem2/introduction_OpenMP/omp  10000000