
#include <omp.h>
#include <stdio.h>

int main(void) {
    int shared = 0;  // общая ячейка

    #pragma omp parallel shared(shared)         
    {
        int tid = omp_get_thread_num();   // номер потока
        int nt  = omp_get_num_threads();   // сколько потоков запущено

        for (int t = 0; t < nt; ++t) {
            #pragma omp barrier   // все выровнялись на тур t
            if (tid == t) {
                shared += 1;    // «произвольное действие»
                printf("Thread %d: shared = %d\n", tid, shared);
            }
            #pragma omp barrier  // ждём завершения тура t
        }
    } // выход: implicit barrier

    return 0;
}

// g++ -std=c++17 -O2 -fopenmp introduction_OpenMP/circle_message_openmp.c -o introduction_OpenMP/omp
// OMP_NUM_THREADS=4 OMP_DYNAMIC=false ./introduction_OpenMP/omp 
