
#include <omp.h>
#include <stdio.h>

int main(void) {
    int shared = 0;  // общая ячейка

    #pragma omp parallel shared(shared)  // компилятор создаст пул потоков
    {
        int tid = omp_get_thread_num();   // номер потока
        int nt  = omp_get_num_threads();   // сколько потоков запущено

        for (int t = 0; t < nt; ++t) {
            #pragma omp barrier   // все ждут начала тура t
            if (tid == t) {
                shared += 1;    // «произвольное действие»
                printf("Thread %d: shared = %d\n", tid, shared);
            }
            #pragma omp barrier  // ждём завершения тура t
        }
    } // выход: implicit barrier

    return 0;
}

// g++ -std=c++17 -O2 -fopenmp sem2/introduction_OpenMP/circle_message_openmp.c -o sem2/introduction_OpenMP/omp
// OMP_NUM_THREADS=4 OMP_DYNAMIC=false ./sem2/introduction_OpenMP/omp 
