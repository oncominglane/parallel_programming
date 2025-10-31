// omp_hello.c
#include <omp.h>
#include <stdio.h>

int main(void) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();      // номер текущего потока
        int nt  = omp_get_num_threads();     // общее число потоков в паралл. области
        printf("Hello World! I'm thread %d out of %d\n", tid, nt);
    } // конец параллельной области 
    return 0;
}
