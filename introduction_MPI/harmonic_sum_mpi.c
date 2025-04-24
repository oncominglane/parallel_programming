#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, N;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // всего процессов

    if (argc != 2) {
        if (rank == 0)
            printf("Usage: %s N\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    N = atoi(argv[1]);
    double local_sum = 0.0;

    // каждый процесс считает свою часть ряда
    for (int i = rank + 1; i <= N; i += size) {
        local_sum += 1.0 / i;
    }

    double total_sum = 0.0;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Partial harmonic sum for N = %d is: %.12f\n", N, total_sum);
    }

    MPI_Finalize();
    return 0;
}
