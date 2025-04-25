#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define M 10000
#define K 5000
#define a 1.0
#define L 1.0
#define T 1.0

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double h = L / (M - 1);
    double tau = T / K;

    int local_M = M / size;
    if (M % size != 0 && rank == 0) {
        printf("M must be divisible by number of processes!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double u[K][local_M + 2];  // +2 для границ

    // начальное условие
    for (int m = 1; m <= local_M; m++) {
        int global_m = rank * local_M + m - 1;
        double x = global_m * h;
        u[0][m] = sin(3.1415926 * x);
    }
    u[0][0] = u[0][local_M + 1] = 0;

    // основная итерация
    for (int k = 0; k < K - 1; k++) {
        // обмен краями
        if (rank > 0) {
            MPI_Sendrecv(&u[k][1], 1, MPI_DOUBLE, rank - 1, 0,
                         &u[k][0], 1, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&u[k][local_M], 1, MPI_DOUBLE, rank + 1, 0,
                         &u[k][local_M + 1], 1, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // расчёт новой строки
        for (int m = 1; m <= local_M; m++) {
            double avg = 0.5 * (u[k][m + 1] + u[k][m - 1]);
            double flux = (a * tau) / (2.0 * h) * (u[k][m + 1] - u[k][m - 1]);
            u[k + 1][m] = avg + flux;
        }

        if (rank == 0) u[k + 1][1] = 0.0;
        if (rank == size - 1) u[k + 1][local_M] = 0.0;
    }

    // сбор всех временных слоёв
    double *full_data = NULL;
    if (rank == 0) {
        full_data = malloc(sizeof(double) * K * M);
    }

    // временный буфер для одного слоя
    double *sendbuf = malloc(sizeof(double) * local_M);
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < local_M; m++) {
            sendbuf[m] = u[k][m + 1];
        }

        MPI_Gather(sendbuf, local_M, MPI_DOUBLE,
                   full_data ? &full_data[k * M] : NULL,
                   local_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // запись
    if (rank == 0) {
        FILE *f = fopen("parallel_transfer_eq.txt", "w");
        for (int k = 0; k < K; k++) {
            for (int m = 0; m < M; m++) {
                fprintf(f, "%.6f ", full_data[k * M + m]);
            }
            fprintf(f, "\n");
        }
        fclose(f);
        free(full_data);
    }

    free(sendbuf);
    MPI_Finalize();
    return 0;
}
