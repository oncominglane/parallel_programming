#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "config.h"

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
    int stride = local_M + 2;

    if (M % size != 0 && rank == 0) {
        printf("M must be divisible by number of processes!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Выделение памяти: u[K][stride] → 1D массив
    double *u = malloc(sizeof(double) * K * stride);
    if (!u) {
        printf("Malloc failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // начальное условие
    for (int m = 1; m <= local_M; m++) {
        int global_m = rank * local_M + m - 1;
        double x = global_m * h;
        u[0 * stride + m] = sin(3.1415926 * x);
    }
    u[0 * stride + 0] = 0.0;
    u[0 * stride + (local_M + 1)] = 0.0;

    // основная итерация по времени
    for (int k = 0; k < K - 1; k++) {
        double *cur = &u[k * stride];
        double *next = &u[(k + 1) * stride];

        // обмен границ
        if (rank > 0) {
            MPI_Sendrecv(&cur[1], 1, MPI_DOUBLE, rank - 1, 0,
                         &cur[0], 1, MPI_DOUBLE, rank - 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Sendrecv(&cur[local_M], 1, MPI_DOUBLE, rank + 1, 0,
                         &cur[local_M + 1], 1, MPI_DOUBLE, rank + 1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // расчёт новой строки
        for (int m = 1; m <= local_M; m++) {
            double avg = 0.5 * (cur[m + 1] + cur[m - 1]);
            double flux = (a * tau) / (2.0 * h) * (cur[m + 1] - cur[m - 1]);
            next[m] = avg + flux;
        }

        if (rank == 0) next[1] = 0.0;
        if (rank == size - 1) next[local_M] = 0.0;
    }

    // сбор всех временных слоёв
    double *full_data = NULL;
    if (rank == 0) {
        full_data = malloc(sizeof(double) * K * M);
    }

    double *sendbuf = malloc(sizeof(double) * local_M);
    for (int k = 0; k < K; k++) {
        double *row = &u[k * stride];
        for (int m = 0; m < local_M; m++) {
            sendbuf[m] = row[m + 1];
        }

        MPI_Gather(sendbuf, local_M, MPI_DOUBLE,
                   full_data ? &full_data[k * M] : NULL,
                   local_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // запись результата
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
    free(u);
    MPI_Finalize();
    return 0;
}
