#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "config.h"  // содержит #define M и K

#define a 1.0
#define L 1.0
#define T 1.0

int main() {
    double h = L / (M - 1);
    double tau = T / K;

    // Выделение памяти: u[K][M] → одномерный массив
    double *u = malloc(K * M * sizeof(double));
    if (!u) {
        printf("Ошибка выделения памяти\n");
        return 1;
    }

    // начальное условие: u[0][m] = sin(pi * x)
    for (int m = 0; m < M; m++) {
        double x = m * h;
        u[0 * M + m] = sin(3.1415926 * x);
    }

    // шаги по времени
    for (int k = 0; k < K - 1; k++) {
        for (int m = 1; m < M - 1; m++) {
            double avg = 0.5 * (u[k * M + m + 1] + u[k * M + m - 1]);
            double flux = (a * tau) / (2.0 * h) * (u[k * M + m + 1] - u[k * M + m - 1]);
            u[(k + 1) * M + m] = avg + flux;
        }
        // граничные условия
        u[(k + 1) * M + 0] = 0;
        u[(k + 1) * M + (M - 1)] = 0;
    }

    // запись в файл
    FILE *fout = fopen("sequential_transfer_eq.txt", "w");
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            fprintf(fout, "%.6f ", u[k * M + m]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
    free(u);

    return 0;
}