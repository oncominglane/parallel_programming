#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 10000        // количество точек по x
#define K 5000        // количество шагов по времени
#define a 1.0        // скорость переноса
#define L 1.0        // длина отрезка
#define T 1.0        // общее время

int main() {
    double h = L / (M - 1);
    double tau = T / K;

    double u[K][M];      // решение u[k][m]

    // начальное условие: sin(pi * x)
    for (int m = 0; m < M; m++) {
        double x = m * h;
        u[0][m] = sin(3.1415926 * x);
    }

    // шаги по времени
    for (int k = 0; k < K - 1; k++) {
        for (int m = 1; m < M - 1; m++) {
            double avg = 0.5 * (u[k][m + 1] + u[k][m - 1]);
            double flux = (a * tau) / (2.0 * h) * (u[k][m + 1] - u[k][m - 1]);
            u[k + 1][m] = avg + flux;
        }
        u[k + 1][0] = u[k + 1][M - 1] = 0;  // граничные условия
    }

    // запись в файл
    FILE *fout = fopen("sequential_transfer_eq.txt", "w");
    for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
            fprintf(fout, "%.6f ", u[k][m]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);

    return 0;
}
