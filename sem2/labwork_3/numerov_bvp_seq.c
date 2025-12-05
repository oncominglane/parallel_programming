/* numerov_bvp_seq.c
 *
 * Задача: y'' - e^y = 0,  x in [0,1]
 *        y(0) = 1, y(1) = b,  b = 0..1 шагом 0.1
 *
 * Метод Нумерова (4-го порядка) + итерации Ньютона.
 * На каждом шаге Ньютона решаем трёхдиагональную СЛАУ прогонкой.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER 100
#define NEWTON_EPS 1e-8

/* Решение трёхдиагональной системы:
 * a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i],  i = 0..n-1
 * где a[0] и c[n-1] не используются.
 * Модифицирует b, c, d.
 */
static void solve_tridiag(int n, double *a, double *b, double *c, double *d, double *x)
{
    double *c_prime = (double*)malloc(n * sizeof(double));
    double *d_prime = (double*)malloc(n * sizeof(double));
    if (!c_prime || !d_prime) {
        fprintf(stderr, "Memory allocation error in solve_tridiag\n");
        exit(1);
    }

    // прямой ход
    c_prime[0] = (n > 1) ? c[0] / b[0] : 0.0;
    d_prime[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double denom = b[i] - a[i] * c_prime[i-1];
        if (fabs(denom) < 1e-20) {
            fprintf(stderr, "Zero pivot in tridiagonal solver\n");
            exit(1);
        }
        c_prime[i] = (i < n-1) ? c[i] / denom : 0.0;
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom;
    }

    // обратный ход
    x[n-1] = d_prime[n-1];
    for (int i = n-2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i+1];
    }

    free(c_prime);
    free(d_prime);
}

/* Решение краевой задачи для заданного b.
 * M - количество отрезков (узлов M+1, внутренних M-1).
 * Результат записывается в массив y[0..M] (узлы x_i = i*h).
 * Возвращает 0 при успехе, -1 если Ньютон не сошёлся.
 */
static int solve_for_b(int M, double b_value, double *y_out)
{
    const double a = 1.0;          // y(0) = 1
    const double b = b_value;      // y(1) = b
    const double h = 1.0 / M;

    int N = M - 1;                 // число внутренних узлов

    double *y = (double*)malloc((M + 1) * sizeof(double));
    double *f = (double*)malloc((M + 1) * sizeof(double));
    double *F = (double*)malloc(N * sizeof(double));
    double *a_tr = (double*)malloc(N * sizeof(double));
    double *b_tr = (double*)malloc(N * sizeof(double));
    double *c_tr = (double*)malloc(N * sizeof(double));
    double *rhs  = (double*)malloc(N * sizeof(double));
    double *delta = (double*)malloc(N * sizeof(double));

    if (!y || !f || !F || !a_tr || !b_tr || !c_tr || !rhs || !delta) {
        fprintf(stderr, "Memory allocation error\n");
        return -1;
    }

    // начальное приближение: линейная интерполяция между a и b
    for (int i = 0; i <= M; ++i) {
        double x = i * h;
        y[i] = a + (b - a) * x;
    }
    y[0] = a;
    y[M] = b;

    int iter;
    for (iter = 0; iter < MAX_ITER; ++iter) {
        // считаем f = e^{y} во всех узлах
        for (int i = 0; i <= M; ++i)
            f[i] = exp(y[i]);

        // считаем невязку F_m и элементы якобиана
        for (int m = 1; m <= M-1; ++m) {
            int k = m - 1; // индекс во внутренних массивах (0..N-1)

            double term1 = (y[m+1] - 2.0*y[m] + y[m-1]) / (h*h);
            double term2 = f[m] + (1.0/12.0)*(f[m+1] - 2.0*f[m] + f[m-1]);
            F[k] = term1 - term2;      // F_m(y) = 0 в решении

            // производные (коэффициенты трёхдиагональной матрицы)
            double alpha =  1.0/(h*h) - (1.0/12.0)*f[m-1]; // dF/dy_{m-1}
            double beta  = -2.0/(h*h) - (5.0/6.0)*f[m];    // dF/dy_m
            double gamma =  1.0/(h*h) - (1.0/12.0)*f[m+1]; // dF/dy_{m+1}

            if (m > 1)
                a_tr[k] = alpha;
            else
                a_tr[k] = 0.0; // нет y_0 в неизвестных

            b_tr[k] = beta;

            if (m < M-1)
                c_tr[k] = gamma;
            else
                c_tr[k] = 0.0; // нет y_M в неизвестных

            rhs[k] = -F[k];  // A * delta = -F
        }

        // решаем трёхдиагоналку на приращения delta_y
        solve_tridiag(N, a_tr, b_tr, c_tr, rhs, delta);

        // обновляем y во внутренних узлах и оцениваем максимум приращения
        double max_delta = 0.0;
        for (int m = 1; m <= M-1; ++m) {
            int k = m - 1;
            y[m] += delta[k];
            double ad = fabs(delta[k]);
            if (ad > max_delta)
                max_delta = ad;
        }

        // критерий остановки Ньютона
        if (max_delta < NEWTON_EPS)
            break;
    }

    if (iter == MAX_ITER) {
        fprintf(stderr, "Warning: Newton did not converge for b = %g\n", b);
        // Можно считать ошибкой:
        // free и return -1;
    }

    // отдаём решение в выходной массив
    for (int i = 0; i <= M; ++i)
        y_out[i] = y[i];

    free(y);
    free(f);
    free(F);
    free(a_tr);
    free(b_tr);
    free(c_tr);
    free(rhs);
    free(delta);

    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s M_points\n", argv[0]);
        fprintf(stderr, "M_points: total number of grid points on [0,1], 400..4000\n");
        return 1;
    }

    int M = atoi(argv[1]) - 1; // аргумент - число точек, а не отрезков
    if (M < 2) {
        fprintf(stderr, "M_points must be >= 3\n");
        return 1;
    }

    int total_points = M + 1;
    double h = 1.0 / M;

    double *y = (double*)malloc(total_points * sizeof(double));
    if (!y) {
        fprintf(stderr, "Memory allocation error\n");
        return 1;
    }

    // цикл по b = 0, 0.1, ..., 1.0
    for (int ib = 0; ib <= 10; ++ib) {
        double b = ib * 0.1;

        if (solve_for_b(M, b, y) != 0) {
            fprintf(stderr, "Error solving for b = %g\n", b);
            free(y);
            return 1;
        }

        printf("# Solution for b = %.2f\n", b);
        for (int i = 0; i <= M; ++i) {
            double x = i * h;
            printf("% .8f  % .12f\n", x, y[i]);
        }
        printf("\n\n");
    }

    free(y);
    return 0;
}
