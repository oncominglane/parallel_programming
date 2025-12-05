/* numerov_bvp_seq_reduction.c
 *
 * Задача:  y'' - e^y = 0,  x ∈ [0,1]
 *          y(0) = 1,  y(1) = b,  b = 0,0.1,...,1
 *
 * Метод Нумерова (4-го порядка) + итерации Ньютона.
 * Трёхдиагональная система на каждом шаге Ньютона решается
 * редукционным алгоритмом (циклическое уменьшение размера системы),
 * без параллелизма.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_ITER   100
#define NEWTON_EPS 1e-8

typedef struct {
    int    n;
    double *a;
    double *b;
    double *c;
    double *d;
    int    *idx;
} Level;

/* Редукционный решатель трёхдиагональной системы:
 *
 * a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i], i = 0..n-1
 * с условиями a[0] = 0, c[n-1] = 0 (дальние узлы уже учтены в правой части).
 *
 * Вход:  n, a, b, c, d  (могут модифицироваться!)
 * Выход: x[0..n-1].
 */
static void solve_tridiag_reduction(int n, double *a_in, double *b_in,
                                    double *c_in, double *d_in,
                                    double *x_out)
{
    if (n <= 0) return;

    Level levels[64];   // с запасом, log2(4000) < 12
    int Lmax = 0;

    // Уровень 0 – исходная система
    levels[0].n   = n;
    levels[0].a   = a_in;
    levels[0].b   = b_in;
    levels[0].c   = c_in;
    levels[0].d   = d_in;
    levels[0].idx = (int*)malloc(n * sizeof(int));
    if (!levels[0].idx) {
        fprintf(stderr, "Memory allocation error in solve_tridiag_reduction\n");
        exit(1);
    }
    for (int i = 0; i < n; ++i)
        levels[0].idx[i] = i;

    /* ---------- Прямой ход: редукция размеров ---------- */

    while (levels[Lmax].n > 2) {
        if (Lmax + 1 >= 64) {
            fprintf(stderr, "Too many reduction levels\n");
            exit(1);
        }
        Level *cur  = &levels[Lmax];
        Level *next = &levels[Lmax + 1];

        int n_cur = cur->n;
        int last_even, has_right_boundary, internal_end;

        // n_cur >= 3 здесь гарантированно
        if ( ((n_cur - 1) & 1) == 0 ) {
            // n_cur нечётное => последний индекс (n_cur-1) чётный
            last_even         = n_cur - 1;
            has_right_boundary = 1;
            internal_end      = last_even - 2;
        } else {
            // n_cur чётное => последний индекс (n_cur-1) нечётный,
            // последний чётный индекс last_even = n_cur - 2
            last_even         = n_cur - 2;
            has_right_boundary = 0;
            internal_end      = last_even;
        }

        int n2 = last_even / 2 + 1;

        next->n   = n2;
        next->a   = (double*)malloc(n2 * sizeof(double));
        next->b   = (double*)malloc(n2 * sizeof(double));
        next->c   = (double*)malloc(n2 * sizeof(double));
        next->d   = (double*)malloc(n2 * sizeof(double));
        next->idx = (int*)   malloc(n2 * sizeof(int));

        if (!next->a || !next->b || !next->c || !next->d || !next->idx) {
            fprintf(stderr, "Memory allocation error in reduction forward\n");
            exit(1);
        }

        double *a = cur->a;
        double *b = cur->b;
        double *c = cur->c;
        double *d = cur->d;
        int    *idx = cur->idx;

        // --- левый край: j = 0, k = 0 ---
        {
            int j = 0;
            int k = 0;

            double bj0 = b[0];
            double cj0 = c[0];
            double aj1 = a[1];
            double bj1 = b[1];
            double cj1 = c[1];
            double dj0 = d[0];
            double dj1 = d[1];

            double inv_b1 = 1.0 / bj1;

            next->a[k]   = 0.0;
            next->b[k]   = bj0 - cj0 * aj1 * inv_b1;
            next->c[k]   = -cj0 * cj1 * inv_b1;
            next->d[k]   = dj0 - cj0 * dj1 * inv_b1;
            next->idx[k] = idx[j];
        }

        // --- внутренние чётные узлы ---
        for (int j = 2; j <= internal_end; j += 2) {
            int k = j / 2;

            int jm1 = j - 1;
            int jp1 = j + 1;

            double ai  = a[j];
            double bi  = b[j];
            double ci  = c[j];
            double di  = d[j];

            double am1 = a[jm1];
            double bm1 = b[jm1];
            double cm1 = c[jm1];
            double dm1 = d[jm1];

            double ap1 = a[jp1];
            double bp1 = b[jp1];
            double cp1 = c[jp1];
            double dp1 = d[jp1];

            double inv_bm1 = 1.0 / bm1;
            double inv_bp1 = 1.0 / bp1;

            next->a[k] = -ai * am1 * inv_bm1;
            next->c[k] = -ci * cp1 * inv_bp1;
            next->b[k] =  bi - ai * cm1 * inv_bm1 - ci * ap1 * inv_bp1;
            next->d[k] =  di - ai * dm1 * inv_bm1 - ci * dp1 * inv_bp1;

            next->idx[k] = idx[j];
        }

        // --- правый край, если есть чёткий правый узел ---
        if (has_right_boundary) {
            int j   = last_even;
            int k   = j / 2;
            int jm1 = j - 1;

            double aj  = a[j];
            double bj  = b[j];
            double dj  = d[j];

            double am1 = a[jm1];
            double bm1 = b[jm1];
            double cm1 = c[jm1];
            double dm1 = d[jm1];

            double inv_bm1 = 1.0 / bm1;

            next->a[k] = -aj * am1 * inv_bm1;
            next->b[k] =  bj - aj * cm1 * inv_bm1;
            next->c[k] =  0.0;
            next->d[k] =  dj - aj * dm1 * inv_bm1;

            next->idx[k] = idx[j];
        }

        Lmax++;
    }

    /* ---------- Решаем верхний уровень (1 или 2 уравнения) ---------- */

    Level *top = &levels[Lmax];
    int ntop   = top->n;
    double *at = top->a;
    double *bt = top->b;
    double *ct = top->c;
    double *dt = top->d;
    int    *idxt = top->idx;

    // x_out заполнен позже полностью, пока можно обнулить
    for (int i = 0; i < n; ++i)
        x_out[i] = 0.0;

    if (ntop == 1) {
        x_out[idxt[0]] = dt[0] / bt[0];
    } else if (ntop == 2) {
        // 2x2:
        // b0 x0 + c0 x1 = d0
        // a1 x0 + b1 x1 = d1
        double det = bt[0] * bt[1] - ct[0] * at[1];
        double x0  = ( dt[0] * bt[1] - ct[0] * dt[1] ) / det;
        double x1  = ( -dt[0] * at[1] + bt[0] * dt[1] ) / det;

        x_out[idxt[0]] = x0;
        x_out[idxt[1]] = x1;
    } else {
        fprintf(stderr, "Unexpected top-level size in reduction solver\n");
        exit(1);
    }

    /* ---------- Обратный ход: восстановление "выкинутых" узлов ---------- */

    for (int L = Lmax - 1; L >= 0; --L) {
        Level *cur = &levels[L];
        int nL     = cur->n;
        double *a  = cur->a;
        double *b  = cur->b;
        double *c  = cur->c;
        double *d  = cur->d;
        int    *idx = cur->idx;

        for (int j = 0; j < nL; ++j) {
            if (j % 2 == 1) {
                int glob = idx[j];
                int jm1  = j - 1;
                int jp1  = j + 1;

                double termL = 0.0;
                double termR = 0.0;

                if (jm1 >= 0)
                    termL = a[j] * x_out[idx[jm1]];
                if (jp1 < nL)
                    termR = c[j] * x_out[idx[jp1]];

                x_out[glob] = (d[j] - termL - termR) / b[j];
            }
        }
    }

    /* ---------- Освобождение памяти ---------- */

    for (int L = 0; L <= Lmax; ++L) {
        if (levels[L].idx)
            free(levels[L].idx);
        if (L > 0) {
            free(levels[L].a);
            free(levels[L].b);
            free(levels[L].c);
            free(levels[L].d);
        }
    }
}

/* Решение краевой задачи для заданного b (границы по x: 0..1).
 * M — число отрезков (узлов M+1, внутренних M-1).
 * Записывает решение в y_out[0..M].
 * Возвращает 0 при успехе, -1 если Ньютон не сошёлся или ошибка.
 */
static int solve_for_b(int M, double b_value, double *y_out)
{
    const double a_left  = 1.0;       // y(0) = 1
    const double b_right = b_value;   // y(1) = b
    const double h       = 1.0 / M;

    int N = M - 1;     // число внутренних узлов (неизвестных)

    if (N < 1) {
        fprintf(stderr, "Too few grid points\n");
        return -1;
    }

    double *y     = (double*)malloc((M + 1) * sizeof(double));
    double *f     = (double*)malloc((M + 1) * sizeof(double));
    double *F     = (double*)malloc(N * sizeof(double));
    double *a_tr  = (double*)malloc(N * sizeof(double));
    double *b_tr  = (double*)malloc(N * sizeof(double));
    double *c_tr  = (double*)malloc(N * sizeof(double));
    double *rhs   = (double*)malloc(N * sizeof(double));
    double *delta = (double*)malloc(N * sizeof(double));

    if (!y || !f || !F || !a_tr || !b_tr || !c_tr || !rhs || !delta) {
        fprintf(stderr, "Memory allocation error in solve_for_b\n");
        free(y); free(f); free(F); free(a_tr); free(b_tr);
        free(c_tr); free(rhs); free(delta);
        return -1;
    }

    // начальное приближение: линейная интерполяция между a_left и b_right
    for (int i = 0; i <= M; ++i) {
        double x = i * h;
        y[i] = a_left + (b_right - a_left) * x;
    }
    y[0] = a_left;
    y[M] = b_right;

    int iter;
    for (iter = 0; iter < MAX_ITER; ++iter) {
        // f = e^y
        for (int i = 0; i <= M; ++i)
            f[i] = exp(y[i]);

        // формируем невязку и матрицу Якоби
        for (int m = 1; m <= M - 1; ++m) {
            int k = m - 1; // индекс 0..N-1

            double term1 = (y[m+1] - 2.0 * y[m] + y[m-1]) / (h * h);
            double term2 = f[m] + (1.0/12.0) * (f[m+1] - 2.0 * f[m] + f[m-1]);
            F[k] = term1 - term2;  // F_m(y)

            // производные F по y_{m-1}, y_m, y_{m+1}
            double alpha =  1.0/(h*h) - (1.0/12.0) * f[m-1]; // dF/dy_{m-1}
            double beta  = -2.0/(h*h) - (5.0/6.0)  * f[m];   // dF/dy_m
            double gamma =  1.0/(h*h) - (1.0/12.0) * f[m+1]; // dF/dy_{m+1}

            // коэффициенты трёхдиагональной системы (только по внутренним узлам)
            if (m > 1)
                a_tr[k] = alpha;
            else
                a_tr[k] = 0.0; // нет неизвестного слева

            b_tr[k] = beta;

            if (m < M - 1)
                c_tr[k] = gamma;
            else
                c_tr[k] = 0.0; // нет неизвестного справа

            rhs[k] = -F[k];
        }

        // решаем трёхдиагональную систему редукционным методом
        solve_tridiag_reduction(N, a_tr, b_tr, c_tr, rhs, delta);

        // обновление и оценка максимального приращения
        double max_delta = 0.0;
        for (int m = 1; m <= M - 1; ++m) {
            int k = m - 1;
            y[m] += delta[k];
            double ad = fabs(delta[k]);
            if (ad > max_delta)
                max_delta = ad;
        }

        if (max_delta < NEWTON_EPS)
            break;
    }

    if (iter == MAX_ITER) {
        fprintf(stderr, "Warning: Newton did not converge for b = %g\n", b_value);
        // можно считать ошибкой, но пусть пока будет только предупреждение
    }

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
        fprintf(stderr, "Usage: %s N_points\n", argv[0]);
        fprintf(stderr, "N_points: total number of grid points on [0,1], 400..4000\n");
        return 1;
    }

    int total_points = atoi(argv[1]); // число узлов на отрезке
    if (total_points < 3) {
        fprintf(stderr, "N_points must be >= 3\n");
        return 1;
    }

    int M = total_points - 1;  // число отрезков
    double h = 1.0 / M;

    double *y = (double*)malloc((M + 1) * sizeof(double));
    if (!y) {
        fprintf(stderr, "Memory allocation error in main\n");
        return 1;
    }

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
