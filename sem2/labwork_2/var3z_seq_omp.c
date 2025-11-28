/* var3z_seq_omp.c */
#include "common.h"

void var3z_seq(double **a, double **b) {
    // первый цикл
    for (int i = 0; i < ISIZE; ++i)
        for (int j = 0; j < JSIZE; ++j)
            a[i][j] = sin(0.1 * a[i][j]);

    // второй цикл
    for (int i = 1; i < ISIZE; ++i)
        for (int j = 0; j < JSIZE - 2; ++j)
            b[i][j] = a[i-1][j+2] * 1.5;
}

void var3z_omp(double **a, double **b) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ISIZE; ++i)
        for (int j = 0; j < JSIZE; ++j)
            a[i][j] = sin(0.1 * a[i][j]);

    #pragma omp parallel for collapse(2)
    for (int i = 1; i < ISIZE; ++i)
        for (int j = 0; j < JSIZE - 2; ++j)
            b[i][j] = a[i-1][j+2] * 1.5;
}

int main(int argc, char **argv) {
    int use_omp = 0;
    const char *out_a = "results/result_3z_a.txt";
    const char *out_b = "results/result_3z_b.txt";

    if (argc > 1 && strcmp(argv[1], "omp") == 0)
        use_omp = 1;
    if (argc > 2)
        out_a = argv[2];
    if (argc > 3)
        out_b = argv[3];

    double *buf_a = alloc_matrix(ISIZE, JSIZE);
    double *buf_b = alloc_matrix(ISIZE, JSIZE);
    double **a = wrap_2d(buf_a, ISIZE, JSIZE);
    double **b = wrap_2d(buf_b, ISIZE, JSIZE);

    init_matrix(a);

    double t0 = omp_get_wtime();
    if (use_omp)
        var3z_omp(a, b);
    else
        var3z_seq(a, b);
    double t1 = omp_get_wtime();

    printf("var3z_%s: time = %.6f s\n",
           use_omp ? "omp" : "seq", t1 - t0);

    write_matrix_txt(out_a, a, ISIZE, JSIZE);
    write_matrix_txt(out_b, b, ISIZE, JSIZE);

    free(a);
    free(b);
    free(buf_a);
    free(buf_b);
    return 0;
}
