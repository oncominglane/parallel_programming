/* var2zh_seq_omp.c */
#include "common.h"

void var2zh_seq(double **a) {
    for (int i = 0; i < ISIZE - 2; ++i)
        for (int j = 3; j < JSIZE; ++j)
            a[i][j] = sin(0.1 * a[i+2][j-3]);
}

void var2zh_omp(double **a) {
    // устраняем антизависимость: работаем из копии
    double *buf_old = alloc_matrix(ISIZE, JSIZE);
    double **a_old = wrap_2d(buf_old, ISIZE, JSIZE);

    for (int i = 0; i < ISIZE; ++i)
        for (int j = 0; j < JSIZE; ++j)
            a_old[i][j] = a[i][j];

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ISIZE - 2; ++i)
        for (int j = 3; j < JSIZE; ++j)
            a[i][j] = sin(0.1 * a_old[i+2][j-3]);

    free(a_old);
    free(buf_old);
}

int main(int argc, char **argv) {
    int use_omp = 0;
    const char *out_path = "results/result_2zh.txt";

    if (argc > 1 && strcmp(argv[1], "omp") == 0)
        use_omp = 1;
    if (argc > 2)
        out_path = argv[2];

    double *buf = alloc_matrix(ISIZE, JSIZE);
    double **a = wrap_2d(buf, ISIZE, JSIZE);

    init_matrix(a);

    double t0 = omp_get_wtime();
    if (use_omp)
        var2zh_omp(a);
    else
        var2zh_seq(a);
    double t1 = omp_get_wtime();

    printf("var2zh_%s: time = %.6f s\n",
           use_omp ? "omp" : "seq", t1 - t0);

    write_matrix_txt(out_path, a, ISIZE, JSIZE);

    free(a);
    free(buf);
    return 0;
}
