/* var1e_seq.c */
#include "common.h"

void var1e_seq(double **a) {
    for (int i = 1; i < ISIZE; ++i)
        for (int j = 8; j < JSIZE; ++j)
            a[i][j] = sin(5.0 * a[i-1][j-8]);
}

int main(int argc, char **argv) {
    const char *out_path = "results/result_1e_seq.txt";
    if (argc > 1)
        out_path = argv[1];

    double *buf = alloc_matrix(ISIZE, JSIZE);
    double **a = wrap_2d(buf, ISIZE, JSIZE);

    init_matrix(a);

    double t0 = omp_get_wtime();
    var1e_seq(a);
    double t1 = omp_get_wtime();

    printf("var1e_seq: time = %.6f s\n", t1 - t0);

    write_matrix_txt(out_path, a, ISIZE, JSIZE);

    free(a);
    free(buf);
    return 0;
}
