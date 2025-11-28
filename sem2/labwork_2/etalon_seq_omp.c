/* etalon_seq_omp.c */
#include "common.h"

void etalon_seq(double **a) {
    for (int i = 0; i < ISIZE; ++i)
        for (int j = 0; j < JSIZE; ++j)
            a[i][j] = sin(2.0 * a[i][j]);
}

void etalon_omp(double **a) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ISIZE; ++i)
        for (int j = 0; j < JSIZE; ++j)
            a[i][j] = sin(2.0 * a[i][j]);
}

int main(int argc, char **argv) {
    int use_omp = 0;
    const char *out_path = "results/result_etalon.txt";

    if (argc > 1 && strcmp(argv[1], "omp") == 0)
        use_omp = 1;
    if (argc > 2)
        out_path = argv[2];

    double *buf = alloc_matrix(ISIZE, JSIZE);
    double **a = wrap_2d(buf, ISIZE, JSIZE);

    init_matrix(a);

    double t0 = omp_get_wtime();
    if (use_omp)
        etalon_omp(a);
    else
        etalon_seq(a);
    double t1 = omp_get_wtime();

    printf("etalon_%s: time = %.6f s\n",
           use_omp ? "omp" : "seq", t1 - t0);

    write_matrix_txt(out_path, a, ISIZE, JSIZE);

    free(a);
    free(buf);
    return 0;
}
