/* common.h */
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#ifndef ISIZE
#define ISIZE 2000
#endif

#ifndef JSIZE
#define JSIZE 2000
#endif

static inline double *alloc_matrix(int nrows, int ncols) {
    double *a = (double*)malloc((size_t)nrows * ncols * sizeof(double));
    if (!a) {
        fprintf(stderr, "Allocation failed (%d x %d)\n", nrows, ncols);
        exit(EXIT_FAILURE);
    }
    return a;
}

static inline double **wrap_2d(double *data, int nrows, int ncols) {
    double **ptrs = (double**)malloc(nrows * sizeof(double*));
    if (!ptrs) {
        fprintf(stderr, "Allocation failed for row pointers\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < nrows; ++i)
        ptrs[i] = data + (size_t)i * ncols;
    return ptrs;
}

static inline void init_matrix(double **a) {
    for (int i = 0; i < ISIZE; ++i)
        for (int j = 0; j < JSIZE; ++j)
            a[i][j] = 10.0 * i + j;
}

static inline void write_matrix_txt(const char *path, double **a,
                                    int nrows, int ncols)
{
    FILE *f = fopen(path, "w");
    if (!f) {
        perror("fopen");
        return;
    }
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j)
            fprintf(f, "%f ", a[i][j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

#endif /* COMMON_H */
