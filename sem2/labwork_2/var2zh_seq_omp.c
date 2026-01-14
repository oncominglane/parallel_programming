/* var2zh_seq_omp.c */
#include "common.h"

void var2zh_seq(double **a) {
    for (int i = 0; i < ISIZE - 2; ++i)
        for (int j = 3; j < JSIZE; ++j)
            a[i][j] = sin(0.1 * a[i+2][j-3]);
}

void var2zh_omp(double **a) {
    const int DEP_OFFSET = 2; // сдвиг по i: a[i+2][...]
    int max_threads = omp_get_max_threads();

    // halo_buffer[поток][строка_границы (0..DEP_OFFSET-1)][столбец]
    double (*halo_buffer)[DEP_OFFSET][JSIZE] =
        malloc(max_threads * sizeof(*halo_buffer));
    if (!halo_buffer) {
        fprintf(stderr, "Failed to allocate halo_buffer\n");
        exit(1);
    }

    #pragma omp parallel
    {
        int tid      = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        // Ручное разбиение строк по потокам
        int rows_per_thread = ISIZE / nthreads;
        int start_row       = tid * rows_per_thread;
        int end_row         = (tid == nthreads - 1)
                                ? ISIZE
                                : (tid + 1) * rows_per_thread;

        // Глобальный предел цикла по i: i < ISIZE - 2
        int loop_limit = ISIZE - DEP_OFFSET;
        int my_end     = end_row;
        if (my_end > loop_limit)
            my_end = loop_limit;

        // 1) Сохраняем первые DEP_OFFSET строк своего блока в halo_buffer
        for (int r = 0; r < DEP_OFFSET; ++r) {
            int global_row = start_row + r;
            if (global_row < ISIZE) {
                for (int j = 0; j < JSIZE; ++j) {
                    halo_buffer[tid][r][j] = a[global_row][j];
                }
            }
        }

        // Ждем, пока все потоки сохранят свои границы
        #pragma omp barrier

        // 2) Основной вычислительный цикл
        for (int i = start_row; i < my_end; ++i) {
            for (int j = 3; j < JSIZE; ++j) {
                int target_row = i + DEP_OFFSET; // строка, из которой читаем
                int src_col    = j - 3;
                double val_source;

                // Если нужная строка ушла в область следующего потока,
                // читаем из его буфера границ
                if (target_row >= end_row && tid < nthreads - 1) {
                    int offset = target_row - end_row; // 0 или 1
                    val_source = halo_buffer[tid + 1][offset][src_col];
                } else {
                    // Иначе строка ещё внутри моего блока
                    val_source = a[target_row][src_col];
                }

                a[i][j] = sin(0.1 * val_source);
            }
        }
    } // конец parallel

    free(halo_buffer);
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
