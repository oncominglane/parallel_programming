/* var1e_mpi.c */

#include <mpi.h>
#include <math.h>
#include "common.h"

int main(int argc, char **argv) {
    const char *out_path = "results/result_1e_mpi.txt";
    if (argc > 1)
        out_path = argv[1];

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Используем зависимость j-8 => размер коммуникатора должен делить 8
    if (8 % size != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "var1e_mpi: this version requires MPI size dividing 8 "
                    "(1,2,4,8), got %d\n", size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Каждый процесс хранит весь массив (проще и достаточно для лабы)
    double *buf = alloc_matrix(ISIZE, JSIZE);
    double **a = wrap_2d(buf, ISIZE, JSIZE);

    if (rank == 0)
        init_matrix(a);

    // Рассылаем исходную матрицу всем
    MPI_Bcast(buf, ISIZE * JSIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Считаем, сколько столбцов принадлежит этому рангу (j % size == rank, j >= 8)
    int my_cols = 0;
    for (int j = 8 + rank; j < JSIZE; j += size) {
        ++my_cols;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Параллельная работа по столбцам
    // Каждый ранг считает только свои столбцы j = 8+rank, 8+rank+size, ...
    // При этом j-8 = rank + size*k, т.е. тот же ранг (так как 8 % size == 0).
    for (int i = 1; i < ISIZE; ++i) {
        for (int j = 8 + rank; j < JSIZE; j += size) {
            a[i][j] = sin(5.0 * a[i-1][j-8]);
        }
    }

    double t1 = MPI_Wtime();

    // Теперь собираем результат на rank 0.
    // rank 0 уже содержит "свои" столбцы (j = 8, 8+size, ...),
    // остальные ранги пришлют ему свои столбцы в виде упакованного буфера.

    if (rank == 0) {
        // Принимаем данные от всех остальных рангов
        for (int src = 1; src < size; ++src) {
            // Считаем, сколько столбцов у этого src
            int cols_src = 0;
            for (int j = 8 + src; j < JSIZE; j += size) {
                ++cols_src;
            }
            if (cols_src == 0)
                continue;

            double *recvbuf = (double*)malloc((size_t)ISIZE * cols_src * sizeof(double));
            if (!recvbuf) {
                fprintf(stderr, "var1e_mpi: recvbuf alloc failed\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            MPI_Recv(recvbuf, ISIZE * cols_src, MPI_DOUBLE,
                     src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Разворачиваем буфер обратно в матрицу a на ранге 0.
            // Порядок упакован: по строкам i, внутри строки — по локальным столбцам.
            for (int i = 0; i < ISIZE; ++i) {
                int idx_row = i * cols_src;
                int k = 0;
                for (int j = 8 + src; j < JSIZE; j += size, ++k) {
                    a[i][j] = recvbuf[idx_row + k];
                }
            }

            free(recvbuf);
        }

        printf("var1e_mpi: time = %.6f s (size=%d)\n", t1 - t0, size);
        write_matrix_txt(out_path, a, ISIZE, JSIZE);
    } else {
        // Ранги > 0 упаковывают свои столбцы и отправляют их на rank 0
        if (my_cols > 0) {
            double *sendbuf = (double*)malloc((size_t)ISIZE * my_cols * sizeof(double));
            if (!sendbuf) {
                fprintf(stderr, "var1e_mpi: sendbuf alloc failed (rank %d)\n", rank);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            int idx = 0;
            for (int i = 0; i < ISIZE; ++i) {
                for (int j = 8 + rank; j < JSIZE; j += size) {
                    sendbuf[idx++] = a[i][j];
                }
            }

            MPI_Send(sendbuf, ISIZE * my_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            free(sendbuf);
        }
    }

    free(a);
    free(buf);

    MPI_Finalize();
    return 0;
}
