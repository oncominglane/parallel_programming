/* var1e_mpi.c */
#include <mpi.h>
#include "common.h"

int main(int argc, char **argv) {
    const char *out_path = "results/result_1e_mpi.txt";
    if (argc > 1)
        out_path = argv[1];

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Каждый процесс хранит весь массив (проще и достаточно для лабы)
    double *buf = alloc_matrix(ISIZE, JSIZE);
    double **a = wrap_2d(buf, ISIZE, JSIZE);

    if (rank == 0)
        init_matrix(a);

    // Рассылаем исходную матрицу всем
    MPI_Bcast(buf, ISIZE * JSIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Делим диапазон строк [1, ISIZE) по процессам (строка 0 остаётся общей "сверху")
    int total_rows = ISIZE - 1;      // строки 1..ISIZE-1 включительно
    int base = total_rows / size;
    int rem  = total_rows % size;

    int my_rows = base + (rank < rem ? 1 : 0);
    int offset  = 1; // начинаем с i = 1
    for (int r = 0; r < rank; ++r)
        offset += base + (r < rem ? 1 : 0);

    int i_start = offset;
    int i_end   = offset + my_rows;  // не включительно

    double t0 = MPI_Wtime();

    // Эстафета по блокам строк:
    // - rank 0 сразу считает свои строки, опираясь на строку 0.
    // - rank > 0 сначала ждёт готовую строку (i_start-1) от предыдущего rank.
    if (rank > 0 && my_rows > 0) {
        int src = rank - 1;
        int row_above = i_start - 1; // эту строку посчитал предыдущий процесс
        MPI_Recv(a[row_above], JSIZE, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Считаем свои строки последовательно, как в var1e_seq
    for (int i = i_start; i < i_end; ++i) {
        for (int j = 8; j < JSIZE; ++j) {
            a[i][j] = sin(5.0 * a[i-1][j-8]);
        }
    }

    // Если это не последний процесс и у нас есть строки —
    // отправляем свою последнюю строку вниз по цепочке
    if (rank < size - 1 && my_rows > 0) {
        int dst = rank + 1;
        int last_row = i_end - 1;
        MPI_Send(a[last_row], JSIZE, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();

    // Собираем результат на rank 0: все строки 1..ISIZE-1
    // через MPI_Gatherv (каждый шлёт только свой блок строк).
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        recvcounts = (int*)malloc(size * sizeof(int));
        displs     = (int*)malloc(size * sizeof(int));

        int off_rows = 1; // начинаем заполнять с строки 1
        for (int r = 0; r < size; ++r) {
            int rows_r = base + (r < rem ? 1 : 0);
            recvcounts[r] = rows_r * JSIZE;
            displs[r]     = off_rows * JSIZE;
            off_rows     += rows_r;
        }
    }

    // Локальный буфер отправки: наш блок строк
    // Важно: даже если my_rows == 0, count=0 и указатель не трогают.
    MPI_Gatherv(
        my_rows > 0 ? a[i_start] : buf,    // отправляем с начала своего блока
        my_rows * JSIZE,
        MPI_DOUBLE,
        buf,
        recvcounts,
        displs,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("var1e_mpi: time = %.6f s (size=%d)\n", t1 - t0, size);
        write_matrix_txt(out_path, a, ISIZE, JSIZE);
    }

    if (recvcounts) free(recvcounts);
    if (displs) free(displs);

    free(a);
    free(buf);

    MPI_Finalize();
    return 0;
}
