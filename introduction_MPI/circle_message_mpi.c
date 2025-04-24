#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, msg;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        msg = 0;  // начальное значение
        printf("Rank %d: initial message = %d\n", rank, msg);
        MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(&msg, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank %d: received final message = %d\n", rank, msg);
    } else {
        MPI_Recv(&msg, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        msg += 1;
        printf("Rank %d: received and incremented to %d\n", rank, msg);
        int next = (rank + 1) % size;
        MPI_Send(&msg, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
