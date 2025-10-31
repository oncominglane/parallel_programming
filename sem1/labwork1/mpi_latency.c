#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank;
    int msg = 42;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double round_trip_time_sum = 0.0;

    for (int i = 0; i < 10; i++) {
        if (rank == 0) {
            double start_time = MPI_Wtime();
            MPI_Send(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&msg, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double end_time = MPI_Wtime();
            round_trip_time_sum += (end_time - start_time);
        } else if (rank == 1) {
            MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        double average_round_trip_time = round_trip_time_sum / 10.0;
        printf("Ping-pong latency (round trip) = %.9f seconds\n", average_round_trip_time);
        printf("Estimated one-way latency      = %.9f seconds\n", average_round_trip_time / 2.0);
    }

    MPI_Finalize();
    return 0;
}
