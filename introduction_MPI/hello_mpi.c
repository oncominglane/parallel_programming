#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int commsize, my_rank;
    MPI_Init(&argc, &argv); //инициализация библиотеки
    MPI_Comm_size(MPI_COMM_WORLD, &commsize); //возвращает общее количество процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); //даёт номер текущего процесса
    printf("Hello World! I'm process %d out of %d\n", my_rank, commsize);
    MPI_Finalize();
    return 0;
}
