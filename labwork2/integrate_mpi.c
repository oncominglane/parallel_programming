#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Функция f(x) = sin(1 / x^2)
double f(double x) {
    return sin(1.0 / (x * x));
}

// Метод трапеций для отрезка [a, b] с шагом h
double integrate(double a, double b, double epsilon) {
    int n = (int)((b - a) / sqrt(epsilon)); // приближённое количество разбиений
    if (n < 1) n = 1;
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        sum += f(x);
    }
    return sum * h;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    double a = 0.01, b = 1.0;
    double epsilon;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Проверка аргумента
    if (argc != 2) {
        if (rank == 0)
            printf("Usage: %s epsilon\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    epsilon = atof(argv[1]);

    // Деление интервала по процессам
    double delta = (b - a) / size;
    double local_a = a + rank * delta;
    double local_b = local_a + delta;

    // Локальное интегрирование
    double local_integral = integrate(local_a, local_b, epsilon);

    // Сбор всех частей в корневой процесс
    double total_integral = 0.0;
    MPI_Reduce(&local_integral, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Integral from %.5f to %.5f = %.12f\n", a, b, total_integral);
    }

    MPI_Finalize();
    return 0;
}
