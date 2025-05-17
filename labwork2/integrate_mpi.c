#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Заданная функция
double f(double x) {
    return sin(1.0 / (x * x));
}

// Метод трапеций для отрезка [a, b] с n разбиениями
double integrate_trapezoid(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    return sum * h;
}

// Интегрирование с автоматическим подбором n до достижения epsilon
double integrate_auto(double a, double b, double epsilon) {
    int n = 10;
    double prev = integrate_trapezoid(a, b, n);
    while (1) {
        n *= 2;
        double current = integrate_trapezoid(a, b, n);
        if (fabs(current - prev) < epsilon) {
            return current;
        }
        prev = current;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    double a = 0.01, b = 1.0;
    double epsilon;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0)
            printf("Usage: %s epsilon\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    epsilon = atof(argv[1]);
    double delta = (b - a) / size;
    double local_a = a + rank * delta;
    double local_b = local_a + delta;

    double local_epsilon = epsilon / size;

    double local_integral = integrate_auto(local_a, local_b, local_epsilon);

    double total_integral = 0.0;
    MPI_Reduce(&local_integral, &total_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Integral from %.5f to %.5f = %.12f (ε = %.2e)\n", a, b, total_integral, epsilon);
    }

    MPI_Finalize();
    return 0;
}
