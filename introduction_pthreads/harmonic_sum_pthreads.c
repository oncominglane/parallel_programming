#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS 4

typedef struct {
    int thread_id;
    int N;
    int num_threads;
    double partial_sum;
} ThreadData;

void* compute_partial_sum(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    int id = data->thread_id;
    int N = data->N;
    int num_threads = data->num_threads;

    double sum = 0.0;

    for (int i = id + 1; i <= N; i += num_threads) {
        sum += 1.0 / i;
    }

    data->partial_sum = sum;
    return NULL;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s N\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    pthread_t threads[NUM_THREADS];
    ThreadData data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        data[i].thread_id = i;
        data[i].N = N;
        data[i].num_threads = NUM_THREADS;
        pthread_create(&threads[i], NULL, compute_partial_sum, &data[i]);
    }

    double total_sum = 0.0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        total_sum += data[i].partial_sum;
    }

    printf("Partial harmonic sum for N = %d is: %.12f\n", N, total_sum);
    return 0;
}

//Как работает:
//Потоки делят работу по принципу round-robin:  поток 0 берёт 1, 5, 9, ..., поток 1 — 2, 6, 10, и т.д.
//Каждый считает свою часть и сохраняет её в partial_sum
//Главный поток ждёт все pthread_join и складывает результат