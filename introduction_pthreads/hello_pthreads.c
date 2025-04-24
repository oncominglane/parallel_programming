#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 4

void* hello_world(void* arg) {
    int thread_id = *(int*)arg;
    printf("Hello World! I'm thread %d out of %d\n", thread_id, NUM_THREADS);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;  // вручную задаём номер потока
        pthread_create(&threads[i], NULL, hello_world, &thread_ids[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
