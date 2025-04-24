#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_THREADS 4

int access_point = 0;
pthread_mutex_t mutex; //Мьютекс — механизм блокировки, который разрешает только одному потоку одновременно войти в критическую секцию

void* increment(void* arg) {
    int thread_id = *(int*)arg;

    // эмуляция задержки для имитации работы
    usleep(10000000); 

    pthread_mutex_lock(&mutex); // вход в критическую секцию
    access_point++;
    printf("Thread %d incremented access point to %d\n", thread_id, access_point);
    pthread_mutex_unlock(&mutex);// выход из критическую секцию

    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS]; //Массив идентификаторов потоков
    int thread_ids[NUM_THREADS]; //Массив для хранения уникального номера потока (0, 1, 2, ...)

    pthread_mutex_init(&mutex, NULL);

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, increment, &thread_ids[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&mutex);

    return 0;
}


/*
Программа запускает несколько потоков (NUM_THREADS), каждый из которых:

Получает доступ к общей переменной access_point

Инкрементирует её (прибавляет 1)

Выводит сообщение, в котором пишет свой номер и текущее значение переменной

Всё это происходит в безопасной (защищённой) секции с использованием mutex
*/
