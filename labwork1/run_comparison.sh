#!/bin/bash

echo "### Компиляция..."
gcc sequential_transfer_eq.c -o sequential_transfer_eq -lm
mpicc parallel_transfer_eq.c -o parallel_transfer_eq -lm

echo "### Запуск последовательной версии..."
START_SEQ=$(date +%s.%N)
./sequential_transfer_eq
END_SEQ=$(date +%s.%N)
TIME_SEQ=$(echo "$END_SEQ - $START_SEQ" | bc)

echo "### Запуск параллельной версии (4 процесса)..."
START_PAR=$(date +%s.%N)
mpirun -np 4 ./parallel_transfer_eq
END_PAR=$(date +%s.%N)
TIME_PAR=$(echo "$END_PAR - $START_PAR" | bc)

echo "### Последовательная: $TIME_SEQ сек"
echo "### Параллельная: $TIME_PAR сек"

# Запись во временный файл для Python
echo "$TIME_SEQ $TIME_PAR" > timings.txt

echo "### Запуск визуализации..."
python3 3D_graph.py
