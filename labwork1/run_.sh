#!/bin/bash

echo "M,K,T_seq,T_par" > results.csv

for M in 100 500 1000 2000 4000 8000 # количество точек по x
do
  K=$M   # количество шагов по времени

  # Генерируем config.h
  echo "#ifndef CONFIG_H" > config.h
  echo "#define CONFIG_H" >> config.h
  echo "#define M $M" >> config.h
  echo "#define K $K" >> config.h
  echo "#endif" >> config.h

  echo "### M=$M, K=$K — компиляция..."
  gcc sequential_transfer_eq.c -o sequential_transfer_eq -lm
  mpicc parallel_transfer_eq.c -o parallel_transfer_eq -lm

  echo "### Последовательная версия..."
  T1=$(date +%s.%N)
  ./sequential_transfer_eq > /dev/null
  T2=$(date +%s.%N)
  T_SEQ=$(echo "$T2 - $T1" | bc)

  echo "### Параллельная версия..."
  T3=$(date +%s.%N)
  mpirun -np 4 ./parallel_transfer_eq > /dev/null
  T4=$(date +%s.%N)
  T_PAR=$(echo "$T4 - $T3" | bc)

  echo "$M,$K,$T_SEQ,$T_PAR" >> results.csv
done

echo "### Готово! Все данные в results.csv"
