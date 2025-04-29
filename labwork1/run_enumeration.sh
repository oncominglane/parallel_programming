#!/bin/bash

echo "M,K,P,T_seq,T_par" > results.csv

for M in 100 500 1000 2000 4000 6000 8000 
do
  K=$M

  echo "### Генерация config.h для M=$M, K=$K..."
  echo "#ifndef CONFIG_H" > config.h
  echo "#define CONFIG_H" >> config.h
  echo "#define M $M" >> config.h
  echo "#define K $K" >> config.h
  echo "#endif" >> config.h

  # Компиляция
  echo "### Компиляция..."
  gcc sequential_transfer_eq.c -o sequential_transfer_eq -lm
  mpicc parallel_transfer_eq.c -o parallel_transfer_eq -lm

  echo "### Последовательная версия..."
  T1=$(date +%s.%N)
  ./sequential_transfer_eq > /dev/null
  T2=$(date +%s.%N)
  T_SEQ=$(echo "$T2 - $T1" | bc)

  # Прогоны с P = 1..4
  for P in 1 2 4 8 16
  do
    if (( M % P != 0 )); then
      echo "⚠️ M=$M не делится на P=$P — пропускаем"
      continue
    fi

    echo "### Параллельная версия: $P процессов..."
    T3=$(date +%s.%N)
    # mpirun -np $P ./parallel_transfer_eq > /dev/null
    mpirun --oversubscribe -np $P ./parallel_transfer_eq > /dev/null

    T4=$(date +%s.%N)
    T_PAR=$(echo "$T4 - $T3" | bc)

    echo "$M,$K,$P,$T_SEQ,$T_PAR" >> results.csv
  done
done

echo "### Готово! Данные сохранены в results.csv"

echo "### ЗПостроение графиков..."
python3 plot_speedup.py