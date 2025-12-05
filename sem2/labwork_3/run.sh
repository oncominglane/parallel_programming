#!/bin/sh
set -e  # если любая команда упадёт – скрипт сразу завершится с ошибкой

# число узлов на отрезке [0,1]
# можно передать первым аргументом, иначе по умолчанию 1001
NPOINTS=${1:-1001}

echo "Compiling..."
gcc numerov_bvp_seq.c -O2 -lm -o numerov_seq
gcc numerov_bvp_seq_reduction.c -O2 -lm -o numerov_red

echo "Running sequential (Thomas) version..."
./numerov_seq "$NPOINTS" > result_seq.txt

echo "Running reduction version..."
./numerov_red "$NPOINTS" > result_red.txt

echo "Comparing outputs (max |Δy|)..."

# Склеиваем файлы построчно и сравниваем второй и четвёртый столбцы (y_seq и y_red)
max_diff=$(
  paste result_seq.txt result_red.txt | \
  awk '
    $1 == "#" { next }       # пропускаем строки-комментарии
    NF < 4     { next }       # защита от пустых строк
    {
      y1 = $2;   # y из первого файла
      y2 = $4;   # y из второго файла
      d  = y1 - y2;
      if (d < 0) d = -d;
      if (d > max) max = d;
    }
    END {
      if (NR > 0) printf "%.12g\n", max; else print "NaN";
    }
  '
)

echo "max |Δy| = $max_diff"

# Порог, который считаем "совпадением" (можно подправить при необходимости)
TOL=1e-10

# численное сравнение
awk -v diff="$max_diff" -v tol="$TOL" '
  BEGIN {
    if (diff != diff) {  # NaN-проверка
      print "ERROR: no data to compare";
      exit 1;
    }
    if (diff <= tol) {
      print "OK: solutions match within tolerance";
      exit 0;
    } else {
      print "WARNING: solutions differ more than tolerance";
      exit 1;
    }
  }
'
