#!/bin/bash
set -e  # останавливаемся при первой ошибке

# --------- набор N ---------
if [[ "$#" -gt 0 ]]; then
    NLIST=("$@")
else
    NLIST=(400 1000 4000 8000 16000 32000 64000)
fi

# сколько раз повторять каждый запуск для усреднения
REPEAT=${REPEAT:-50}

echo "Grid points list: ${NLIST[*]}"
echo "Repeats per configuration: $REPEAT"

# --------- каталоги ---------
mkdir -p bin
mkdir -p results

# --------- CSV для времён ---------
CSV="results/bvp_results.csv"
# перезаписываем файл каждый запуск
echo "algo,N,p,repeat,time_sec" > "$CSV"

# --------- компиляция ---------
echo "Compiling..."
gcc numerov_bvp_seq.c            -O2 -lm          -o bin/numerov_seq
gcc numerov_bvp_seq_reduction.c  -O2 -lm          -o bin/numerov_red
gcc numerov_bvp_omp_reduction.c  -O2 -lm -fopenmp -o bin/numerov_omp_red

# --------- функция запуска с измерением времени (date +%s%N) ---------
# ВАЖНО: порядок аргументов:
#   run_and_time exe label outfile algo_name p N extra_args...
#
# В CSV идёт строка:
#   algo_name,N,p,REPEAT,avg_time
#
run_and_time() {
    local exe="$1"       # имя бинарника в bin/
    local label="$2"     # строка для логов
    local outfile="$3"   # имя txt-файла в results/
    local algo_name="$4" # bvp_seq / bvp_red_seq / bvp_red_omp
    local p="$5"         # число потоков
    local N="$6"         # число точек
    shift 6
    local args=("$@")    # оставшиеся аргументы программы

    local outpath="results/$outfile"

    echo "Running $label (N=$N, p=$p, repeats=$REPEAT)..."

    local start_ns end_ns dt_ns
    start_ns=$(date +%s%N)

    for ((i=0; i<REPEAT; i++)); do
        if [[ $i -eq 0 ]]; then
            "bin/$exe" "${args[@]}" > "$outpath"
        else
            "bin/$exe" "${args[@]}" > /dev/null
        fi
    done

    end_ns=$(date +%s%N)
    dt_ns=$((end_ns - start_ns))

    # среднее время одного запуска в секундах
    local avg_time
    avg_time=$(LC_ALL=C LC_NUMERIC=C awk -v ns="$dt_ns" -v rep="$REPEAT" 'BEGIN {printf "%.9f", ns/1e9/rep}')

    echo "  $label avg time = ${avg_time} s (output: $outpath)"

    # пример строки: bvp_seq,400,1,50,0.000123456
    echo "${algo_name},${N},${p},${REPEAT},${avg_time}" >> "$CSV"
}

# --------- запускаем все варианты для каждого N ---------
for NPOINTS in "${NLIST[@]}"; do
    echo
    echo "===== N = $NPOINTS ====="

    # 1) последовательная с прогонкой (Thomas)
    run_and_time numerov_seq "seq + Thomas" \
        "result_seq_N${NPOINTS}.txt" \
        bvp_seq 1 "$NPOINTS" \
        "$NPOINTS"

    # 2) последовательная с редукцией
    #run_and_time numerov_red "seq + reduction" \
    #    "result_red_N${NPOINTS}.txt" \
    #    bvp_red_seq 1 "$NPOINTS" \
    #    "$NPOINTS"

    # 3) параллельная редукция с 4 потоками
    run_and_time numerov_omp_red "OMP reduction, 4 threads" \
        "result_omp4_N${NPOINTS}.txt" \
        bvp_red_omp 4 "$NPOINTS" \
        "$NPOINTS" 4
done

echo
echo "All runs finished. CSV collected in $CSV"

# --------- построение графиков ---------
echo "Plotting time, speedup and efficiency vs N from $CSV ..."

python3 plot_time_vs_N.py "$CSV" \
    "results/time_vs_N.png" \
    "results/speedup_vs_N.png" \
    "results/efficiency_vs_N.png" \
    || echo "Plotting failed (no python/matplotlib/pandas?)"

echo
echo "Done."
