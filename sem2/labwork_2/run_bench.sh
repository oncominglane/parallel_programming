#!/usr/bin/env bash
# run_lab2_bench.sh
# Компилирует программы лабы 2, запускает их с разным числом исполнителей,
# пишет CSV для графиков и human-readable time.txt, проверяет корректность через diff.

set -euo pipefail

#############################
# Параметры стенда
#############################

ISIZE=${ISIZE:-2000}
JSIZE=${JSIZE:-2000}
REPEATS=${REPEATS:-3}

# Число потоков OpenMP и процессов MPI (можно переопределить перед запуском)
OMP_THREAD_LIST="${OMP_THREAD_LIST:-1 2 3 4 6 8 12}"
MPI_PROC_LIST="${MPI_PROC_LIST:-1 2 3 4 6}"

CC=${CC:-gcc}
MPICC=${MPICC:-mpicc}
CFLAGS="-O3 -std=c11 -Wall -Wextra -DISIZE=${ISIZE} -DJSIZE=${JSIZE}"

RESULTS_DIR="results"
BIN_DIR="bin"
CSV_PATH="${RESULTS_DIR}/lab2_results.csv"
TIME_LOG="${RESULTS_DIR}/time.txt"

mkdir -p "${RESULTS_DIR}" "${BIN_DIR}"

#############################
# Компиляция
#############################

echo "Compiling with ISIZE=${ISIZE}, JSIZE=${JSIZE} ..."

${CC} ${CFLAGS} etalon_seq_omp.c  -fopenmp -lm -o "${BIN_DIR}/etalon"
${CC} ${CFLAGS} var2zh_seq_omp.c  -fopenmp -lm -o "${BIN_DIR}/var2zh"
${CC} ${CFLAGS} var3z_seq_omp.c   -fopenmp -lm -o "${BIN_DIR}/var3z"
${CC} ${CFLAGS} var1e_seq.c       -fopenmp -lm -o "${BIN_DIR}/var1e_seq"
${MPICC} ${CFLAGS} var1e_mpi.c    -lm       -o "${BIN_DIR}/var1e_mpi"

echo "Compilation done."

#############################
# Вспомогательные функции
#############################

extract_time() {
    awk '/time =/ {
        for (i = 1; i <= NF; ++i) {
            if ($i ~ /^[0-9]+(\.[0-9]+)?$/) {
                print $i;
                exit
            }
        }
    }'
}

max_in_list() {
    tr ' ' '\n' | sort -n | tail -1
}

log_measurement() {
    # usage: log_measurement algo N p repeat time_sec
    local algo="$1" N="$2" p="$3" rep="$4" t="$5"
    echo "${algo},${N},${p},${rep},${t}" >> "${CSV_PATH}"
    printf "%-15s N=%-6s p=%-3s rep=%-2s time=%s s\n" "${algo}" "${N}" "${p}" "${rep}" "${t}" >> "${TIME_LOG}"
}

#############################
# Подготовка логов
#############################

echo "algo,N,p,repeat,time_sec" > "${CSV_PATH}"
echo "# Timing log (lab2)" > "${TIME_LOG}"
echo "# ISIZE=${ISIZE}, JSIZE=${JSIZE}, REPEATS=${REPEATS}" >> "${TIME_LOG}"
echo "# OMP_THREAD_LIST=${OMP_THREAD_LIST}, MPI_PROC_LIST=${MPI_PROC_LIST}" >> "${TIME_LOG}"
echo >> "${TIME_LOG}"

#############################
# Эталон (seq + omp)
#############################

echo "Running etalon (seq + omp) ..."

# seq
for rep in $(seq 1 "${REPEATS}"); do
    export OMP_NUM_THREADS=1
    output="$("${BIN_DIR}/etalon" seq 2>/dev/null)"
    time_sec=$(echo "${output}" | extract_time)
    log_measurement "etalon_seq" "${ISIZE}" 1 "${rep}" "${time_sec}"
done

# omp
for p in ${OMP_THREAD_LIST}; do
    for rep in $(seq 1 "${REPEATS}"); do
        export OMP_NUM_THREADS="${p}"
        output="$("${BIN_DIR}/etalon" omp 2>/dev/null)"
        time_sec=$(echo "${output}" | extract_time)
        log_measurement "etalon_omp" "${ISIZE}" "${p}" "${rep}" "${time_sec}"
    done
done

# Проверка корректности: seq vs omp
echo "Checking correctness: etalon seq vs omp ..."
MAX_OMP_THREADS=$(echo "${OMP_THREAD_LIST}" | max_in_list)

export OMP_NUM_THREADS=1
"${BIN_DIR}/etalon" seq "${RESULTS_DIR}/etalon_seq_check.txt" >/dev/null 2>&1

export OMP_NUM_THREADS="${MAX_OMP_THREADS}"
"${BIN_DIR}/etalon" omp "${RESULTS_DIR}/etalon_omp_check.txt" >/dev/null 2>&1

if ! diff -q "${RESULTS_DIR}/etalon_seq_check.txt" "${RESULTS_DIR}/etalon_omp_check.txt" >/dev/null; then
    echo "ERROR: mismatch in etalon (seq vs omp)!" >&2
    exit 1
else
    echo "OK: etalon seq and omp results are identical."
fi

#############################
# Вариант 2ж (seq + omp)
#############################

echo "Running var2zh (seq + omp) ..."

# seq
for rep in $(seq 1 "${REPEATS}"); do
    export OMP_NUM_THREADS=1
    output="$("${BIN_DIR}/var2zh" seq 2>/dev/null)"
    time_sec=$(echo "${output}" | extract_time)
    log_measurement "var2zh_seq" "${ISIZE}" 1 "${rep}" "${time_sec}"
done

# omp
for p in ${OMP_THREAD_LIST}; do
    for rep in $(seq 1 "${REPEATS}"); do
        export OMP_NUM_THREADS="${p}"
        output="$("${BIN_DIR}/var2zh" omp 2>/dev/null)"
        time_sec=$(echo "${output}" | extract_time)
        log_measurement "var2zh_omp" "${ISIZE}" "${p}" "${rep}" "${time_sec}"
    done
done

# Проверка корректности: seq vs omp
echo "Checking correctness: var2zh seq vs omp ..."
MAX_OMP_THREADS=$(echo "${OMP_THREAD_LIST}" | max_in_list)

export OMP_NUM_THREADS=1
"${BIN_DIR}/var2zh" seq "${RESULTS_DIR}/var2zh_seq_check.txt" >/dev/null 2>&1

export OMP_NUM_THREADS="${MAX_OMP_THREADS}"
"${BIN_DIR}/var2zh" omp "${RESULTS_DIR}/var2zh_omp_check.txt" >/dev/null 2>&1

if ! diff -q "${RESULTS_DIR}/var2zh_seq_check.txt" "${RESULTS_DIR}/var2zh_omp_check.txt" >/dev/null; then
    echo "ERROR: mismatch in var2zh (seq vs omp)!" >&2
    exit 1
else
    echo "OK: var2zh seq and omp results are identical."
fi

#############################
# Вариант 3з (seq + omp)
#############################

echo "Running var3z (seq + omp) ..."

# seq
for rep in $(seq 1 "${REPEATS}"); do
    export OMP_NUM_THREADS=1
    output="$("${BIN_DIR}/var3z" seq 2>/dev/null)"
    time_sec=$(echo "${output}" | extract_time)
    log_measurement "var3z_seq" "${ISIZE}" 1 "${rep}" "${time_sec}"
done

# omp
for p in ${OMP_THREAD_LIST}; do
    for rep in $(seq 1 "${REPEATS}"); do
        export OMP_NUM_THREADS="${p}"
        output="$("${BIN_DIR}/var3z" omp 2>/dev/null)"
        time_sec=$(echo "${output}" | extract_time)
        log_measurement "var3z_omp" "${ISIZE}" "${p}" "${rep}" "${time_sec}"
    done
done

# Проверка корректности: seq vs omp (оба массива)
echo "Checking correctness: var3z seq vs omp ..."
MAX_OMP_THREADS=$(echo "${OMP_THREAD_LIST}" | max_in_list)

export OMP_NUM_THREADS=1
"${BIN_DIR}/var3z" seq \
    "${RESULTS_DIR}/var3z_seq_a.txt" \
    "${RESULTS_DIR}/var3z_seq_b.txt" >/dev/null 2>&1

export OMP_NUM_THREADS="${MAX_OMP_THREADS}"
"${BIN_DIR}/var3z" omp \
    "${RESULTS_DIR}/var3z_omp_a.txt" \
    "${RESULTS_DIR}/var3z_omp_b.txt" >/dev/null 2>&1

if ! diff -q "${RESULTS_DIR}/var3z_seq_a.txt" "${RESULTS_DIR}/var3z_omp_a.txt" >/dev/null; then
    echo "ERROR: mismatch in var3z A (seq vs omp)!" >&2
    exit 1
fi

if ! diff -q "${RESULTS_DIR}/var3z_seq_b.txt" "${RESULTS_DIR}/var3z_omp_b.txt" >/dev/null; then
    echo "ERROR: mismatch in var3z B (seq vs omp)!" >&2
    exit 1
fi

echo "OK: var3z seq and omp results (A and B) are identical."

#############################
# Вариант 1е (seq)
#############################

echo "Running var1e_seq ..."

for rep in $(seq 1 "${REPEATS}"); do
    export OMP_NUM_THREADS=1
    output="$("${BIN_DIR}/var1e_seq" 2>/dev/null)"
    time_sec=$(echo "${output}" | extract_time)
    log_measurement "var1e_seq" "${ISIZE}" 1 "${rep}" "${time_sec}"
done

#############################
# Вариант 1е (MPI)
#############################

echo "Running var1e_mpi (MPI) ..."

for np in ${MPI_PROC_LIST}; do
    for rep in $(seq 1 "${REPEATS}"); do
        export OMP_NUM_THREADS=1
        output="$(mpirun -np "${np}" "${BIN_DIR}/var1e_mpi" 2>/dev/null)"
        time_sec=$(echo "${output}" | extract_time)
        log_measurement "var1e_mpi" "${ISIZE}" "${np}" "${rep}" "${time_sec}"
    done
done

# Проверка корректности: seq vs mpi (max np)
echo "Checking correctness: var1e seq vs mpi ..."
MAX_MPI_PROCS=$(echo "${MPI_PROC_LIST}" | max_in_list)

export OMP_NUM_THREADS=1
"${BIN_DIR}/var1e_seq" "${RESULTS_DIR}/var1e_seq_check.txt" >/dev/null 2>&1

export OMP_NUM_THREADS=1
mpirun -np "${MAX_MPI_PROCS}" "${BIN_DIR}/var1e_mpi" "${RESULTS_DIR}/var1e_mpi_check.txt" >/dev/null 2>&1

if ! diff -q "${RESULTS_DIR}/var1e_seq_check.txt" "${RESULTS_DIR}/var1e_mpi_check.txt" >/dev/null; then
    echo "ERROR: mismatch in var1e (seq vs mpi)!" >&2
    exit 1
else
    echo "OK: var1e seq and mpi results are identical."
fi

#############################
# Финал
#############################

echo "All done."
echo "CSV  : ${CSV_PATH}"
echo "TIMES: ${TIME_LOG}"
