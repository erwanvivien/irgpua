#!/bin/sh

# Hack to find ncu
ncu_store=$(ls -1 '/nix/store' | grep 'nsight_compute' | grep -v 'drv')
ncu=/nix/store/${ncu_store}/nsight-compute/2022.1.1/ncu

OUTPUT_PATH=${1}
BENCH_PATH=${2:-./build/kernel}

if [ -z "${OUTPUT_PATH}" ]; then
    echo "usage: $0 <output> [binary]"
    exit 1
fi

if ! [ -d "build" ]; then
    cmake -B build
fi
make -C build -j4

date=$(date +'%F-%Hh%Mm%Ss')

echo "${ncu}" -o "${OUTPUT_PATH}" -f --set full "${BENCH_PATH}" --bench-nsight
"${ncu}" \
  -o "${date}_${OUTPUT_PATH}" \
  -f --set full \
  "${BENCH_PATH}" --bench-nsight

  # --metrics regex:data_bank_conflicts_pipe_lsu_mem_shared_op \
