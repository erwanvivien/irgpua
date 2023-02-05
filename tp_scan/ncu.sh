#!/bin/sh

# Hack to find ncu
ncu_store=$(ls -1 '/nix/store' | grep 'nsight_compute' | grep -v 'drv')
ncu=/nix/store/${ncu_store}/nsight-compute/2022.1.1/ncu

BENCH_PATH=${1}

if [ -z "${BENCH_PATH}" ]; then
    echo "usage: $0 <output>"
    exit 1
fi

make -C build -j4

date=$(date +'%F-%Hh%Mm%Ss')

echo ${ncu} -o ${BENCH_PATH} -f --set full ./build/bench --bench-nsight
${ncu} \
  -o "${date}_${BENCH_PATH}" \
  --metrics regex:data_bank_conflicts_pipe_lsu_mem_shared_op \
  -f --set full \
  ./build/bench --bench-nsight
