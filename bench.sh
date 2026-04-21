#!/usr/bin/env bash
set -euo pipefail

# parse args
EXTRA_FLAGS=""
N_DEVICES=""
OUTFILE=""

for arg in "$@"; do
    case $arg in
        -r)
            echo "Building for release"
            EXTRA_FLAGS="-O2 -DNDEBUG"
            ;;
        n=*)
            N_DEVICES="${arg#*=}"
            ;;
        f=*)
            OUTFILE="${arg#*=}"
            ;;
    esac
done

if [[ -z "$N_DEVICES" ]]; then
    echo "Error: n=<N_DEVICES> is required"
    exit 1
fi

if [[ -z "$OUTFILE" ]]; then
    echo "Error: f=<OUTFILE> is required"
    exit 1
fi

conda activate $PSCRATCH/project

# TODO: add new impls here
nvcc -o benchmark \
    src/benchmark.cu src/utils.cu \
    src/nccl_ringreduce.cu src/naive_ringreduce.cu \
    src/pipelined_ringreduce_async.cu \
    src/pipelined_ringreduce_nccl.cu \
    -I$PSCRATCH/project/include \
    -L$PSCRATCH/project/lib \
    -lnccl -lpthread \
    $EXTRA_FLAGS

LD_LIBRARY_PATH=$PSCRATCH/project/lib NCCL_DEBUG=WARN ./benchmark $N_DEVICES $OUTFILE