# AsyncAllReduce

## How to run

```shell
cd $PSCRATCH/async-ring-allreduce/
./build.sh          # compile, optionally pass -r to build in release mode
sbatch ./run.sh     # run, modify node count before running
```

## Implementations

| Name | File | Description |
|------|------|-------------|
| Classic Ring | `src/naive_ringreduce.cu` | Flat ring RS+AG using ncclSend/ncclRecv |
| Pipelined Ring | `src/pipelined_ringreduce_nccl.cu` | 2-stream pipelined RS+AG to overlap compute/comm |
| Hierarchical Ring | `src/hier_ringreduce.cu` | PAARD-style 2-level: intra-node RS (NVLink) → cross-node all-reduce (Slingshot once) → intra-node AG (NVLink) |

## Synthetic inter-node penalty sweep

Since 2 nodes on Perlmutter are likely in the same Dragonfly group (local links only),
`GLOBAL_PENALTY_US` artificially inflates the inter-node crossing cost to model topologies
where the global/local link ratio is larger. The flat ring pays the penalty 14× per
all-reduce; the hierarchical algorithm pays it once.

```shell
for p in 0 10 100 1000; do
    GLOBAL_PENALTY_US=$p sbatch run.sh
done
```

Results land in `results/bench_<jobid>.csv`. Plot with `utils/plot.py`.

## Contributing

To add a new implementation, you will have to modify these files
- `src/your-impl.cu` containing the implementation, refer to `src/interface.h`
- `src/interface.h` containing the function signature for your implementation
- `src/benchmark.cu` with `impls` and `impl_names` updated accordingly
- `build.sh` to compile with the newly created `your-impl.cu`