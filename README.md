# AsyncAllReduce

## How to run

```shell
cd $PSCRATCH/async-ring-allreduce/
./build.sh       # compile, optionally pass -r to build in release mode
sbatch ./run.sh  # run, optionally pass -r to run in release mode, and -n=N_RANKS to run with N_RANKS ranks
```

## Synthetic inter-node penalty (sweeps)

Cross-group `ncclSend`/`ncclRecv` pairs (see `ncclSendRecv` in `src/utils.cu`) can add an in-stream delay modelling a slow wide-area or global link:

- `GLOBAL_PENALTY_US` — fixed latency α in microseconds (default `0`).
- `GLOBAL_BW_GBPS` — inverse bandwidth β: extra ns per byte as `1 / rate` when rate is in GB/s (default `0` = no β term).

Both read once per process. Example:

```shell
GLOBAL_PENALTY_US=100 GLOBAL_BW_GBPS=5 sbatch run.sh -r
```

## Contributing

To add a new implementation, you will have to modify these files
- `src/your-impl.cu` containing the implementation, refer to `src/interface.h`
- `src/interface.h` containing the function signature for your implementation
- `src/benchmark.cu` with `impls` and `impl_names` updated accordingly
- `bench.sh` to compile with the newly created `your-impl.cu`