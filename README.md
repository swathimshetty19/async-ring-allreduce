# AsyncAllReduce

## How to run

```shell
cd $PSCRATCH/async-ring-allreduce/
./build.sh       # compile, optionally pass -r to build in release mode
sbatch ./run.sh  # run, optionally pass -r to run in release mode, and -n=N_RANKS to run with N_RANKS ranks
```

## Contributing

To add a new implementation, you will have to modify these files
- `src/your-impl.cu` containing the implementation, refer to `src/interface.h`
- `src/interface.h` containing the function signature for your implementation
- `src/benchmark.cu` with `impls` and `impl_names` updated accordingly
- `bench.sh` to compile with the newly created `your-impl.cu`