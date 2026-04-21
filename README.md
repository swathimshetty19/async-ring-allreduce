# AsyncAllReduce

## How to run

First-time:

```shell
conda create --prefix $PSCRATCH/project -c nvidia nccl
```

Every-time:

```shell
# start up GPUs & navigate to directory
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account m4341_g
cd $PSCRATCH/async-ring-allreduce/

# specify number of ranks, outputfile, and optionally pass -r to compile for release
./bench.sh n=4 f=output.csv
```

## Contributing

To add a new implementation, you will have to modify these files
- `src/your-impl.cu` containing the implementation, refer to `src/interface.h`
- `src/interface.h` containing the function signature for your implementation
- `src/benchmark.cu` with `impls` and `impl_names` updated accordingly
- `bench.sh` to compile with the newly created `your-impl.cu`