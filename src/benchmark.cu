// benchmark.cu
// Calls the various ring all reduce implementations with various buffer sizes and measures the
// latency while checking for correctness

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "interface.h"


// TODO: add new implementations here
static RingRunFunc impls[] = {
    ring_pipelined_nccl,
    ring_naive,
    ring_hierarchical,
};

static const char* impl_names[] = {
    "Pipelined Ring",
    "Classic Ring",
    "Hierarchical Ring",
};



// Usage: ./benchmark
int main(int argc, char** argv) {
    // initialize MPI
    MPI_Init(&argc, &argv);
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    if (n_ranks % 2 != 0) {
        if (rank == 0) {
            printf("the number of ranks must be even, got %d\n", n_ranks);
            fflush(stdout);
        }
        MPI_Finalize();
        return 1;
    }

    // determine node membership: ranks on the same node share memory
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank, local_size;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);
    int node_id = rank / local_size;  // assumes contiguous rank assignment per node

    // build rank_to_node table so impls can identify cross-node peers
    int* rank_to_node = (int*)malloc(n_ranks * sizeof(int));
    MPI_Allgather(&node_id, 1, MPI_INT, rank_to_node, 1, MPI_INT, MPI_COMM_WORLD);

    // set up GPU for this process using MPI-derived local_rank
    CUDA_CALL(cudaSetDevice(local_rank));

    // get NCCL Unique ID from rank 0
    ncclUniqueId id;
    if (rank == 0) NCCL_CALL(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // initialize NCCL communicator
    ncclComm_t comm;
    NCCL_CALL(ncclCommInitRank(&comm, n_ranks, id, rank));

    if (rank == 0) {
        printf(
            "impl,input_size,input_bytes,avg_latency,std_latency,min_latency,max_latency,"
            "throughput\n"
        );
        fflush(stdout);
    }

    const int n_warmup = 200;
    const int n_iters = 200;
    const float atol = 1e-3f;
    const long min_sz = 256;         // 1KB
    const long max_sz = 2147483648;  // 8GB

    const int n_impl = sizeof(impls) / sizeof(impls[0]);
    for (int i = 0; i < n_impl; i++) {
        for (long input_size = min_sz; input_size <= max_sz; input_size *= 2) {
            size_t n_bytes = (size_t)input_size * sizeof(float);

            double local_avg = 1.0;
            double local_std = 0.0;
            double local_min = 0.0;
            double local_max = 0.0;
            bool local_correct = 0;

            RunArgs args;
            args.input_size = input_size;
            args.comm = comm;
            args.local_rank = local_rank;
            args.local_size = local_size;
            args.node_id = node_id;
            args.rank_to_node = rank_to_node;
            args.n_warmup = n_warmup;
            args.n_iters = n_iters;
            args.atol = atol;
            args.correct = &local_correct;
            args.avg_latency = &local_avg;
            args.std_latency = &local_std;
            args.min_latency = &local_min;
            args.max_latency = &local_max;

            // run the impl
            impls[i](&args);

            // run correctness check
            int local_correct_int = int(local_correct);
            int global_correct = 0;
            MPI_Allreduce(&local_correct_int, &global_correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (global_correct != 1) {
                if (rank == 0) {
                    printf("FAILED, stopping\n");
                    fflush(stdout);
                }
                break;
            }

            // get global metrics (max to get slowest out of all ranks)
            double global_avg;
            double global_std;
            double global_min;
            double global_max;
            MPI_Reduce(&local_avg, &global_avg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_std, &global_std, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                double throughput = n_bytes / global_avg;
                printf(
                    "%s,%lu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                    impl_names[i],
                    input_size,
                    n_bytes,
                    global_avg,
                    global_std,
                    global_min,
                    global_max,
                    throughput
                );
                fflush(stdout);
            }
        }
    }

    // cleanup
    free(rank_to_node);
    MPI_Comm_free(&node_comm);
    NCCL_CALL(ncclCommDestroy(comm));
    MPI_Finalize();

    return 0;
}
