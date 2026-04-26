// utils.cu

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "interface.h"

__global__ void init_input_kernel(float* buf, int rank, long input_size) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size) buf[idx] = 100.0f * rank + idx * 100.0f / input_size;
}

__global__ void add_kernel(float* dest, const float* src, long n) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    __nanosleep(5000);  // simulate "more work"
    if (idx < n) dest[idx] += src[idx];
}

// in-stream delay on the same stream as the nccl op; preserves async overlap
// (unlike host-side sleep). __nanosleep caps at ~1 ms per call on sm_70+.
__global__ void internode_delay_kernel(long total_ns) {
    const unsigned int chunk_ns = 100000;  // 100 us
    long remaining = total_ns;
    while (remaining > 0) {
        unsigned int s = remaining > chunk_ns ? chunk_ns : (unsigned int)remaining;
        __nanosleep(s);
        remaining -= s;
    }
}

// LogGP-style affine inter-node cost after a cross-group nccl send/recv:
//   delay(bytes) = GLOBAL_PENALTY_US (µs)  +  bytes / GLOBAL_BW_GBPS (ns/byte for GB/s)
// env read once. no-op if both are unset/zero. bytes = sizeof(float) * element count.
static void maybe_penalize_after_internode(cudaStream_t stream, long bytes) {
    static long penalty_us = -1;           // GLOBAL_PENALTY_US: fixed (alpha)
    static double inv_bw_ns_per_byte = -1; // 1 / GLOBAL_BW_GBPS
    if (penalty_us < 0) {
        const char* p = getenv("GLOBAL_PENALTY_US");
        penalty_us = (p && p[0] != '\0') ? atol(p) : 0;
        const char* bw = getenv("GLOBAL_BW_GBPS");
        double gbps = (bw && bw[0] != '\0') ? atof(bw) : 0.0;
        inv_bw_ns_per_byte = (gbps > 0.0) ? (1.0 / gbps) : 0.0;
    }
    long total_ns = penalty_us * 1000L + (long)(inv_bw_ns_per_byte * (double)bytes);
    if (total_ns <= 0) return;
    internode_delay_kernel<<<1, 1, 0, stream>>>(total_ns);
}

void ncclSendRecv(
    float* send_buf,
    float* recv_buf,
    size_t buf_sz,
    int rank,
    int send_rank,
    int recv_rank,
    ncclComm_t comm,
    cudaStream_t stream
) {
    NCCL_CALL(ncclGroupStart());
    NCCL_CALL(ncclSend(send_buf, buf_sz, ncclFloat, send_rank, comm, stream));
    NCCL_CALL(ncclRecv(recv_buf, buf_sz, ncclFloat, recv_rank, comm, stream));
    NCCL_CALL(ncclGroupEnd());

    static constexpr int group_size = 2;
    const int group = rank / group_size;
    const int send_group = send_rank / group_size;
    const int recv_group = recv_rank / group_size;

    if (group != send_group || group != recv_group) {
        long bytes = (long)buf_sz * (long)sizeof(float);
        maybe_penalize_after_internode(stream, bytes);
    }
}

bool check_correctness(float* h_res, int rank, int n_ranks, long input_size, float atol) {
    int sum_ranks = n_ranks * (n_ranks - 1) * 50;

    for (long i = 0; i < input_size; i++) {
        float expected = (float)sum_ranks + (float)n_ranks * 100.0f * i / input_size;
        float got = h_res[i];
        float diff = fabsf(got - expected);
        if (diff > atol) {
            fprintf(
                stderr,
                "Rank %d: verification FAILED, mismatch at idx %d: got %f expected %f (diff %f)\n",
                rank,
                i,
                got,
                expected,
                diff
            );
            return false;
        }
    }
    return true;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1e6 + (double)tv.tv_usec;
}

void analyze_runtime(RunArgs* args, double* deltas) {
    const int n_iters = args->n_iters;

    double sum_latency = 0.0;
    double min_latency = deltas[0];
    double max_latency = deltas[0];
    for (int i = 0; i < n_iters; i++) {
        double t = deltas[i];
        sum_latency += t;
        if (t < min_latency) min_latency = t;
        if (t > max_latency) max_latency = t;
    }

    double avg_latency = sum_latency / n_iters;
    double sum_std = 0.0;
    for (int i = 0; i < n_iters; i++)
        sum_std += (deltas[i] - avg_latency) * (deltas[i] - avg_latency);

    *(args->avg_latency) = avg_latency;
    *(args->std_latency) = sqrt(sum_std / n_iters);
    *(args->min_latency) = min_latency;
    *(args->max_latency) = max_latency;
}
