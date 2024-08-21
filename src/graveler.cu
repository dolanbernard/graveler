#include "stdio.h"
#include "math.h"
#include "stdint.h"

#include <curand.h>
#include <curand_kernel.h>

#include "timers.h"

#define N (1000000000)

#define THREADS_PER_BLOCK (512)
#define BLOCKS_PER_GRID (131072)


__global__
void graveler(size_t n, uint32_t *one_counts, curandState *rand_states, uint64_t seed) {
  int i;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  curandState *state = rand_states + tid;
  curand_init(seed + tid, tid, 0, state);

  for(i = tid; i < N; i += nthreads) {
    uint32_t one_count = 0;
    for(size_t rolls = 0; rolls < 231; rolls++) {
      uint32_t roll = (uint32_t)(curand_uniform(state) * 3.0 + 0.5f);
      if(roll == 0) {
        ++one_count;
      }
    }
    one_counts[i] = one_count;
  }
}

int main(void) {
  uint32_t *h_one_counts, *d_one_counts;
  curandState *d_rand_states;
  cudaMalloc(&d_rand_states, sizeof(curandState) * THREADS_PER_BLOCK * BLOCKS_PER_GRID);
  cudaMalloc(&d_one_counts, N * sizeof(uint32_t));
  h_one_counts = (uint32_t *)malloc(N * sizeof(uint32_t));

  DECLARE_TIMER(GravelerKernelTimer);
  START_TIMER(GravelerKernelTimer);
  graveler<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(N, d_one_counts, d_rand_states, time(0));
  cudaDeviceSynchronize();
  STOP_TIMER(GravelerKernelTimer);

  cudaMemcpy(h_one_counts, d_one_counts, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  uint32_t max_ones = 0;
  for(size_t i = 0; i < N; ++i) {
    if(h_one_counts[i] > max_ones) {
      max_ones = h_one_counts[i];
    }
  }
  fprintf(stdout, "Max ones rolled: %u\n", max_ones);
  PRINT_TIMER(GravelerKernelTimer);

  cudaFree(d_one_counts);
  cudaFree(d_rand_states);
  free(h_one_counts);
  
  return 0;
}