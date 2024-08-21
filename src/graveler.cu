#include "stdio.h"
#include "math.h"
#include "stdint.h"

#include <curand.h>
#include <curand_kernel.h>

//#define N (1000000000)
#define N (1000000)
//#define N (100)

#define BLOCKS_PER_GRID (1)
#define THREADS_PER_BLOCK (256)


__global__
void graveler(size_t n, uint32_t *one_counts, curandState *rand_states, uint64_t seed) {
  int i;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;
  curandState *state = rand_states + tid;
  curand_init(seed, tid, 0, state);

  for(i = tid; i < N; i += nthreads) {
    float rand = curand_uniform(state) * 4.0 + 0.999999;// TODO:
    uint32_t one_count = 0;
    for(size_t rolls = 0; rolls < 231; rolls++) {
      if(((int)rand) % 4 == 0) {
        ++one_count;
      }
    }
    one_counts[i] = one_count;
    //one_counts[i] = curand_uniform(state) * 100;
  }
  if(tid < N) one_counts[tid] = 69;
}

int main(void) {
  uint32_t *h_one_counts, *d_one_counts;
  curandState *d_rand_states;
  cudaMalloc(&d_rand_states, sizeof(curandState) * THREADS_PER_BLOCK * BLOCKS_PER_GRID);
  cudaMalloc(&d_one_counts, N * sizeof(uint32_t));
  h_one_counts = (uint32_t *)malloc(N * sizeof(uint32_t));

  graveler<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(N, d_one_counts, d_rand_states, time(0));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString(err));
  }

  cudaMemcpy(h_one_counts, d_one_counts, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  uint32_t max_ones = 0;
  for(size_t i = 0; i < N; ++i) {
    if(h_one_counts[i] > max_ones) {
      max_ones = h_one_counts[i];
    }
  }
  fprintf(stdout, "%ld\n", max_ones);

  cudaFree(d_one_counts);
  cudaFree(d_rand_states);
  free(h_one_counts);
  
  return 0;
}