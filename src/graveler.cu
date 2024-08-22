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
void init_device_state(size_t n, curandState *rand_states, uint64_t seed) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  curandState *rand_state = rand_states + index;
  curand_init(seed, index, 0, rand_state);
}

__global__
void graveler_streaks(size_t n, uint32_t *one_counts, curandState *rand_states) {
  size_t i;
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t num_threads = gridDim.x * blockDim.x;
  curandState *rand_state = rand_states + index;

  for(i = index; i < N; i += num_threads) {
    uint32_t one_count = 0;
    for(size_t rolls = 0; rolls < 231; rolls++) {
      uint32_t roll = (uint32_t)(curand_uniform(rand_state) * 3.0 + 0.5f);
      if(roll != 0) break;
      ++one_count;
    }
    one_counts[i] = one_count;
  }
}

__global__
void graveler_total(size_t n, uint32_t *one_counts, curandState *rand_states) {
  size_t i;
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t num_threads = gridDim.x * blockDim.x;
  curandState *rand_state = rand_states + index;

  for(i = index; i < N; i += num_threads) {
    uint32_t one_count = 0;
    for(size_t rolls = 0; rolls < 231; rolls++) {
      uint32_t roll = (uint32_t)(curand_uniform(rand_state) * 3.0 + 0.5f);
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

  int default_device;
  cudaDeviceProp device_props;
  cudaGetDevice(&default_device);
  cudaGetDeviceProperties(&device_props, default_device);
  fprintf(stdout, "Using device %s\n", device_props.name);

  init_device_state<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(N, d_rand_states, time(0));
  cudaDeviceSynchronize();
  puts("Device initialized");

  DECLARE_TIMER(GravelerKernelTimer);
  START_TIMER(GravelerKernelTimer);
  graveler_streaks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(N, d_one_counts, d_rand_states);
  cudaDeviceSynchronize();
  STOP_TIMER(GravelerKernelTimer);

  cudaMemcpy(h_one_counts, d_one_counts, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  puts("Computation complete");
  puts("Calculating statistics");
  uint32_t max_ones = 0;
  uint32_t min_ones = ~0;
  uint64_t total_ones = 0;
  for(size_t i = 0; i < N; ++i) {
    total_ones += h_one_counts[i];
    if(h_one_counts[i] > max_ones) {
      max_ones = h_one_counts[i];
    }
    if(h_one_counts[i] < min_ones) {
      min_ones = h_one_counts[i];
    }
  }
  fprintf(stdout, "\nNumber of attempts: %lu\nMax ones rolled: %u\nMin ones rolled:%u\nAverage ones: %0.2lf\n",
    N, max_ones, min_ones, (double)total_ones / (double)N);
  PRINT_TIMER(GravelerKernelTimer);

  cudaFree(d_one_counts);
  cudaFree(d_rand_states);
  free(h_one_counts);
  
  return 0;
}
