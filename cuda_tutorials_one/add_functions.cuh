#include <stdio.h>
#include "cuda_runtime.h"


/* warm up GPU */
__global__ void warmup_knl() {
	int i, j;
	i = 1;
	j = 10;
	i = i + j;
}

void warmup() {
	for (int i = 0; i < 8; i++) {
		warmup_knl<<<1, 256 >>>();
	}
}


/* get thread id: 1D block and 2D grid */
#define get_tid() (blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x)

/* device function */
__global__ void vec_add(float* x, float* y, float* z, int N) {
	/* 1D block */
	int idx = get_tid();

	if (idx < N) z[idx] = x[idx] + y[idx] + z[idx];
}

/* host function */
void vec_add_host(float *x, float *y, float *z, int N) {
	for (int i = 0; i < N; i++) z[i] = x[i] + y[i] + z[i];
}

