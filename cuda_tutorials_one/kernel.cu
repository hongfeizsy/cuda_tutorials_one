#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "add_functions.cuh"


#define USE_UNIX 0

#if USE_UNIX
#include <sys/time.h>
#include <time.h>

double get_time() {
	struct timeval tv;
	double t;

	gettimeofday(&tv, (struct timezone *)0);
	t = tv.tv_sec + (double)tv.tv_usec * 1e-6;
	
	return t;
}

#else
#include <Windows.h>

double get_time() {
	LARGE_INTEGER timer;
	static LARGE_INTEGER fre;
	static int init = 0;
	double t;

	if (init != 1) {
		QueryPerformanceFrequency(&fre);
		init = 1;
	}

	QueryPerformanceCounter(&timer);
	t = timer.QuadPart * 1.0 / fre.QuadPart;
	
	return t;
}
#endif


int main() {
	int N = 20000000;
	int nbytes = sizeof(float) * N;

	/* 1D block */
	int block_size = 256;

	/* 2D grid */
	int s = ceil(sqrt((N + block_size - 1.0) / block_size));
	dim3 grid = dim3(s, s);
	
	float *dx = NULL, *hx = NULL;
	float *dy = NULL, *hy = NULL;
	float *dz = NULL, *hz = NULL;

	int itr = 30;
	int i;
	double th, td;

	/* allocate GPU memory */
	cudaMalloc((void **)&dx, nbytes);
	cudaMalloc((void **)&dy, nbytes);
	cudaMalloc((void **)&dz, nbytes);

	if (dx == NULL || dy == NULL || dz == NULL) {
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
	printf("allocated %.2f MB on GPU\n", nbytes / (1024.0 * 1024.0));

	/* allocate CPU memory */
	hx = (float*)malloc(nbytes);
	hy = (float*)malloc(nbytes);
	hz = (float*)malloc(nbytes);

	/* init */
	for (int i = 0; i < N; i++) {
		hx[i] = 1;
		hy[i] = 1;
		hz[i] = 1;
	}

	/* copy data to GPU */
	cudaMemcpy(dx, hx, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, hy, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dz, hz, nbytes, cudaMemcpyHostToDevice);

	/* warm up GPU */
	warmup();

	/* call GPU */
	cudaThreadSynchronize();
	td = get_time();
	for (int i = 0; i < itr; i++) vec_add<<<grid, block_size >>> (dx, dy, dz, N);
	cudaThreadSynchronize();
	td = get_time() - td;

	/* call CPU */
	th = get_time();
	for (int i = 0; i < itr; i++) vec_add_host(hx, hy, hz, N);
	th = get_time() - th;
	float temp = hz[0];

	printf("GPU time: %e, CPU time: %e, speedup: %g\n", td, th, th / td);
	free(hx);
	free(hy);
	free(hz);
	cudaFree(hx);
	cudaFree(hy);
	cudaFree(hz);

	return 0;
}
