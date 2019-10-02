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





//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//
//#include <stdio.h>
//
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
//
//__global__ void addKernel(int *c, const int *a, const int *b)
//{
//    int i = threadIdx.x;
//    c[i] = a[i] + b[i];
//}
//
//int main()
//{
//    const int arraySize = 5;
//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
//    int c[arraySize] = { 0 };
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
//        c[0], c[1], c[2], c[3], c[4]);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
