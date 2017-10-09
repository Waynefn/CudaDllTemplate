#include "stdafx.h"
#include <stdio.h>
#include<math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "dllCuda.h"
int addWithCuda(int *c, const int *a, const int *b, unsigned int size);
bool isLoadDevice = false;


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
 int addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	 if (isLoadDevice == false)return -1;
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    int cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed\n");
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

 int deviceReset(void)
 {
	 int cnt;
	 cudaGetDeviceCount(&cnt);
	 if (cnt > 0)
	 {
		 isLoadDevice = true;
		 return cudaDeviceReset();
	 }
		 
	 else
		 return -1;
 }

#define BLOCK_SIZE 1024
#define   LIGHT   3e8
#define PI 3.14159265358979333f
 const float r_ref = 10530;
 const float factor_a = 0.5f;//(float)(Va/prf/1.0);//index=1
 const float factor_r = 2.5f;//(float)(LIGHT/2/fs/1.0);//index=1
 struct Paralist{
	 
	 const float r_ref = 10530;
	 const float fs = 60e6f;
	 const float lmda = 0.1f;
	 const float prf = 600;
	 const float R_near = 10000;//=R0
	 const int Va = 300;
	 const float bita = (float)1.2*PI / 180;

	 int n_a; 
	 float pow_ref_point_rj; 
	 float point_aj; 
	 float point_rj;
	 int interp_num;
 };

/*
result应该置零!
*/
__global__ void phaseKernel(float*result,Paralist plist)
{

	int i = blockIdx.x*gridDim.x+ threadIdx.x;
	if (i<plist.n_a)
	{
		float tai = (float)((i - plist.n_a / 2) / plist.prf);
		float Rj = (float)sqrt(plist.pow_ref_point_rj + pow(plist.point_aj - plist.Va*tai, 2));
		float angle0 = (float)acos((plist.r_ref + plist.point_rj) / Rj);
		if (angle0>plist.bita / 2)
			return;

		float phase = fmodf(-4.0*PI*Rj / plist.lmda, 2 * PI);
		float delay = (float)((Rj - plist.R_near) * 2 / LIGHT);
		int p_num = (int)floor(delay*plist.fs * 4 + 0.5);
		if (p_num < plist.interp_num)
		{
			result[i] = phase;
		}
	}
	
}


 int mapWithCuda(int interm_j, int interp_num, int n_a, float*resl)
 {
	 if (isLoadDevice == false)return -1;
	 int cudaStatus;
	 float point_aj = (interm_j / 300)*factor_a;     
	 float point_rj = (interm_j % 300)*factor_r;     
	 float pow_ref_point_rj = (float)pow(r_ref + point_rj, 2);
	 dim3 gridSize((n_a+1) / BLOCK_SIZE);

	 float*dev_resl;
	 cudaStatus= cudaMalloc((void**)&dev_resl, sizeof(float)*n_a);
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "device malloc error!\n");
		 
	 }
	 cudaMemset(dev_resl, 0, n_a);

	 Paralist pl;

	 pl.n_a = n_a;
	 pl.pow_ref_point_rj = pow_ref_point_rj;
	 pl.point_aj = point_aj;
	 pl.point_rj = point_rj;
	 pl.interp_num = interp_num;

	 phaseKernel << <gridSize, BLOCK_SIZE >> >(dev_resl,pl);
	
	 cudaStatus = cudaGetLastError();
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "kernel launch failed.\n");

	 }
	 cudaStatus=cudaMemcpy(resl, dev_resl, sizeof(float)*n_a, cudaMemcpyDeviceToHost);
	 if (cudaStatus != cudaSuccess) {
		 fprintf(stderr, "Memcopy error.\n");
	 }

	 return cudaFree(dev_resl);

 }
