#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel
__global__
void vecCompare(int *R, int *G, int *B, int *result, int n)     //A is for the green array
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        if(G[i] > 2 * R[i] || G[i] > 2 * B[i])
        {
            result[i] = 1;
        }
        else
        {
            result[i] = 0;
        }
    }
}

extern "C"
void compareMatrices(int height, int width, int*r, int*g, int*b, int*green)
{
    // Size of vectors
    int n = height * width;

    // Device input vectors
    int *d_r;
    int *d_g;
    int *d_b;
    //Device output vector
    int *d_green;

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(int);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_r, bytes);
    cudaMalloc(&d_g, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_green, bytes);

    // Copy host vectors to device
    cudaMemcpy( d_r, r, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_g, g, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    vecCompare<<<gridSize, blockSize>>>(d_r, d_g, d_b, d_green, n);

    // Copy array back to host
    cudaMemcpy( green, d_green, bytes, cudaMemcpyDeviceToHost );

    // Release device memory
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_green);

    // Release host memory
    free(r);
    free(g);
    free(b);
}
