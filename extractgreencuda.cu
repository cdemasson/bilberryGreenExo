#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA kernel. Each thread takes care of one element of c
__global__
void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
    {
        c[id] = a[id] + b[id];
    }
}

extern "C"
double function()
{
    // Size of vectors
    int n = 100000;

    // Host input vectors
    double *h_a;
    double *h_b;
    //Host output vector
    double *h_c;

    // Device input vectors
    double *d_a;
    double *d_b;
    //Device output vector
    double *d_c;

    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_a[i] = sin(i)*sin(i);
        h_b[i] = cos(i)*cos(i);
    }

    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize, gridSize;

    // Number of threads in each thread block
    blockSize = 1024;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)n/blockSize);

    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy array back to host
    cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );

    // Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    sum = sum/n;

    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return sum;
}



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
int compareMatrices(int height, int width, int*r, int*g, int*b, int*green)
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

    return 0;
}
