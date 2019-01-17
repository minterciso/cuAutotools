/*
 * =====================================================================================
 *
 *       Filename:  kernel.cu
 *
 *    Description:  Sample Kernel for Automake template
 *
 *        Version:  1.0
 *        Created:  16/01/2019 13:08:01
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Mateus Interciso (mi), minterciso@gmail.com
 *        Company:  Geekvault
 *
 * =====================================================================================
 */
#include "kernel.h"
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

__global__ void setup_prng(curandState *state, unsigned long long seed, unsigned int qtd){
    const int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid < qtd)
        curand_init(seed, tid, 0, &state[tid]);
}

__global__ void create_random_data(curandState *state, unsigned int qtd_states, float *data, unsigned int qtd_data){
    const int tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(tid < qtd_states){
        curandState local_state = state[tid];
        float val = 0.0f;
        for(int i=0;i<qtd_data;i++)
            val += curand_uniform(&local_state);
        val /= qtd_data;
        data[tid] += val;
        state[tid] = local_state;
    }
}

__global__ void dummy_kernel(void){
    const int tid = threadIdx.x + blockIdx.x*blockDim.x;
    printf("Hello from thread %d\n", tid);
}

int call_rand_kernel(void){
    // Variables
    float *h_data, *d_data;
    curandState *d_state;
    unsigned int size = 1024*1024;
    unsigned int data_bytes = sizeof(float)*size;
    unsigned int state_bytes = sizeof(curandState)*size;
    int numThreads = 512; // numThreads is an arbitrary value on this case
    int numBlocks = size/numThreads + 1; // numBlocks however is the amount of blocks with numThreads we need to create all size data

    fprintf(stdout,"[*] Using kernel configuration %d, %d\n", numBlocks, numThreads);
    fprintf(stdout,"[*] Allocating memory for random kernel\n");
    fprintf(stdout,"[*] Allocating %5.2f MB of data\n", data_bytes/(1024.0f*1024.0f));
    if((h_data = (float*)malloc(data_bytes))==NULL){
        perror("malloc");
        return -1;
    }
    CUDA_CALL(cudaMalloc((void**)&d_data, data_bytes));
    CUDA_CALL(cudaMemset(d_data, 0, data_bytes));
    fprintf(stdout,"[*] Allocating PRNG state memory\n");
    CUDA_CALL(cudaMalloc((void**)&d_state, state_bytes));
    fprintf(stdout,"[*] Starting PRNG\n");
    unsigned long long seed = time(NULL);
    setup_prng<<<numBlocks, numThreads>>>(d_state, seed, size);
    CUDA_CALL(cudaDeviceSynchronize());
    fprintf(stdout,"[*] Creating random values\n");
    create_random_data<<<numBlocks, numThreads>>>(d_state, size, d_data, 1000);
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(h_data, d_data, data_bytes, cudaMemcpyDeviceToHost));
    fprintf(stdout,"[*] Calculating mean values\n");
    float sum = 0.0;
    float mean = 0.0;
    for(int i=0;i<size;i++) sum += h_data[i];
    mean = sum/size;
    fprintf(stdout,"[*] Mean values = %.10f\n", mean);

    fprintf(stdout,"[*] Cleaning memory\n");
    CUDA_CALL(cudaFree(d_data));
    CUDA_CALL(cudaFree(d_state));
    free(h_data);
    return 0;
}

void call_dummy_kernel(void){
    fprintf(stdout,"[*] Calling Dummy kernel.\n");
    dummy_kernel<<<1, 1>>>();
    CUDA_CALL(cudaDeviceSynchronize()); // This is optional, but since we are just demoing...
}

void start_device(void){
    // Select the first device, we could change this to select the BEST device on a Multi GPU system.
    int dev = 0;
    cudaDeviceProp device_properties;
    
    CUDA_CALL(cudaGetDeviceProperties(&device_properties, dev));

    fprintf(stdout,"[*] Setting device %d: %s\n", dev, device_properties.name);
    CUDA_CALL(cudaSetDevice(dev));
}

void reset_device(void){
    CUDA_CALL(cudaDeviceReset());
}

