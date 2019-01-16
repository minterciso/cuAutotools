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
#include <cuda.h>

__global__ void dummy_kernel(void){
    const int tid = threadIdx.x + blockIdx.x*blockDim.x;
    printf("Hello from thread %d\n", tid);
}

void call_kernel(void){
    fprintf(stdout,"[*] Calling Dummy kernel.\n");
    dummy_kernel<<<1, 1>>>();
}

void start_device(void){
    int dev = 0;
    cudaDeviceProp device_properties;
    
    cudaGetDeviceProperties(&device_properties, dev);

    fprintf(stdout,"[*] Setting device %d: %s\n", dev, device_properties.name);
    cudaSetDevice(dev);
}

void reset_device(void){
    cudaDeviceReset();
}

