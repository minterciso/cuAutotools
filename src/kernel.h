/*
 * =====================================================================================
 *
 *       Filename:  kernel.h
 *
 *    Description:  Simple CUDA Kernel for automake template
 *
 *        Version:  1.0
 *        Created:  16/01/2019 13:07:07
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Mateus Interciso (mi), minterciso@gmail.com
 *        Company:  Geekvault
 *
 * =====================================================================================
 */
#ifndef __KERNEL_H
#define __KERNEL_H

#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(call)                                                                                 \
{                                                                                                       \
            const cudaError_t error=call;                                                               \
            if(error != cudaSuccess){                                                                   \
                                printf("Error: %s:%d, ", __FILE__, __LINE__);                           \
                                printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
                                exit(1);                                                                \
                        }                                                                               \
}


/**
 * @brief Starts the PRNG for each curandState with seed and a different starting id
 * @param state A pointer to a device memory containing an array of curandState values
 * @param seed The seed, this should be different on each execution
 * @param qtd The size of the state array
 */
__global__ void setup_prng(curandState *state, unsigned long long seed, unsigned int qtd);

/**
 * @brief Create Random qtd_data random floats and store the average on the data array
 * @param state The already initialized array of curandState states
 * @param qtd_states The amount of states (this must be the same size of data)
 * @param data A float array initialized on the GPU to store the results
 * @param qtd_data How may samples we will create
 */
__global__ void create_random_data(curandState *state, unsigned int qtd_states, float *data, unsigned int qtd_data);

/**
 * @brief A simple dummy kernel that just prints a Hello World
 */
__global__ void dummy_kernel(void);

/**
 * @brief Call a dummy kernel that simply prints a Hello World message from the GPU
 */
void call_dummy_kernel(void);

/**
 * @brief Calls a kernel to create 4.0 MB of random float numbers and print the average of those numbers.
 *
 * This works by creating the same amount of curandState (default to XORXOW generator) and calling the kernel 1000 times. On each kernel we create the
 * average call and store them on the data array passed. Once all kernels are finished we create the average of it.
 * 
 * @note Honestly this is overkill and a completelly bogus scenario, we could create for instance 512 generators, create the random data on each thread
 * and just return the value created, and then finding the average. But since this is just to show how to run a more complex kernel, we'll leave it like this.
 * @return 0 on success, -1 on error
 */
int call_rand_kernel(void);

/**
 * @brief Start the device with id 0
 */
void start_device(void);

/**
 * @brief Reset the started device and clear it for usage
 */
void reset_device(void);

#endif //__KERNEL_H

