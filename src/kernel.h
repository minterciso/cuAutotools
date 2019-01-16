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

