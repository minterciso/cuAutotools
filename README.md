# Intro
This repository is a simple template to use GNU Autotools with CUDA. There are of course other ways to make this work, even better ways, but this is the way it
works for me.

## Details
The big detail here is that *all* CUDA code **must** be in a .cu file, otherwise the g++ compiler will get lost. Even for .h files, you can of course create
global CUDA variables (for instance curandState) on the .h file, but you can't use any __global__, __host__, __device__ or others on the header file. So what
I like to do is to treat the program as a normal C/C++ program, and create some files *only* for CUDA code. 

On this exempla you'll find the files main.c, kernel.h and kernel.cu. You'll also find that, if you ignore the kernel.cu, kernel.h seems like a normal C/C++
header file, so main.c can include it with no issues and call their functions. Since the kernel.cu is the one handed by the nvcc it'll also compile in an .o
file normally, even if you put some CUDA Kernels in it.

Basically it goes like this:

* main.c ==> kernel.h (g++ handles this)
* kernel.cu ==> cuda.h, curand_kernel.h (nvcc handles this)

And the linking is done by g++ by using -lcudart
