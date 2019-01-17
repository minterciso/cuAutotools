# Intro
This repository is a simple template to use GNU Autotools with CUDA. There are of course other ways to make this work, even better ways, but this is the way it
works for me.

## Details
The big detail here is that all CUDA code must be inside a .cu file, otherwise the Autotools will try to compile using the g++ compiler, instead of the nvcc. You
can use .h files for headers with mixed host/device code, even for leaving macros in the .h file and using them on the .cu files. 

Apart from this, everything is like a normal autotools project.
