#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernel.h"

int main(int argc, char **argv){
    
    start_device();
    call_dummy_kernel();
    if(call_rand_kernel() < 0)
        fprintf(stderr,"[E] Error calling random kernel\n");
    reset_device();

    return EXIT_SUCCESS;
}
