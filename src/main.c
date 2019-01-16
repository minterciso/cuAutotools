#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "kernel.h"

int main(int argc, char **argv){
    
    start_device();
    call_kernel();
    reset_device();

    return EXIT_SUCCESS;
}
