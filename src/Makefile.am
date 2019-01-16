.cu.o:
	$(NVCC) -o $@ -c $(NVCC_CFLAGS) --ptxas-options=-v --compiler-options="$(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS)" $<

CC = g++
CCLD=$(CC)
bin_PROGRAMS=cuAutotools
cuAutotools_SOURCES = 	main.c\
			kernel.h kernel.cu
cuAutotools_CFLAGS  = $(CUDA_CFLAGS)
cuAutotools_LDADD  = $(CUDA_LIBS) -lcurand
