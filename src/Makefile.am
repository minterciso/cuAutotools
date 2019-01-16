.cu.o:
	$(NVCC) -o $@ -c $(NVCC_CFLAGS) --ptxas-options=-v --compiler-options="$(CFLAGS) $(DEFAULT_INCLUDES) $(INCLUDES) $(AM_CPPFLAGS) $(CPPFLAGS)" $<

CC = g++
CCLD=$(CC)
bin_PROGRAMS=autotools_template
autotools_template_SOURCES = main.c\
			     kernel.h kernel.cu
autotools_template_CFLAGS  = $(CUDA_CFLAGS)
autotools_template_LDADD  = $(CUDA_LIBS)
