CSRC_BASE = histogram.c 
CSRC_OMP = histogram_omp.c
CSRC_CUDA = histogram_cuda.cu

PROG_CPU = histogram
PROG_OMP = histogram_omp
PROG_CUDA = histogram_cuda
PROG = $(PROG_CPU) $(PROG_OMP) $(PROG_CUDA)

CC = gcc
CFLAGS = -Wall -g -O3 
LDLIBS = -lm

OMP_CFLAGS = -fopenmp
OMP_LDLIBS = -fopenmp

NVCC = nvcc
NVCCFLAGS = -g -O3 -arch=sm_86 #archi gpu cremi
CUDALDLIB = -lcudart -lm

.phony: all clean

all: $(PROG)

$(PROG_CPU): $(CSRC_BASE)
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

$(PROG_OMP): $(CSRC_OMP)
	$(CC) $(CFLAGS) $(OMP_CFLAGS) $< -o $@ $(LDLIBS) $(OMP_LDLIBS)

$(PROG_CUDA): $(CSRC_CUDA)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(CUDALDLIB)

clean:
	rm -fv $(PROG)