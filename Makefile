# Makefile
NVCC=nvcc
ARCH_A100=-arch=sm_80
ARCH_H100=-arch=sm_90
LIBS=-lcublas -lcudnn

all: bench_a100 bench_h100

bench_a100: benchmark.cu
	$(NVCC) $(ARCH_A100) benchmark.cu -o bench_a100 $(LIBS)

bench_h100: benchmark.cu
	$(NVCC) $(ARCH_H100) benchmark.cu -o bench_h100 $(LIBS)