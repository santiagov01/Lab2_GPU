# Makefile for K-means implementations

CC = gcc
NVCC = nvcc
CFLAGS = -O2 -Wall
NVCCFLAGS = -O2

# Targets
TARGETS = cpu basic optimized

all: $(TARGETS)

cpu: kmeans.c
	$(CC) $(CFLAGS) -o cpu kmeans.c -lm

basic: kmeans_basic.cu
	$(NVCC) $(NVCCFLAGS) -o basic kmeans_basic.cu

optimized: kmeans_optimized.cu
	$(NVCC) $(NVCCFLAGS) -o optimized kmeans_optimized.cu

clean:
	rm -f $(TARGETS)

.PHONY: all clean
