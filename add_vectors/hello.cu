#include <stdio.h>
__global__ void kernelHello() {
    printf("Hi, I’m the GPU!");
}
int main() {
    kernelHello<<<1, 1>>>();
    if(cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
    }
    printf("Hi, I’m the CPU!\n");
    return 0;
}