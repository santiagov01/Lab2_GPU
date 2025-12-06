#include <iostream>

__global__
void add(int n, float *x, float *y){
    int index = threadIdx.x  + blockIdx.x * blockDim.x;
    if(index < n){
        y[index] = x[index] + y[index];
    }
}
int main(void){
    int N = 1 << 20;
    float *x = new float[N];
    float *y = new float[N];
    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int size = N*sizeof(float);
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add<<numBlocks, blockSize>>(N, d_x, d_y);
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    //Free memory
    cudaFree(d_x); cudaFree(d_y);
    delete[] x; delete[] y;
    return 0;
}
