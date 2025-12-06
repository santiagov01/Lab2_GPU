#include <iostream>

__global__
void add(int n, lfloat *x, float *y){
    int index = threadIdx.x;
    int stride = blockDim.x;
    //In all the threads first iteration will execute the first 256 elements
    // Then second iteration will execute the next 256 elements and so on
    for (int i = index; i < n; i+=stride){
        y[i] = x[i] + y[i];
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
    //Only one block. 256 threads
    add<<<1,256>>>(N, d_x, d_y);
    cudaMemcpy(y, d_y, size, cudaMemcpyDeviceToHost);
    //Free memory
    cudaFree(d_x); cudaFree(d_y);
    delete[] x; delete[] y;
    return 0;
}
