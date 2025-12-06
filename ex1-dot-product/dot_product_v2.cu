//%%writefile dot_v2.cu
#include "iostream"
#define threadsPerBlock 256

__global__ void dot_product(int N, float *x, float *y,float *result) {
    __shared__ float cache[threadsPerBlock];
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;

    for(int i = index; index < N; index += stride)
        temp += x[i] * y[i];

    // Store partial result in shared memory
    cache[cacheIndex] = temp;
    __syncthreads();

    int tam = blockDim.x/2;
    // Reduction within block
    while(tam>0) {
        __syncthreads();
        if (cacheIndex < tam)
            cache[cacheIndex] += cache[cacheIndex+tam];
        tam /= 2;
    }

    // First thread writes block result
    if(cacheIndex == 0) {
        result[blockIdx.x] = cache[0];
    }
}

int main() {
    int N = 1<<20; // 1 million threads
    float *x = new float[N];
    float *y = new float[N];

    for (int i = 0; i < N; i++) {
        x[i] = 2.0f; y[i] = 3.0f;
    }

    int size = N * sizeof(float);
    float *d_x, *d_y, *result;
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);
    cudaMalloc((void **)&result, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);


    int blockSize = 256;
    int numBlocks = (blockSize + N + 1)/N;
    dot_product<<<numBlocks,blockSize>>>(N, d_x, d_y,result);

    float *final_result = new float[numBlocks];

    cudaMemcpy(final_result, result, numBlocks*sizeof(float), cudaMemcpyDeviceToHost);

    float final = 0;
    for(int i = 0; i < numBlocks ; i++){
      final += final_result[i];
    }

    printf("final = %f\n",final);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(result);


    delete[] x;
    delete[] y;
    delete[] final_result;

    return 0;
}