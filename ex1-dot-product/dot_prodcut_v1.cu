//%%writefile dot_v1.cu
#include "iostream"

__global__ void mul(int N, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index;i<N;i+=stride)
        y[i] = x[i] * y[i];
}

__global__ void add(int N, float *respuesta, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N; i += stride)
        atomicAdd(respuesta, y[i]);
}

int main() {
    int N = 1<<20; // 1 million threads
    float *x = new float[N];
    float *y = new float[N];

    for (int i = 0; i < N; i++) {
        x[i] = 2.0f; y[i] = 3.0f;
    }

    int size = N * sizeof(float);
    float *d_x, *d_y;

    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    mul<<<32* numSMs,blockSize>>>(N, d_x, d_y);

    cudaDeviceSynchronize();

    float * respuesta;
    cudaMalloc((void **)&respuesta, sizeof(float));
    add<<<32* numSMs,blockSize>>>(N, respuesta,d_y);
    cudaMemcpy(&y[0], respuesta, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "respuesta = " << y[0] << std::endl;


    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(respuesta);


    delete[] x;
    delete[] y;

    return 0;
}