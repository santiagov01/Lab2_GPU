#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#define MAX_POINTS 10000000 // maximum number of points allowed

// Load points from a CSV file
int load_csv(const char *filename, float **out_points)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        printf("Error: cannot open file %s\n", filename);
        return -1;
    }

    float *points = (float*)malloc(MAX_POINTS * 2 * sizeof(float));
    if (!points)
    {
        printf("Error allocating memory\n");
        fclose(f);
        return -1;
    }

    int count = 0;
    float x, y;

    while (fscanf(f, "%f,%f", &x, &y) == 2)
    {
        if (count >= MAX_POINTS)
            break;
        points[2 * count] = x;
        points[2 * count + 1] = y;
        count++;
    }

    fclose(f);
    *out_points = points;
    return count;
}

// Compute squared Euclidean distance (no sqrt for speed)
__device__ static inline float dist2(float px, float py, float cx, float cy)
{
    float dx = px - cx;
    float dy = py - cy;
    return dx * dx + dy * dy;
}

__global__ void assign_clusters_gpu(const float *points, const float *centroids, int *labels, int N, int K)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= N) return;

    float px = points[2 * index];
    float py = points[2 * index + 1];

    float best_dist = FLT_MAX;
    int best_c = -1;

    for (int c = 0; c < K; c++)
    {
        float cx = centroids[2 * c];
        float cy = centroids[2 * c + 1];

        float d = dist2(px, py, cx, cy);
        if (d < best_dist)
        {
            best_dist = d;
            best_c = c;
        }
    }

    labels[index] = best_c;
}

__device__  static inline  void reduce(int index, int k, float * sum) {
    for(int c = 0; c < k; c++){
        int tam = blockDim.x / 2; 
        while(tam > 0) {
            __syncthreads();
            if (index < tam){ 
                sum[2 * (c * blockDim.x + index)] += sum[2 * (c * blockDim.x + index + tam)];
                sum[2 * (c * blockDim.x + index) + 1] += sum[2 * (c * blockDim.x + index + tam) + 1];
            }
            tam /= 2;
        }
    }
}

__device__ static inline  void reduce_count(int index, int k, int * count) {
    for(int c=0; c<k; c++){
      int tam = blockDim.x/2;
        while(tam>0) {
             __syncthreads();
            if (index < tam){
              count[c * blockDim.x + index] += count[c * blockDim.x + index + tam];
            }
            tam /= 2;
            //__syncthreads();
        }
    }
}


// Recompute centroids and return total centroid movement
__global__ void update_centroids_gpu(const float *points, const int *labels, int N, int K, float *results_sum, int *results_count)
{
    extern __shared__ unsigned char s[];

    float *sum = (float*) s; 
    int   *count = (int*)(s + 2 * K * blockDim.x * sizeof(float));      

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    
    // TODOS los threads inicializan shared memory
    for (int i = threadIdx.x; i < K * blockDim.x; i += blockDim.x){
        sum[2 * i] = 0.0f;
        sum[2 * i + 1] = 0.0f;
        count[i] = 0;
    }


    // Solo threads válidos acumulan datos
    if (index < N) {
        int c = labels[index];
        sum[2 * (c * blockDim.x + threadIdx.x)]     = points[2 * index];
        sum[2 * (c * blockDim.x + threadIdx.x) + 1] = points[2 * index + 1];
        count[c * blockDim.x + threadIdx.x]         = 1;
    }

    __syncthreads();

    // TODOS los threads participan en la reducción
    reduce(threadIdx.x, K, sum);
    reduce_count(threadIdx.x, K, count);

    __syncthreads(); 

    if(threadIdx.x == 0) {
        for(int c = 0; c < K; c++){
            results_sum[2 * (blockIdx.x * K + c)] = sum[2 * (c * blockDim.x)];
            results_sum[2 * (blockIdx.x * K + c) + 1] = sum[2 * (c * blockDim.x) + 1];
            results_count[blockIdx.x * K + c] = count[c * blockDim.x];
        }
    }
}

// Main K-means loop with convergence test
void kmeans_cpu(float *points, float *centroids,int N, int K, int max_iters, float epsilon)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    int size = 2 * N * sizeof(float);
    int *labels = (int*)malloc(N * sizeof(int));
    double *iter_times = (double *)malloc(max_iters * sizeof(double));
    
    if(!labels){
        fprintf(stderr, "Failed to allocate memory for labels\n");
        exit(EXIT_FAILURE);
    }

    float *d_points, *d_centroids,*d_results_sum;
    int *d_labels,*d_results_count;
    cudaMalloc((void **)&d_points, size);
    cudaMalloc((void **)&d_centroids, 2 * K * sizeof(float));
    cudaMalloc((void **)&d_labels, N * sizeof(int));

    cudaMalloc((void **)&d_results_sum, 2 * K * numBlocks * sizeof(float));
    cudaMalloc((void **)&d_results_count,K * numBlocks * sizeof(int));

    cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, 2 * K * sizeof(float), cudaMemcpyHostToDevice);



    float *results_sum = (float*)malloc(K * 2 * numBlocks * sizeof(float));
    int *results_count = (int*)malloc(K * numBlocks * sizeof(int));
    int actual_iters = 0;

    for (int it = 0; it < max_iters; it++)
    {
        clock_t start = clock();
        
        cudaMemcpy(d_centroids, centroids, K * 2 * sizeof(float), cudaMemcpyHostToDevice);
        assign_clusters_gpu<<<numBlocks, blockSize>>>(d_points, d_centroids, d_labels, N, K);
        // cudaMemcpy(labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

        update_centroids_gpu<<<numBlocks, blockSize, K * blockSize * 2 * sizeof(float) + K * blockSize * sizeof(int)>>>(d_points, d_labels, N, K,d_results_sum, d_results_count);
        cudaMemcpy(results_sum, d_results_sum, K * 2 * numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(results_count, d_results_count,K * numBlocks * sizeof(int), cudaMemcpyDeviceToHost);
        

        float result_sum_x = 0,result_sum_y = 0,result_count=0;
        float movement = 0.0f;
        for(int c = 0; c < K; c++) {
          result_sum_x = 0,result_sum_y = 0,result_count=0;
          for(int i = 0; i < numBlocks; i++){
            result_sum_x += results_sum[2*(i * K + c)];
            result_sum_y += results_sum[2*(i * K + c)+1];
            result_count += results_count[(i * K + c)];
          }

        //   // Compute new centroids + track movement for convergence
          float newx = centroids[2 * c];
          float newy = centroids[2 * c + 1];

          if (result_count > 0)
          {
            newx = result_sum_x / result_count;
            newy = result_sum_y / result_count;
          }

          float dx = newx - centroids[2 * c];
          float dy = newy - centroids[2 * c + 1];
          movement += dx * dx + dy * dy;

          centroids[2 * c] = newx;
          centroids[2 * c + 1] = newy;
        }
        
        clock_t end = clock();
        double iter_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
        iter_times[it] = iter_time;
        actual_iters++;
        
        printf("Iteration %d - centroid movement = %.6f, time = %.3f ms\n", it, movement, iter_time);

        if (movement < epsilon)
        {
            printf("Converged after %d iterations.\n", it);
            break;
        }
    }

    // Calculate and display average time
    double total_time = 0.0;
    for (int i = 0; i < actual_iters; i++)
    {
        total_time += iter_times[i];
    }
    double avg_time = total_time / actual_iters;
    
    printf("Total iterations: %d\n", actual_iters);
    printf("Total time: %.3f ms\n", total_time);
    printf("Average time per iteration: %.3f ms\n", avg_time);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_results_sum);
    cudaFree(d_results_count);
    free(results_sum);
    free(results_count);
    free(iter_times);
    free(labels);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage: %s data.csv\n", argv[0]);
        return 1;
    }

    float *points = NULL;
    int N = load_csv(argv[1], &points);
    if (N <= 0)
        return 1;

    int K = 3;
    float centroids[6] = {0, 0, 5, 5, 10, 10}; // simple initial seeds

    printf("Loaded %d points.\n", N);

    int max_iters = 100;
    float ep = 1e-4f;

    kmeans_cpu(points, centroids, N, K, max_iters, ep);

    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; c++)
    {
        printf("C%d = (%f, %f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }

    free(points);
    return 0;
}