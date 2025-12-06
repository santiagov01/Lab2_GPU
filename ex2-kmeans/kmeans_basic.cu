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

    float *points = (float *)malloc(MAX_POINTS * 2 * sizeof(float));
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

__global__ void assign_clusters_gpu(
    const float *points, // size N*2
    const float *centroids, // size K*2
    int *labels, // size N
     int N, int K,
    float *sum, //size K*2
    int *count //size K
    )
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    //if (index >= N) return;
    if(index < N){
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

    // //Asegurar que vector de suma es cero
    // if(index == 0){
    //     for(int i = 0; i < K*2; i++){
    //         sum[i] = 0.0f;
    //     }
    //     for(int i = 0; i < K; i++){
    //         count[i] = 0;
    //     }
    // }

    // __syncthreads();

    // Sumar de forma atomica a los resultados globales
    atomicAdd(&sum[2 * best_c], px);
    atomicAdd(&sum[2 * best_c + 1], py);
    atomicAdd(&count[best_c], 1);
    }

    //imprimir la suma y conteo por cluster
    // if(index == 0){
    //     for(int c = 0; c < K; c++){
    //         printf("Cluster %d: SumX = %f, SumY = %f, Count = %d\n", c, sum[2 * c], sum[2 * c + 1], count[c]);
    //     }
    // }

}

// Recompute centroids and return total centroid movement
__global__ void update_centroids_gpu(
    const float *points,
    float *centroids,
    const int *labels,
    int N, int K,
    float *sum,
    int *count,
    float *movement)
{


    // Compute new centroids + track movement for convergence
    *movement = 0.0f;

    for (int c = 0; c < K; c++)
    {

        float newx = centroids[2 * c];
        float newy = centroids[2 * c + 1];

        if (count[c] > 0)
        {
            newx = sum[2 * c] / count[c];
            newy = sum[2 * c + 1] / count[c];
        }

        float dx = newx - centroids[2 * c];
        float dy = newy - centroids[2 * c + 1];
        *movement += dx * dx + dy * dy;

        centroids[2 * c] = newx;
        centroids[2 * c + 1] = newy;
    }
}

// Save labels from all iterations to a CSV file
void save_labels_to_csv(const char *filename, int **all_labels, int N, int num_iters)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        printf("Error: cannot create file %s\n", filename);
        return;
    }

    // Write labels row by row (each row is a point, each column is an iteration)
    for (int i = 0; i < N; i++)
    {
        for (int it = 0; it < num_iters; it++)
        {
            fprintf(f, "%d", all_labels[it][i]);
            if (it < num_iters - 1)
                fprintf(f, ",");
        }
        fprintf(f, "\n");
    }

    fclose(f);
    printf("Labels saved to %s\n", filename);
}

// Save centroids from all iterations to a CSV file
void save_centroids_to_csv(const char *filename, float **all_centroids, int K, int num_iters)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        printf("Error: cannot create file %s\n", filename);
        return;
    }

    // Write header
    fprintf(f, "x,y,iteration\n");

    // Write centroids: each iteration has K centroids
    for (int it = 0; it < num_iters; it++)
    {
        for (int c = 0; c < K; c++)
        {
            float x = all_centroids[it][2 * c];
            float y = all_centroids[it][2 * c + 1];
            fprintf(f, "%f,%f,%d\n", x, y, it);
        }
    }

    fclose(f);
    printf("Centroids saved to %s\n", filename);
}

// Main K-means loop with convergence test
void kmeans_gpu(
    float *points,
    float *centroids,
    int N, int K,
    int max_iters,
    float epsilon)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int size = 2 * N * sizeof(float);
    
    int *labels = (int *)malloc(N * sizeof(int));
    double *iter_times = (double *)malloc(max_iters * sizeof(double));
    int **all_labels = (int **)malloc(max_iters * sizeof(int *));
    float **all_centroids = (float **)malloc(max_iters * sizeof(float *));

    for (int i = 0; i < max_iters; i++)
    {
        all_labels[i] = (int *)malloc(N * sizeof(int));
        all_centroids[i] = (float *)malloc(K * 2 * sizeof(float));
    }
    int actual_iters = 0;

    float *d_points, *d_centroids,*d_results_sum;
    int *d_labels,*d_results_count;
    cudaMalloc((void **)&d_points, size);
    cudaMalloc((void **)&d_centroids, K * 2 * sizeof(float));
    cudaMalloc((void **)&d_labels, N * sizeof(int));
    float *d_movement; // to track centroid movement
    cudaMalloc((void **)&d_movement, sizeof(float));

    cudaMalloc((void **)&d_results_sum, K * 2 * sizeof(float));
    cudaMalloc((void **)&d_results_count, K  * sizeof(int));

    cudaMemcpy(d_points, points, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * 2 * sizeof(float), cudaMemcpyHostToDevice);

    float movement;
    for (int it = 0; it < max_iters; it++)
    {
        // Save centroids before this iteration
        memcpy(all_centroids[it], centroids, K * 2 * sizeof(float));
        clock_t start = clock();
        cudaMemset(d_results_sum, 0, K * 2 * sizeof(float));
        cudaMemset(d_results_count, 0, K * sizeof(int));

        
        // Asigna clusters, suma distancias de cada label, y acumula conteos por label
        assign_clusters_gpu<<<numBlocks, blockSize>>>(d_points, d_centroids, d_labels, N, K, d_results_sum, d_results_count);
        // Se actualiza en un solo hilo ya que los datos están guardados en memoria de GPU
        update_centroids_gpu<<<1, 1>>>(d_points, d_centroids, d_labels, N, K, d_results_sum, d_results_count, d_movement);
        cudaDeviceSynchronize();
        cudaMemcpy(&movement, d_movement, sizeof(float), cudaMemcpyDeviceToHost);

        clock_t end = clock();
        double iter_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
        iter_times[it] = iter_time;
        actual_iters++;
        // Copy labels and movement back to host
        
        cudaMemcpy(centroids, d_centroids, K * 2 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(labels, d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);
        


        // Save labels for this iteration
        memcpy(all_labels[it], labels, N * sizeof(int));

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

    // Save labels to CSV file
    save_labels_to_csv("labels_history.csv", all_labels, N, actual_iters);
    
    // Save centroids to CSV file
    save_centroids_to_csv("centroids_history.csv", all_centroids, K, actual_iters);


    // Free memory in GPU
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_movement);
    cudaFree(d_results_sum);
    cudaFree(d_results_count);
    // Free memory in CPU
    for (int i = 0; i < max_iters; i++)
    {
        free(all_labels[i]);
        free(all_centroids[i]);
    }
    free(all_labels);
    free(all_centroids);
    free(iter_times);
    free(labels);
}

// Example usage: read CSV points, run K-means with fixed K
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

    kmeans_gpu(points, centroids, N, K, max_iters, ep);

    printf("\nFinal centroids:\n");
    for (int c = 0; c < K; c++)
    {
        printf("C%d = (%f, %f)\n", c, centroids[2 * c], centroids[2 * c + 1]);
    }

    free(points);
    return 0;
}