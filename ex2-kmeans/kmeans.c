#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#define MAX_POINTS 10000000 // maximum number of points allowed

// Compute squared Euclidean distance (no sqrt for speed)
static inline float dist2(float px, float py, float cx, float cy)
{
    float dx = px - cx;
    float dy = py - cy;
    return dx * dx + dy * dy;
}

// Load points from a CSV file
int load_csv(const char *filename, float **out_points)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        printf("Error: cannot open file %s\n", filename);
        return -1;
    }

    float *points = malloc(MAX_POINTS * 2 * sizeof(float));
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

// Assign each point to its nearest centroid
void assign_clusters_cpu(
    const float *points,    // size N*2
    const float *centroids, // size K*2
    int *labels,            // size N
    int N, int K)
{
    for (int i = 0; i < N; i++)
    {

        float px = points[2 * i];
        float py = points[2 * i + 1];

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

        labels[i] = best_c;
    }
}

// Recompute centroids and return total centroid movement
float update_centroids_cpu(
    const float *points,
    float *centroids,
    const int *labels,
    int N, int K)
{
    float *sum = calloc(K * 2, sizeof(float));
    int *count = calloc(K, sizeof(int));

    // Accumulate positions of points per cluster
    for (int i = 0; i < N; i++)
    {
        int c = labels[i];
        sum[2 * c] += points[2 * i];
        sum[2 * c + 1] += points[2 * i + 1];
        count[c]++;
    }

    // for (int c = 0; c < K; c++)
    // {
    //     printf("Cluster %d: SumX = %f, SumY = %f, Count = %d\n", c, sum[2 * c], sum[2 * c + 1], count[c]);
    // }

    // Compute new centroids + track movement for convergence
    float movement = 0.0f;

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
        movement += dx * dx + dy * dy;

        centroids[2 * c] = newx;
        centroids[2 * c + 1] = newy;
    }

    free(sum);
    free(count);
    return movement;
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
void kmeans_cpu(
    float *points,
    float *centroids,
    int N, int K,
    int max_iters,
    float epsilon)
{
    int *labels = malloc(N * sizeof(int));
    double *iter_times = malloc(max_iters * sizeof(double));
    int **all_labels = malloc(max_iters * sizeof(int *));
    float **all_centroids = malloc(max_iters * sizeof(float *));
    for (int i = 0; i < max_iters; i++)
    {
        all_labels[i] = malloc(N * sizeof(int));
        all_centroids[i] = malloc(K * 2 * sizeof(float));
    }
    int actual_iters = 0;

    for (int it = 0; it < max_iters; it++)
    {
        // Save centroids before this iteration
        memcpy(all_centroids[it], centroids, K * 2 * sizeof(float));

        clock_t start = clock();

        assign_clusters_cpu(points, centroids, labels, N, K);
        float movement = update_centroids_cpu(points, centroids, labels, N, K);

        clock_t end = clock();
        double iter_time = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // in milliseconds
        iter_times[it] = iter_time;
        actual_iters++;

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

    // Free memory
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
    float centroids[8] = {0, 0, 5, 5, 10, 10}; // simple initial seeds

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