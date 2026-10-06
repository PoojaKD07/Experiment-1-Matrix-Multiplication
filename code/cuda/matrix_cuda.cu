#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define N 4000
#define BLOCK_SIZE 16

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        return 1;                                                 \
    }                                                              \
} while (0)

__global__ void matrixMultiply(const double *A, const double *B, double *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[(size_t)row * N + k] *
                   B[(size_t)k * N + col];
        }
        C[(size_t)row * N + col] = sum;
    }
}

int main(void) {
    const size_t bytes = (size_t)N * N * sizeof(double);

    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        free(h_A);
        free(h_B);
        free(h_C);
        return 1;
    }

    for (size_t i = 0; i < (size_t)N * N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 1.0;
        h_C[i] = 0.0;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    matrixMultiply<<<grid, block>>>(d_A, d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_seconds =
        std::chrono::duration<double>(total_end - total_start).count();

    printf("CUDA Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Grid Size = %d x %d blocks\n", grid.x, grid.y);
    printf("Block Size = %d x %d threads\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("Kernel Execution Time = %.6f seconds\n", kernel_ms / 1000.0f);
    printf("Total CUDA Phase Time = %.6f seconds\n", total_seconds);
    printf("Verification C[0][0] = %.2f\n", h_C[0]);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
