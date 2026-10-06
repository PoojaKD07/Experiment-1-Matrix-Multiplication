#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 4000

int main(void) {
    printf("Initializing %d x %d matrices...\n", N, N);

    double *A = (double *)malloc((size_t)N * N * sizeof(double));
    double *B = (double *)malloc((size_t)N * N * sizeof(double));
    double *C = (double *)malloc((size_t)N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    omp_set_num_threads(8);
    double start = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[(size_t)i * N + k] * B[(size_t)k * N + j];
            }
            C[(size_t)i * N + j] = sum;
        }
    }

    double elapsed = omp_get_wtime() - start;

    printf("\nOpenMP Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Number of Threads Used = %d\n", omp_get_max_threads());
    printf("Execution Time = %.6f seconds\n", elapsed);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
