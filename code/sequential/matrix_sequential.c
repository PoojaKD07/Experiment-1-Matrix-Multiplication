#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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

    clock_t start = clock();

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[(size_t)i * N + k] * B[(size_t)k * N + j];
            }
            C[(size_t)i * N + j] = sum;
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nSequential Matrix Multiplication Completed\n");
    printf("Matrix Size = %d x %d\n", N, N);
    printf("Execution Time = %.6f seconds\n", elapsed);
    printf("Verification C[0][0] = %.2f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
