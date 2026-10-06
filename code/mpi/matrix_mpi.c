#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define N 4000

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Matrix size %d must be divisible by process count %d.\n", N, size);
        MPI_Finalize();
        return 1;
    }

    const int rows_per_process = N / size;
    const size_t full_size = (size_t)N * N;
    const size_t local_size = (size_t)rows_per_process * N;

    double *A = NULL;
    double *C = NULL;
    double *B = (double *)malloc(full_size * sizeof(double));
    double *local_A = (double *)malloc(local_size * sizeof(double));
    double *local_C = (double *)malloc(local_size * sizeof(double));

    if (B == NULL || local_A == NULL || local_C == NULL) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
        free(B);
        free(local_A);
        free(local_C);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Initializing %d x %d matrices...\n", N, N);
        A = (double *)malloc(full_size * sizeof(double));
        C = (double *)malloc(full_size * sizeof(double));

        if (A == NULL || C == NULL) {
            fprintf(stderr, "Rank 0: memory allocation failed.\n");
            free(A);
            free(C);
            free(B);
            free(local_A);
            free(local_C);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (size_t i = 0; i < full_size; ++i) {
            A[i] = 1.0;
            B[i] = 1.0;
            C[i] = 0.0;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Bcast(B, (int)full_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, (int)local_size, MPI_DOUBLE,
                local_A, (int)local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname));

    printf("Rank %d on %s computing %d rows\n",
           rank, hostname, rows_per_process);

    for (int i = 0; i < rows_per_process; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += local_A[(size_t)i * N + k] *
                       B[(size_t)k * N + j];
            }
            local_C[(size_t)i * N + j] = sum;
        }
    }

    MPI_Gather(local_C, (int)local_size, MPI_DOUBLE,
               C, (int)local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;

    if (rank == 0) {
        printf("\nMPI Matrix Multiplication Completed\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Number of MPI Processes = %d\n", size);
        printf("Execution Time = %.6f seconds\n", elapsed);
        printf("Verification C[0][0] = %.2f\n", C[0]);
    }

    free(A);
    free(B);
    free(C);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}
