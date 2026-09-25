#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define N 4000

int main(int argc, char *argv[])
{
    int rank, size;
    int rows_per_process;
    double *A, *B, *C;
    double *local_A, *local_C;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            printf("Error: N must be divisible by number of processes.\n");

        MPI_Finalize();
        return 1;
    }

    rows_per_process = N / size;

    B = (double *)malloc(N * N * sizeof(double));
    local_A = (double *)malloc(rows_per_process * N * sizeof(double));
    local_C = (double *)malloc(rows_per_process * N * sizeof(double));

    if (B == NULL || local_A == NULL || local_C == NULL) {
        printf("Rank %d: Memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        C = (double *)malloc(N * N * sizeof(double));

        if (A == NULL || C == NULL) {
            printf("Memory allocation failed on rank 0.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Initializing %d x %d matrices...\n", N, N);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i * N + j] = 1.0;
                B[i * N + j] = 1.0;
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatter(
        A,
        rows_per_process * N,
        MPI_DOUBLE,
        local_A,
        rows_per_process * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    printf("Rank %d on ", rank);
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("%s computing %d rows\n", hostname, rows_per_process);

    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0.0;

            for (int k = 0; k < N; k++) {
                local_C[i * N + j] +=
                    local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    MPI_Gather(
        local_C,
        rows_per_process * N,
        MPI_DOUBLE,
        C,
        rows_per_process * N,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\nMPI Matrix Multiplication Completed\n");
        printf("Matrix Size = %d x %d\n", N, N);
        printf("Number of MPI Processes = %d\n", size);
        printf("Execution Time = %f seconds\n",
               end_time - start_time);
        printf("Verification C[0][0] = %.2f\n", C[0]);
    }

    free(B);
    free(local_A);
    free(local_C);

    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();

    return 0;
}
