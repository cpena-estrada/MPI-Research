#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <assert.h>

// Function to compute the sum of numbers from start to end
long long compute_sum(long long start, long long end) {
    long long sum = 0;
    for (long long i = start; i <= end; i++) {
        sum += i;
    }
    return sum;
}

int main(int argc, char** argv) {
    const long long start_num = 1;
    const long long end_num = 1000000;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Number of elements per process
    int elements_per_process;
    if (argc < 2) {
        if (world_rank == 0) {
            printf("Usage: %s elements_per_process\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    } else {
        elements_per_process = atoi(argv[1]);
    }

    // Calculate the start and end indices for this process
    long long start_index = start_num + world_rank * elements_per_process;
    long long end_index = start_index + elements_per_process - 1;

    // Ensure that the last process handles the remaining elements
    if (world_rank == world_size - 1) {
        end_index = end_num;
    }

    // Record the start time for the entire program
    clock_t start_total_time = clock();

    // Compute the sum of numbers for this process
    long long local_sum = compute_sum(start_index, end_index);

    // Gather all partial sums to compute the total sum
    long long total_sum;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // Compute the average of the total sum
        double average = (double)total_sum / (end_num - start_num + 1);

        // Print the average
        printf("Average of sum of numbers from %lld to %lld is %f\n", start_num, end_num, average);
    }

    // Finalize the MPI environment
    MPI_Finalize();

    // Record the end time for the entire program
    clock_t end_total_time = clock();

    // Calculate the elapsed time for the entire program in milliseconds
    double elapsed_total_time = ((double)(end_total_time - start_total_time)) / CLOCKS_PER_SEC * 1000;

    // Print out the elapsed time for the entire program in milliseconds
    if (world_rank == 0) {
        printf("Total elapsed time: %f milliseconds\n", elapsed_total_time);
    }

    return 0;
}
