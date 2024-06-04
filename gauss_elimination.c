#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>


// Function to generate a random matrix and a random column for the answers
void generate_augmented_matrix(double* matrix, double* answers, int n) {
   for (int i = 0; i < n * n; i++) {
       matrix[i] = (double)rand() / RAND_MAX * 100.0; // Random numbers between 0 and 100
 }
   for (int i = 0; i < n; i++) {
       answers[i] = (double)rand() / RAND_MAX * 100.0; // Random answers between 0 and 100
   }
}


// Function to print the augmented matrix
void print_augmented_matrix(double* matrix, double* answers, int n) {
   for (int i = 0; i < n; i++) {
       for (int j = 0; j < n; j++) {
           printf("%6.2f ", matrix[i * n + j]);
       }
       printf("| %6.2f\n", answers[i]);
   }
}


// Function to perform Gaussian elimination on the augmented matrix
void gauss_elimination(double* matrix, double* answers, int n, int rank, int size) {
   for (int k = 0; k < n; k++) {
       if (k % size == rank) { // Row k is handled by process rank
           for (int j = k + 1; j < n; j++) {
               matrix[k * n + j] /= matrix[k * n + k]; // Normalize the pivot row
           }
           answers[k] /= matrix[k * n + k]; // Normalize the answer
           matrix[k * n + k] = 1.0; // Set the pivot element to 1
       }
       // Broadcast the pivot row and pivot answer to all processes
       MPI_Bcast(&matrix[k * n], n, MPI_DOUBLE, k % size, MPI_COMM_WORLD);
       MPI_Bcast(&answers[k], 1, MPI_DOUBLE, k % size, MPI_COMM_WORLD);


       for (int i = k + 1; i < n; i++) {
           if (i % size == rank) { // Row i is handled by process rank
               for (int j = k + 1; j < n; j++) {
                   matrix[i * n + j] -= matrix[i * n + k] * matrix[k * n + j];
               }
               answers[i] -= matrix[i * n + k] * answers[k];
               matrix[i * n + k] = 0.0; // Set the eliminated element to 0
           }
       }
   }
}


// Function to perform back substitution to solve for the unknowns
void back_substitution(double* matrix, double* answers, double* solution, int n) {
   for (int i = n - 1; i >= 0; i--) {
       solution[i] = answers[i];
       for (int j = i + 1; j < n; j++) {
           solution[i] -= matrix[i * n + j] * solution[j];
       }
   }
}


// Function to print the solutions
void print_solution(double* solution, int n) {
   char variables[] = "abcdefghijklmnopqrstuvwxyz";
   for (int i = 0; i < n; i++) {
       if (i < 26) { // Ensure that we do not exceed the number of letters available
           printf("%c = %6.2f\n", variables[i], solution[i]);
       } else {
           printf("Variable %d = %6.2f\n", i, solution[i]); // Fallback if n > 26
       }
   }
}


int main(int argc, char** argv) {
   // Initialize the MPI environment
   MPI_Init(&argc, &argv);


   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
   MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the number of processes


   if (argc < 2) { // Check if the matrix size is provided
       if (rank == 0) {
           printf("Usage: %s <matrix_size>\n", argv[0]);
       }
       MPI_Finalize();
       return 1;
   }


   int n = atoi(argv[1]); // Get the matrix size from the command line arguments
   srand(time(NULL) + rank); // Seed for random number generation


   // Allocate memory for the matrix and the answers
   double* matrix = (double*)malloc(n * n * sizeof(double));
   double* answers = (double*)malloc(n * sizeof(double));
   double* solution = (double*)malloc(n * sizeof(double));
   if (rank == 0) {
       generate_augmented_matrix(matrix, answers, n); // Generate the matrix and the answers with random numbers
       printf("Original augmented matrix:\n");
       print_augmented_matrix(matrix, answers, n); // Print the original augmented matrix
   }


   // Broadcast the initial matrix and the answers to all processes
   MPI_Bcast(matrix, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(answers, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);


   double start_time = MPI_Wtime(); // Record the start time
   gauss_elimination(matrix, answers, n, rank, size); // Perform Gaussian elimination
   MPI_Barrier(MPI_COMM_WORLD); // Ensure all processes complete elimination


   // Only the root process performs back substitution
   if (rank == 0) {
       back_substitution(matrix, answers, solution, n); // Perform back substitution
       printf("Solution:\n");
       print_solution(solution, n); // Print the solution
   }


   double end_time = MPI_Wtime(); // Record the end time


   if (rank == 0) {
       printf("Upper triangular matrix:\n");
       print_augmented_matrix(matrix, answers, n); // Print the upper triangular matrix
       printf("Time taken: %f seconds\n", end_time - start_time); // Print the time taken
   }


   free(matrix); // Free the allocated memory
   free(answers); // Free the allocated memory
   free(solution); // Free the allocated memory
   MPI_Finalize(); // Finalize the MPI environment
   return 0;
}
