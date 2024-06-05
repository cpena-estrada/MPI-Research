#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

double func1(double x1, double x2) {
    return x1 + x2;
}

double func2(double x1, double x2) {
    return x1 * x2;
}

double deri_f1_x1(double x1, double x2, double h) {
    return (func1(x1 + h, x2) - func1(x1, x2)) / h;
}

double deri_f1_x2(double x1, double x2, double h) {
    return (func1(x1, x2 + h) - func1(x1, x2)) / h;
}

double deri_f2_x1(double x1, double x2, double h) {
    return (func2(x1 + h, x2) - func2(x1, x2)) / h;
}

double deri_f2_x2(double x1, double x2, double h) {
    return (func2(x1, x2 + h) - func2(x1, x2)) / h;
}

void gauss_elimination(double* matrix, double* answers, int n) {
    for (int k = 0; k < n; k++) {
        for (int j = k + 1; j < n; j++) {
            matrix[k * n + j] /= matrix[k * n + k]; // Normalize the pivot row
        }
        answers[k] /= matrix[k * n + k]; // Normalize the answer
        matrix[k * n + k] = 1.0; // Set the pivot element to 1

        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                matrix[i * n + j] -= matrix[i * n + k] * matrix[k * n + j];
            }
            answers[i] -= matrix[i * n + k] * answers[k];
            matrix[i * n + k] = 0.0; // Set the eliminated element to 0
        }
    }
}

void back_substitution(double* matrix, double* answers, double* solution, int n) {
    for (int i = n - 1; i >= 0; i--) {
        solution[i] = answers[i];
        for (int j = i + 1; j < n; j++) {
            solution[i] -= matrix[i * n + j] * solution[j];
        }
    }
}

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
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double y[2] = {15, 50};
    double x[2] = {4, 9};

    double result[2] = {y[0] - func1(x[0], x[1]),
                        y[1] - func2(x[0], x[1])};
    printf("Contents of the result array:\n");
    for (int i = 0; i < 2; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    double h = 0.1;
    double x1 = x[0];
    double x2 = x[1];

    double j11 = deri_f1_x1(x1, x2, h);
    double j12 = deri_f1_x2(x1, x2, h);
    double j21 = deri_f2_x1(x1, x2, h);
    double j22 = deri_f2_x2(x1, x2, h);

    double j[4] = {j11, j12, j21, j22};
    printf("\n%f  %f\n", j11, j12);
    printf("%f  %f\n", j21, j22);

    // Define the matrix and results
    double matrix[4] = {1, 1, 9, 4}; // Partial derivatives in the form of a 2x2 matrix
    double answers[2] = {2, 14}; // The right-hand side of the equations

    // Perform Gaussian elimination and back substitution
    gauss_elimination(matrix, answers, 2);

    double solution[2];
    back_substitution(matrix, answers, solution, 2);

    // Print the solutions
    printf("Solution:\n");
    print_solution(solution, 2);

    MPI_Finalize();
    return 0;
}
