from mpi4py import MPI
import numpy as np
import sys

def gaussian_elimination(matrix, n, rank, size):
    for k in range(n):
        if rank == k % size:
            MPI.COMM_WORLD.Bcast(matrix[k, :], root=k % size)
        else:
            MPI.COMM_WORLD.Bcast(matrix[k, :], root=k % size)

        for i in range(k + 1, n):
            if i % size == rank:
                factor = matrix[i, k] / matrix[k, k]
                matrix[i, k:] -= factor * matrix[k, k:]

def back_substitution(matrix, solution, n):
    for i in range(n - 1, -1, -1):
        solution[i] = matrix[i, n]
        for j in range(i + 1, n):
            solution[i] -= matrix[i, j] * solution[j]
        solution[i] /= matrix[i, i]

def main():
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: mpirun -np <num_processes> python gauss_elimination_with_logging.py <matrix_size>")
        sys.exit()

    n = int(sys.argv[1])  # Matrix size
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    if rank == 0:
        matrix = np.random.rand(n, n + 1) * 100
    else:
        matrix = None

    matrix = MPI.COMM_WORLD.bcast(matrix, root=0)

    start_time = MPI.Wtime()
    gaussian_elimination(matrix, n, rank, size)
    end_time = MPI.Wtime()

    if rank == 0:
        upper_matrix = np.empty((n, n + 1), dtype=np.float64)
    else:
        upper_matrix = None

    MPI.COMM_WORLD.Reduce(matrix, upper_matrix, op=MPI.SUM, root=0)

    if rank == 0:
        elapsed_time = end_time - start_time
        with open('timing_results200x200.txt', 'a') as f:
            f.write(f"{size},{elapsed_time}\n")

if __name__ == "__main__":
    main()
