import numpy as np
from mpi4py import MPI

def func1(x1, x2):
    return x1 + x2

def func2(x1, x2):
    return x1 * x2

def deri_f1_x1(x1, x2, h):
    return round((func1(x1 + h, x2) - func1(x1, x2)) / h)

def deri_f1_x2(x1, x2, h):
    return round((func1(x1, x2 + h) - func1(x1, x2)) / h)

def deri_f2_x1(x1, x2, h):
    return round((func2(x1 + h, x2) - func2(x1, x2)) / h)

def deri_f2_x2(x1, x2, h):
    return round((func2(x1, x2 + h) - func2(x1, x2)) / h)

def gauss_elimination(matrix, answers):
    n = len(answers)
    for k in range(n):
        for j in range(k + 1, n):
            matrix[k, j] = round(matrix[k, j] / matrix[k, k])
        answers[k] = round(answers[k] / matrix[k, k])
        matrix[k, k] = 1.0

        for i in range(k + 1, n):
            for j in range(k + 1, n):
                matrix[i, j] -= matrix[i, k] * matrix[k, j]
            answers[i] -= matrix[i, k] * answers[k]
            matrix[i, k] = 0.0

def back_substitution(matrix, answers):
    n = len(answers)
    solution = np.zeros(n)
    for i in range(n - 1, -1, -1):
        solution[i] = answers[i]
        for j in range(i + 1, n):
            solution[i] -= matrix[i, j] * solution[j]
    return solution

def print_solution(solution):
    variables = "abcdefghijklmnopqrstuvwxyz"
    for i in range(len(solution)):
        if i < 26:
            print(f"{variables[i]} = {round(solution[i], 2):6.2f}")
        else:
            print(f"Variable {i} = {round(solution[i], 2):6.2f}")

def main():
    comm = MPI.COMM_WORLD
    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    y = np.array([15, 50])
    x = np.array([4, 9])

    result = np.array([y[0] - func1(x[0], x[1]), y[1] - func2(x[0], x[1])])
    if world_rank == 0:
        print("Contents of the result array:")
        for res in result:
            print(f"{res} ", end="")
        print()

    h = 0.1
    x1, x2 = x

    j11 = deri_f1_x1(x1, x2, h)
    j12 = deri_f1_x2(x1, x2, h)
    j21 = deri_f2_x1(x1, x2, h)
    j22 = deri_f2_x2(x1, x2, h)

    j = np.array([[j11, j12], [j21, j22]], dtype=int)
    if world_rank == 0:
        print(f"\n{j11}  {j12}")
        print(f"{j21}  {j22}")

    matrix = np.array([[1, 1], [9, 4]], dtype=float)
    answers = np.array([2, 14], dtype=float)

    gauss_elimination(matrix, answers)

    solution = back_substitution(matrix, answers)
    if world_rank == 0:
        print("Solution:")
        print_solution(solution)

if __name__ == "__main__":
    main()