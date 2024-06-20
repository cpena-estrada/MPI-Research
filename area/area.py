from mpi4py import MPI
import numpy as np

# Define the function 4 / (1 + x^2)
def func(x):
    return 4 / (1 + x * x)

# Function to approximate the integral using the rectangle method
def rectangle_method(a, b, interval, rank, size):
    n = int((b - a) / interval)  # Total number of subintervals
    local_sum = 0.0

    # Each process calculates its part of the integral
    for i in range(rank, n, size):
        x = a + (i + 0.5) * interval  # Midpoint of each interval
        local_sum += func(x)

    return local_sum * interval  # Multiply by the width of the rectangles

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Initialize variables
    a = b = interval = 0.0
    global_sum = 0.0

    # Only rank 0 will get the user input
    if rank == 0:
        # Get user input for the limits of integration
        a = float(input("Enter the lower limit of integration (a): "))
        b = float(input("Enter the upper limit of integration (b): "))
        interval = float(input("Enter the interval between each point: "))

    # Broadcast the input values to all processes
    a = comm.bcast(a, root=0)
    b = comm.bcast(b, root=0)
    interval = comm.bcast(interval, root=0)

    # Start the timer
    start_time = MPI.Wtime()

    # Calculate the local integral
    local_sum = rectangle_method(a, b, interval, rank, size)

    # Reduce all local sums to a global sum
    global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

    # Stop the timer
    end_time = MPI.Wtime()

    # Only rank 0 will print the result and log the elapsed time
    if rank == 0:
        elapsed_time = end_time - start_time
        print(f"The approximate area under the curve from {a:.2f} to {b:.2f} is: {global_sum:.5f}")
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
        
        # Log the elapsed time
        with open("timing_log.txt", "a") as log_file:
            log_file.write(f"{size},{elapsed_time}\n")

if __name__ == "__main__":
    main()
