from mpi4py import MPI
import numpy as np
import time

def compute_pi(N):
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Compute number of intervals handled by each process
    local_n = N // size + (rank < N % size)
    start = rank * local_n
    end = start + local_n

    start_time = time.time()  # Start timing on each process

    # Calculate local integral
    x = np.linspace(start/N, end/N, local_n, endpoint=False)
    local_sum = np.sum(np.sqrt(1 - x**2)) * (1.0 / N)

    # Gather all local integrals to the root process
    pi_approx = 4 * comm.reduce(local_sum, op=MPI.SUM, root=0)

    end_time = time.time()  # End timing on each process

    # Root process prints the result
    if rank == 0:
        print("Pi with MPI:", pi_approx)
        print(f"Time taken (root process): {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    N = 10_500_000
    compute_pi(N)

