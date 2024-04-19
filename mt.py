from multiprocessing import Pool
import numpy as np
import time
import cProfile

def f(i, dx):
    """Calculate a segment of the quarter circle using the midpoint rule."""
    x = i * dx
    return np.sqrt(1 - x**2) * dx

def compute_pi_multiprocessing(N, num_processes):
    """Calculate π using multiprocessing."""
    dx = 1.0 / N
    with Pool(num_processes) as pool:
        result = pool.starmap(f, [(i, dx) for i in range(N)])
    return 4 * sum(result)

def main():
    test_values = [10000, 100000, 1000000]  # Different values of N to test
    num_processes = 4  # Number of processes for multiprocessing

    results = []
    for N in test_values:
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()
        pi_approx = compute_pi_multiprocessing(N, num_processes)
        end_time = time.time()
        profiler.disable()

        execution_time = end_time - start_time
        results.append((N, pi_approx, execution_time))
        
        profiler.print_stats(sort='time')

        print(f"N = {N}: Multiprocessing π approximation = {pi_approx:.6f}, Time taken = {execution_time:.2f} seconds")

    return results

if __name__ == '__main__':
    main()
