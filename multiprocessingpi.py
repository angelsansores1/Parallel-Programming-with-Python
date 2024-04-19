from multiprocessing import Pool
import numpy as np
import time

def f(i, dx):
    x = i * dx
    return np.sqrt(1 - x**2) * dx

def compute_pi_multiprocessing(N, num_processes):
    dx = 1.0 / N
    with Pool(num_processes) as pool:
        result = pool.starmap(f, [(i, dx) for i in range(N)])
    return 4 * sum(result)

def test_different_Ns():
    test_values = [10000, 100000, 1000000]  # Example values for N
    num_processes = 4  # Example number of processes
    results = []
    
    for N in test_values:
        start_time = time.time()
        pi_approx = compute_pi_multiprocessing(N, num_processes)
        end_time = time.time()
        
        execution_time = end_time - start_time
        results.append((N, pi_approx, execution_time))
        
        print(f"N = {N}: Multiprocessing π approximation = {pi_approx:.6f}, Time taken = {execution_time:.2f} seconds")

    return results

# Execute the test function
if __name__ == '__main__':
    test_results = test_different_Ns()


