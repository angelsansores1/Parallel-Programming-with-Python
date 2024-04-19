import numpy as np
import time

def compute_pi_sequential(N):
    dx = 1.0 / N
    total = sum(np.sqrt(1 - (i * dx) ** 2) for i in range(N))
    return 4 * total * dx

N = 1000000
start_time = time.time()  # Start timing
pi_approx_sequential = compute_pi_sequential(N)
end_time = time.time()  # End timing

print(f"Sequential π approximation: {pi_approx_sequential}")
print(f"Time taken: {end_time - start_time:.4f} seconds")

