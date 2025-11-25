import sys
import os
import numpy as np
import time

# Add parent directory to path to import gsi_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsi_core import GeometricShadowTomography, generate_random_mixed_state

def run_benchmark_sweep():
    """
    Reproduces the scaling benchmark from Table I of the paper.
    Runs GSI for 3, 4, 5, and 6 qubits.
    """
    print("\n--- GSI Scaling Benchmark (Reproduction of Table I) ---\n")
    
    # Configuration
    qubit_sweep = [3, 4, 5, 6]
    settings_map = {3: 2000, 4: 2000, 5: 2000, 6: 3000} # Settings used in paper
    shots = 1000
    trials = 3 

    print(f"{'Qubits':<8} | {'Params':<8} | {'Settings':<10} | {'Time (s)':<12} | {'Fidelity':<10}")
    print("-" * 65)

    for n in qubit_sweep:
        dim = 2**n
        params = 4**n - 1
        settings = settings_map[n]
        
        # Initialize Engine
        # Note: Initialization time is excluded from reconstruction time in the paper
        engine = GeometricShadowTomography(n, sigma=0.78)
        
        times = []
        fidelities = []
        
        for t in range(trials):
            # Generate State
            true_rho = generate_random_mixed_state(dim)
            
            # Measure
            data = engine.measure_state(true_rho, settings, shots)
            
            # Timed Reconstruction
            t0 = time.time()
            rho_est = engine.reconstruct(data)
            t1 = time.time()
            
            # Metrics
            fid = engine.fidelity(true_rho, rho_est)
            
            times.append(t1 - t0)
            fidelities.append(fid)
            
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_fid = np.mean(fidelities)
        std_fid = np.std(fidelities)
        
        print(f"{n:<8} | {params:<8} | {settings:<10} | {avg_time:.4f} ±{std_time:.2f} | {avg_fid:.4f} ±{std_fid:.4f}")

if __name__ == "__main__":
    run_benchmark_sweep()

