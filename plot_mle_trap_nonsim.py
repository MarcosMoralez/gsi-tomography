import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from tqdm import tqdm  # pip install tqdm
from gsi_core import GeometricShadowTomography, generate_random_mixed_state

class IterativeMLESolver:
    """
    A standard Iterative Maximum Likelihood Estimation (iMLE) solver.
    Implements the Hradil 'R rho R' algorithm used as the baseline in QST literature.
    """
    def __init__(self, num_qubits, max_iter=200, tol=1e-4):
        self.dim = 2**num_qubits
        self.max_iter = max_iter
        self.tol = tol
        self.identity = np.eye(self.dim, dtype=complex)

    def reconstruct(self, measurement_data):
        """
        Iterative MLE reconstruction.
        Complexity: O(Iterations * M * D^3) - effectively the 'slow' method.
        """
        # 1. Initialize rho as maximally mixed
        rho = self.identity / self.dim
        
        # 2. Pre-compute POVM elements (Projectors) to speed up the loop
        # This expands the measurement data into a flat list of operators
        povm_elements = []
        frequencies = []
        
        for data in measurement_data:
            U = data['unitary']
            counts = data['counts'] # Normalized frequencies
            
            # Reconstruct projectors: Pi_k = U^dagger |k><k| U
            # We can do this efficiently by rotating the standard basis
            for k, freq in enumerate(counts):
                if freq > 0: # Optimization: only track outcomes that happened
                    # Create |k> vector
                    vec = np.zeros(self.dim)
                    vec[k] = 1.0
                    
                    # Rotate: |psi> = U^dagger |k>
                    rotated_vec = U.conj().T @ vec
                    
                    # Projector: |psi><psi|
                    Pi = np.outer(rotated_vec, rotated_vec.conj())
                    
                    povm_elements.append(Pi)
                    frequencies.append(freq)

        # 3. Iterative Optimization (Hradil R-rho-R algorithm)
        for i in range(self.max_iter):
            R = np.zeros((self.dim, self.dim), dtype=complex)
            
            # R = Sum (f_i / prob_i) * Pi_i
            for f, Pi in zip(frequencies, povm_elements):
                # Calculate probability of outcome: Tr(Pi * rho)
                prob = np.real(np.trace(Pi @ rho))
                prob = max(prob, 1e-9) # Avoid division by zero
                
                weight = f / prob
                R += weight * Pi
            
            # Update step: rho_new = R * rho * R
            rho_new = R @ rho @ R
            
            # Normalize (Trace = 1)
            rho_new /= np.trace(rho_new)
            
            # Check convergence (Trace Distance)
            dist = 0.5 * np.sum(np.abs(la.eigvalsh(rho - rho_new)))
            rho = rho_new
            
            if dist < self.tol:
                break
                
        return rho

def run_dynamic_mle_trap():
    print("\n--- Generating 'The MLE Trap' (Dynamic) ---")
    print("Simulating comparisons between Iterative MLE and GSI...")
    
    # Configuration
    # We use 3 qubits to keep the MLE runtime reasonable for a demo.
    # The paper uses 5, but MLE for n=5 takes significantly longer per point.
    QUBITS = 3 
    DIM = 2**QUBITS
    SHOTS = 1000
    
    # Measurement Settings to sweep for MLE (The x-axis)
    # Low settings = Underdetermined (The Trap) -> High settings = Overdetermined
    mle_settings_sweep = [10, 20, 40, 60, 80, 100, 150, 200, 300]
    
    # Initialize Engines
    gsi_engine = GeometricShadowTomography(QUBITS, sigma=0.78)
    mle_solver = IterativeMLESolver(QUBITS)
    
    # Generate one "True" State for the experiment
    print(f"Generating random {QUBITS}-qubit mixed state...")
    true_rho = generate_random_mixed_state(DIM)
    
    # --- 1. GSI Baseline ---
    # We run GSI at a "Safe" measurement count (Information Threshold region)
    # The paper uses M=2000 for the baseline line.
    print("Calculating GSI Baseline (M=1000)...")
    gsi_data = gsi_engine.measure_state(true_rho, num_settings=1000, shots=SHOTS)
    gsi_rho = gsi_engine.reconstruct(gsi_data)
    gsi_fidelity = gsi_engine.fidelity(true_rho, gsi_rho)
    print(f"GSI Baseline Fidelity: {gsi_fidelity:.4f}")
    
    # --- 2. MLE Sweep ---
    mle_fidelities = []
    
    print("Running MLE Sweep (this may take a moment)...")
    for M in tqdm(mle_settings_sweep):
        # Generate data specifically for this M
        # Note: We generate NEW data for each point to simulate "Given M settings"
        data = gsi_engine.measure_state(true_rho, num_settings=M, shots=SHOTS)
        
        # Run Iterative MLE
        mle_rho = mle_solver.reconstruct(data)
        
        # Calculate Fidelity
        fid = gsi_engine.fidelity(true_rho, mle_rho)
        mle_fidelities.append(fid)

    # --- 3. Plotting ---
    print("Plotting results...")
    plt.figure(figsize=(8, 6), dpi=150)

    # Plot GSI Baseline (Constant Line)
    plt.axhline(gsi_fidelity, color='#1f77b4', linestyle='--', linewidth=2, 
             label=f'GSI Baseline (M=1000)')

    # Plot MLE Curve
    plt.plot(mle_settings_sweep, mle_fidelities, color='#d62728', marker='o', linewidth=2, 
             label='MLE Polish (Variable M)')

    # Highlight "The Trap"
    # Find where MLE is worse than GSI
    mle_arr = np.array(mle_fidelities)
    trap_mask = mle_arr < gsi_fidelity
    
    if np.any(trap_mask):
        # Fill strictly up to the crossover point
        plt.fill_between(mle_settings_sweep, 0, 1, 
                         where=(mle_arr < gsi_fidelity),
                         color='#d62728', alpha=0.1)
        
        # Add Text Annotations based on the data range
        mid_trap = mle_settings_sweep[len(mle_settings_sweep)//4]
        plt.text(mid_trap, min(mle_fidelities) + 0.05, "The MLE Trap\n(Overfitting/Instability)", 
                 color='#d62728', fontsize=12, fontweight='bold', ha='left')

    plt.xlabel('Number of Measurement Settings (M)', fontsize=12)
    plt.ylabel('Reconstruction Fidelity', fontsize=12)
    plt.title(f'Dynamic MLE Trap: {QUBITS}-Qubit System', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Set limits based on data to make it look nice
    plt.ylim(min(min(mle_fidelities), gsi_fidelity) - 0.05, 1.0)
    plt.xlim(0, max(mle_settings_sweep) + 10)

    filename = "dynamic_mle_trap.png"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Figure saved to {filename}")

if __name__ == "__main__":
    run_dynamic_mle_trap()