import numpy as np
import scipy.linalg as la
from scipy.stats import unitary_group
from functools import reduce
import itertools
import time

class GeometricShadowTomography:
    """
    Geometric Shadow Inversion (GSI) Engine.
    
    Implements quantum state tomography via Kernel Regression on the density matrix manifold.
    References: Moralez, M. T. (2025). "Geometric Shadow Inversion..."
    """

    def __init__(self, num_qubits, sigma=0.78, alpha=3.0):
        """
        Initialize the GSI engine.
        
        Args:
            num_qubits (int): Number of qubits (n).
            sigma (float): Kernel bandwidth parameter. Default 0.78 is robust for n=3-6.
            alpha (float): Regularization scaling factor.
        """
        self.n = num_qubits
        self.dim = 2**num_qubits
        self.sigma = sigma
        self.alpha = alpha
        
        # Standard Pauli Matrices
        self.Paulis = {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }

        # Pre-compute Generalized Pauli Basis
        # Note: This scales as 4^n. For n=6, this is 4096 matrices of size 64x64.
        print(f"Initializing GSI Engine for {num_qubits} Qubits (Basis Size: {4**num_qubits - 1})...")
        self.basis_ops, _ = self._generate_operator_basis()
        
        # Flatten basis for fast vectorized dot products
        # Shape: (Num_Operators, Dim*Dim)
        self.basis_flat = self.basis_ops.reshape(len(self.basis_ops), -1)

    def _generate_operator_basis(self):
        """Generates the Generalized Pauli basis (excluding Identity)."""
        keys = ['I', 'X', 'Y', 'Z']
        all_combos = itertools.product(keys, repeat=self.n)
        ops_stack = []
        
        for combo in all_combos:
            # Skip the all-Identity operator (it's the center of the Bloch sphere)
            if all(k == 'I' for k in combo): continue
            
            matrices = [self.Paulis[k] for k in combo]
            full_op = reduce(np.kron, matrices)
            ops_stack.append(full_op)
            
        return np.array(ops_stack), None

    def _gaussian_kernel(self, overlap_metric):
        """
        The Geometric Prior.
        Weights basis operators based on their Hilbert-Schmidt distance to the shadow.
        """
        # overlap_metric is in range [-1, 1]
        # Distance d^2 ~ (1 - overlap)
        return np.exp(- (1 - overlap_metric)**2 / (2 * self.sigma**2))

    def measure_state(self, rho, num_settings, shots=1000):
        """
        Simulates the measurement process to generate empirical shadows.
        
        Args:
            rho (np.array): True density matrix.
            num_settings (int): Number of random unitary bases (M).
            shots (int): Shots per basis.
            
        Returns:
            List of dictionaries containing {'unitary': U, 'counts': frequencies}
        """
        measurements = []
        for _ in range(num_settings):
            # 1. Random Basis Rotation (Haar Random)
            U = unitary_group.rvs(self.dim)

            # 2. Project state
            rho_rot = U @ rho @ U.conj().T
            probs = np.real(np.diag(rho_rot)).copy()
            probs = np.clip(probs, 1e-10, None) # Numerical stability
            probs /= np.sum(probs)

            # 3. Sample
            counts = np.random.multinomial(shots, probs)
            freqs = counts / shots

            measurements.append({'unitary': U, 'counts': freqs})
        return measurements

    def reconstruct(self, measurement_data):
        """
        Core GSI Reconstruction Algorithm.
        O(M * 4^n * D^2) complexity. Single-pass.
        """
        t0 = time.time()
        
        # Accumulator for the weighted basis sum
        coeff_accumulator = np.zeros(len(self.basis_ops))
        
        # Regularization scaling
        regularizer = self.alpha / (len(measurement_data) * self.sigma)

        for data in measurement_data:
            U = data['unitary']
            freqs = data['counts']
            
            # 1. Construct Local Empirical Shadow
            local_rho = np.diag(freqs)
            global_shadow = U.conj().T @ local_rho @ U
            shadow_flat = global_shadow.flatten()

            # 2. Calculate Overlaps (Triangulation)
            # Dot product corresponds to Hilbert-Schmidt Inner Product Tr(A @ B)
            overlaps = np.real(np.dot(self.basis_flat.conj(), shadow_flat))
            
            # 3. Apply Kernel (Smoothness Prior)
            weights = self._gaussian_kernel(np.abs(overlaps))
            
            # 4. Accumulate
            coeff_accumulator += overlaps * weights

        # 5. Deconvolution & Basis Summation
        coeff_est = coeff_accumulator * regularizer
        # Tensor contraction: Sum(c_i * sigma_i)
        rho_correction = np.tensordot(coeff_est, self.basis_ops, axes=([0], [0]))
        
        # Add identity term (center of manifold)
        rho_recon = (np.eye(self.dim) / self.dim) + rho_correction
        
        # 6. Geometric Projection to PSD Cone (Enforce Physicality)
        final_rho = self._enforce_positivity(rho_recon)
        
        dt = time.time() - t0
        return final_rho

    def _enforce_positivity(self, rho):
        """
        Projects the matrix onto the closest Positive Semi-Definite (PSD) matrix.
        Solved analytically via Eigendecomposition.
        """
        evals, evecs = la.eigh(rho)
        evals[evals < 0] = 0 # Clip negative eigenvalues
        if np.sum(evals) > 0:
            evals /= np.sum(evals) # Renormalize Trace to 1
        else:
            evals = np.ones(len(evals)) / len(evals) # Fallback to Maximally Mixed
            
        return evecs @ np.diag(evals) @ evecs.conj().T

    @staticmethod
    def fidelity(rho1, rho2):
        """Computes Quantum Fidelity F = (Tr sqrt(sqrt(rho1) rho2 sqrt(rho1)))^2"""
        sqrt_rho1 = la.sqrtm(rho1)
        val = sqrt_rho1 @ rho2 @ sqrt_rho1
        return np.real(np.trace(la.sqrtm(val)))**2

def generate_random_mixed_state(dim):
    """Generates a random mixed density matrix (Hilbert-Schmidt ensemble)."""
    G = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = G @ G.conj().T
    rho /= np.trace(rho)
    return rho
