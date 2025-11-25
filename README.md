[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17716380.svg)](https://doi.org/10.5281/zenodo.17716380) 

# Geometric Shadow Inversion (GSI)

**Real-Time Quantum State Tomography via Kernel Regression**

## Abstract

Quantum state tomography (QST) is the bottleneck for characterizing intermediate-scale quantum devices. Standard Maximum Likelihood Estimation (MLE) becomes computationally intractable and statistically unstable when data is sparse. **Geometric Shadow Inversion (GSI)** is a reconstruction framework that treats quantum state recovery as kernel regression on the density matrix manifold.

By utilizing a Gaussian-weighted adjoint projection, GSI triangulates state geometry in a single non-iterative pass. We identify an **Information Threshold** at measurement ratio $M/P \approx 1.17$, above which GSI enables robust, high-fidelity reconstruction ($F > 0.99$) independent of bandwidth tuning.

**Paper:** Moralez, M. (2025). Geometric Shadow Inversion: Real-Time Quantum State Tomography at the Information Threshold (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.17716380

## Features

- **Fast:** Reconstructs 5-qubit states in $\approx 10$ms and 6-qubit states in $\approx 5$min (CPU).
    
- **Robust:** Operates in the under-determined regime ($M/P < 1$) where linear inversion fails.
    
- **Physical:** Guarantees positive semi-definite (PSD) output via geometric projection.
    
- **No-MLE:** Eliminates iterative optimization and convergence issues.
    

## Installation

1. Clone the repository:
    
    ```
    git clone [https://github.com/MarcosMoralez/gsi-tomography.git](https://github.com/MarcosMoralez/gsi-tomography.git)
    cd gsi-tomography
    ```
    
2. Install dependencies:
    
    ```
    pip install -r requirements.txt
    ```
    

## Usage

```
import numpy as np
from gsi_core import GeometricShadowTomography, generate_random_mixed_state

# 1. Setup System
qubits = 4
engine = GeometricShadowTomography(qubits, sigma=0.78)

# 2. Generate Truth & Simulate Measurements
true_rho = generate_random_mixed_state(2**qubits)
data = engine.measure_state(true_rho, num_settings=500, shots=1000)

# 3. Reconstruct (Single Pass)
rho_est = engine.reconstruct(data)

# 4. Validate
fidelity = engine.fidelity(true_rho, rho_est)
print(f"Reconstruction Fidelity: {fidelity:.4f}")
```

## Reproducing Paper Results

To reproduce the scaling benchmark (Table I in the paper) for 3, 4, 5, and 6 qubits:

```
python benchmarks/reproduce_paper_results.py
```

To generate "The MLE Trap" visualization (Figure 1):

```
python figures/plot_mle_trap.py
```

## Citation

If you use this code in your research, please cite our work:

```
@article{Moralez2025GSI,
  title={Geometric Shadow Inversion: Real-Time Quantum State Tomography at the Information Threshold},
  author={Moralez, Marcos T.},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17716380},
  url={https://doi.org/10.5281/zenodo.17716380}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.

**Note:** This software is the subject of a pending patent application.
