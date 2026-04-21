# RTM Inverse Problem — Surrogate Model (MLP + PCA)

> Solving the inverse problem of Resin Transfer Molding using a Machine Learning surrogate model — orders of magnitude faster than a Genetic Algorithm, with better accuracy.

---

## Overview

In Resin Transfer Molding (RTM), controlling the final particle distribution inside a composite part is critical. The classical approach couples a **Genetic Algorithm (GA)** with a numerical RTM solver to find the optimal injection concentration profile $C_0(t)$ — but this requires thousands of costly simulations per target profile.

This work replaces the GA with a **Multi-Layer Perceptron (MLP)** trained on Latin Hypercube Sampling (LHS) data, combined with **Principal Component Analysis (PCA)** for dimensionality reduction. The result is a surrogate model that solves the inverse problem in **under 1 millisecond**, with better accuracy than the GA baseline from Mtibaa et al. (2024).

---

## Problem Statement

Given a desired final particle distribution $q_{\text{desired}}(x)$, find the optimal injection concentration profile $C_0(t)$ such that:

$$q(C_0, x) \approx q_{\text{desired}}(x)$$

where $q(C_0, x) = (\varepsilon C + \sigma)(x, T_f)$ is the particle distribution obtained after RTM simulation.

---

## ML Pipeline

$$q(x) ---> \{\text{StandardScaler\} --> \{\text{PCA}\} (k_x=12) ----> \{\text{MLP}\} --->\{\text{PCA}^{-1}\} (k_y=10) ---> \{\text{Scaler}^{-1}\} C_0(t)$$

---

## Dataset

Training data is generated from a **1D RTM numerical model** (MATLAB) via Latin Hypercube Sampling. The injection profile $C_0(t)$ is parameterized as a cubic spline with control points sampled at three complexity levels:

| Type    | Control points | Samples |
|---------|---------------|---------|
| Simple  | 3 nodes       | 1 000   |
| Medium  | 6 nodes       | 1 000   |
| Complex | 10 nodes      | 1 000   |

**Total: 3 000 simulations**

- Input $Q_p$ — final particle distribution $q(x, T_f)$, shape `(3000, 301)`
- Output $C_0$ — optimal injection concentration profile, shape `(3000, 301)`

> The dataset (`.npy` files) and trained model (`.keras`) are not included in this repository due to file size. Contact the author or reproduce them using the MATLAB RTM solver described in Mtibaa et al. (2024).

---

## Model Architecture

| Layer   | Units | Activation | Regularization    |
|---------|-------|------------|-------------------|
| Input   | 12    | —          | —                 |
| Dense 1 | 256   | Swish      | L2 ($10^{-4}$)   |
| Dense 2 | 128   | Swish      | L2 ($10^{-4}$)   |
| Dense 3 | 64    | Swish      | L2 ($10^{-4}$)   |
| Output  | 10    | Linear     | —                 |

**Trainable parameters:** 45,130

### Training Configuration

| Parameter              | Value                          |
|------------------------|-------------------------------|
| Optimizer              | Adam                          |
| Loss                   | Huber                         |
| Batch size             | 32                            |
| Max epochs             | 4 000                         |
| Learning rate schedule | Cosine decay $10^{-3} \to 0$ |
| Early stopping patience| 800 epochs                    |

---

## Results

### Global Metrics (Test Set — 450 samples)

| Metric | Value   |
|--------|---------|
| RMSE   | 0.00506 |
| MAE    | 0.00117 |
| R²     | 0.99512 |

### MLP vs Genetic Algorithm — Target FGM Profiles

| Profile | GA Error max | MLP Error max | GA RMSE  | MLP RMSE | Winner |
|---------|-------------|---------------|----------|----------|--------|
| UD      | 2.93%       | **0.62%**     | 0.00237  | **0.00042** | ✅ MLP |
| FG-O    | 0.86%       | **0.57%**     | 0.000722 | **0.00048** | ✅ MLP |
| FG-X    | 6.8%        | **1.78%**     | 0.00385  | **0.00086** | ✅ MLP |
| FG-NI   | 8.0%        | **2.82%**     | 0.00457  | **0.00087** | ✅ MLP |

### Computational Cost

| Method            | Cost per profile                          |
|-------------------|-------------------------------------------|
| Genetic Algorithm | 270 – 15 000 iterations × RTM simulation |
| **MLP inference** | **< 1 millisecond**                       |

---

## Requirements

```bash
pip install numpy scipy scikit-learn tensorflow matplotlib seaborn joblib
```

| Package      | Tested version |
|--------------|---------------|
| Python       | 3.10+         |
| TensorFlow   | 2.15+         |
| scikit-learn | 1.4+          |
| NumPy        | 1.26+         |
| SciPy        | 1.12+         |

---

## Organisation

Open and run `SimLHS_inverse.ipynb` cell by cell.

The notebook is organized as follows:

1. **Data Loading** — load `Q_p.npy` and `C_0.npy`
2. **Preprocessing** — train/test split, StandardScaler, PCA
3. **Model Training** — build and train the MLP (or load `best_inverse_model.keras`)
4. **Evaluation** — global metrics and per-simulation R² distribution
5. **Visualization** — predicted vs ground truth on 12 random test samples
6. **Inference** — predict optimal $C_0(t)$ for UD, FG-O, FG-X, FG-NI profiles
7. **Comparison** — MLP vs GA error and RMSE on all four profiles

---

## Reference

Mtibaa M, Saouab A, El Moumen A, Bouaziz S, El Hami A, Haddar M (2024).
*Contribution to manufacturing control of particle-filled composites by RTM process.*
The International Journal of Advanced Manufacturing Technology, 134:75–95.
https://doi.org/10.1007/s00170-024-14074-w

---

## License

This project is released for academic and research purposes.
