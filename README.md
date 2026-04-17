# RTM-inverse-MLP


# RTM Inverse Problem — MLP + PCA

Surrogate model replacing the Genetic Algorithm (GA) from Mtibaa et al. (2024).

## Objective
Given a desired particle distribution profile q(x), predict the optimal 
injection concentration C₀(t) using a MLP neural network.

## Results vs Article GA

| Profile | GA Err max | MLP Err max | GA RMSE  | MLP RMSE |
|---------|-----------|-------------|----------|----------|
| UD      | 2.93%     | 0.69%       | 0.00237  | 0.00033  |
| FG-O    | 0.86%     | 0.97%       | 0.000722 | 0.00038  |
| FG-X    | 6.8%      | 2.97%       | 0.00385  | 0.00289  |
| FG-NI   | 8%        | 3.40%       | 0.00457  | 0.00106  |

## Global Metrics
- RMSE : 0.00489
- MAE  : 0.00146
- R²   : 0.99543

## Pipeline
q(x) → StandardScaler → PCA (12 modes) → MLP (256→128→64) → PCA inverse (10 modes) → StandardScaler inverse → C₀(t)

## Dataset
- 3000 simulations via LHS sampling
- Input  : particle distribution q(x) — 301 nodes
- Output : optimal C₀(t) — 301 time steps

## Reference
Mtibaa et al. (2024) — Int. J. Adv. Manuf. Technol. 134:75–95
https://doi.org/10.1007/s00170-024-14074-w