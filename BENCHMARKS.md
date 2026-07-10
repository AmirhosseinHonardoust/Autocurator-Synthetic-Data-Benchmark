# Generator sensitivity benchmark

Each real dataset is split into members (seen by the generator) and a holdout (unseen). Metrics should reward the high-quality `resample` generator and flag the row-copying `leaky` generator via the holdout MIA.

| Dataset | Generator | Precision | Recall | Coverage | CorrDist | HoldoutMIA | Utility |
|---|---|--:|--:|--:|--:|--:|--:|
| breast_cancer | resample | 1.00 | 0.96 | 0.99 | 0.033 | 0.83 | 1.00 |
| breast_cancer | independent | 0.00 | 0.74 | 0.00 | 0.389 | 0.49 | 0.39 |
| breast_cancer | noise | 0.00 | 1.00 | 0.00 | 0.391 | 0.53 | 0.77 |
| breast_cancer | leaky | 1.00 | 0.96 | 1.00 | 0.033 | 0.83 | 1.00 |
| wine | resample | 1.00 | 0.97 | 0.99 | 0.044 | 0.81 | 1.00 |
| wine | independent | 0.38 | 0.94 | 0.28 | 0.300 | 0.50 | 0.76 |
| wine | noise | 0.00 | 0.00 | 0.00 | 0.312 | 0.51 | 0.17 |
| wine | leaky | 1.00 | 0.97 | 0.99 | 0.042 | 0.81 | 1.00 |
| diabetes | resample | 1.00 | 0.97 | 1.00 | 0.036 | 0.84 | 0.48 |
| diabetes | independent | 0.52 | 0.88 | 0.54 | 0.275 | 0.56 | 0.09 |
| diabetes | noise | 0.00 | 1.00 | 0.00 | 0.286 | 0.44 | -4.18 |
| diabetes | leaky | 1.00 | 0.96 | 1.00 | 0.036 | 0.85 | 0.48 |

Utility is TSTR AUC for classification datasets and TSTR R² for regression (diabetes). Higher precision/recall/coverage/utility and lower CorrDist are better; HoldoutMIA near 0.5 means low leakage.
