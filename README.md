# Autocurator: Synthetic Data Benchmark Evaluator

## Overview
**Autocurator** is a modular evaluation toolkit that benchmarks **synthetic vs. real tabular data**.  
It computes quantitative metrics for *fidelity*, *coverage*, *privacy*, and *utility*, then visualizes results through a detailed HTML report and publication-ready figures.

This benchmark analyzes a small dataset of customer attributes (`age`, `income`, `score`, `visits`, `target`) to assess how closely synthetic samples resemble real data.
 
---
 
## Evaluation Summary

| Category | Metric | Value | Interpretation |
|:--|:--|:--:|:--|
| **Fidelity** | Mean JSD | 0.475 | Moderate divergence between histograms |
|  | Mean KS | 0.12 | Slight difference in cumulative distributions |
|  | Mean Wasserstein | 290.8 | Variation in scale between features |
|  | Correlation Distance | 0.0065 | Very high structure preservation |
| **Coverage** | Precision | 1.0 | Synthetic points inside a real neighborhood |
|  | Recall | 1.0 | Real points inside a synthetic neighborhood |
|  | Density | 1.16 | Avg. real neighborhoods each synthetic point falls in (÷k) |
|  | Coverage | 1.0 | Real points with a synthetic neighbor nearby |
| **Privacy** | Mean NN Distance | 0.29 | Average synthetic-to-real nearest-neighbor distance |
|  | Min NN Distance | 0.11 | Closest any synthetic point sits to a real point |
|  | MIA AUC | 0.04 | Far from 0.5 → real and synthetic are distinguishable; below 0.5 signals memorization (see note) |
| **Utility** | TSTR AUC | 1.0 | Synthetic training transfers to real |
|  | TRTS AUC | 1.0 | Real training transfers to synthetic |

> **Note on the bundled example.** The toy `synthetic.csv` is a lightly perturbed copy of `real.csv`. The corrected membership-inference metric (AUC ≈ 0.0) correctly flags this near-duplication as a privacy risk, and the perfect utility scores reflect that the two sets are almost identical. Treat it as a wiring example, not as a model of good synthetic data.

---

## Metric Definitions

### **Fidelity**
Measures how close synthetic data is to real data.
- **Jensen Shannon Divergence (JSD):** Overlap between distributions, lower is better.
- **Kolmogorov Smirnov (KS):** Difference in cumulative distributions.
- **Wasserstein Distance:** "Effort" needed to transform one distribution into another.
- **Correlation Distance:** Difference between correlation matrices of real and synthetic data.

### **Coverage**
PRDC metrics (Naeem et al., 2020) using per-point k-nearest-neighbor hyperspheres.
- **Precision:** Fraction of synthetic points inside a real hypersphere.
- **Recall:** Fraction of real points inside a synthetic hypersphere.
- **Density:** Average number of real hyperspheres each synthetic point falls into, normalized by `k` (robust to outliers).
- **Coverage:** Fraction of real points with at least one synthetic point nearby.

### **Privacy**
Quantifies distance and distinguishability.
- **Nearest Neighbor Distance:** Larger synthetic-to-real distances indicate less overlap.
- **Membership Inference Attack (MIA):** Distance-based re-identification proxy. An AUC near **0.5** means real and synthetic are indistinguishable by nearest-neighbor distance (low risk). Values far from 0.5 mean they are distinguishable; **below 0.5** indicates synthetic points sit unusually close to real ones (possible memorization / higher risk). Real points are compared against their nearest *other* real point so self-matches don't make the attack trivially perfect.
- **Holdout MIA (optional):** If you pass `--holdout` a disjoint real set the generator never saw, `mia_auc_holdout` runs the stronger distance-to-closest-record attack: members vs. non-members scored by proximity to synthetic. **0.5 = no leakage**; above 0.5 means training records are closer to synthetic than holdout records.

### **Utility**
Assesses whether synthetic data supports the same predictions.
- **TSTR:** Train on synthetic, test on real.
- **TRTS:** Train on real, test on synthetic.

---

## Architecture Overview
```
┌───────────────┐
│   real.csv    │
│ synthetic.csv │
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Data Loader  │ → Align schema, preprocess columns
└──────┬────────┘
       ▼
┌───────────────┐
│ Metrics Suite │ → Fidelity, Coverage, Privacy, Utility
└──────┬────────┘
       ▼
┌───────────────┐
│ Visualization │ → PCA, Histograms, Heatmaps
└──────┬────────┘
       ▼
┌───────────────┐
│  HTML Report  │ → Jinja2 templated dashboard
└───────────────┘
```

---

## File Structure

```
autocurator/
├── data/
│   ├── real.csv
│   └── synthetic.csv
├── outputs/
│   └── runs/example_run/
│       ├── metrics.json
│       └── plots/
│           ├── pca.png
│           ├── distributions.png
│           └── correlations.png
├── reports/
│   └── example_run.html
├── configs/
│   └── example.yaml
├── src/
│   └── autocurator/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── loaders.py
│       ├── preprocess.py
│       ├── viz.py
│       ├── report.py
│       ├── templates/
│       │   └── report.html
│       └── metrics/
│           ├── __init__.py
│           ├── fidelity.py
│           ├── coverage.py
│           ├── utility.py
│           └── privacy.py
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_metrics.py
│   └── test_pipeline.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Figures and Analysis

### PCA Projection
PCA projection shows global similarity, blue (real) and orange (synthetic) points overlap strongly.

<img width="600" height="500" alt="pca" src="https://github.com/user-attachments/assets/528897b6-ab78-4794-98ec-c680f0674287" />

---

### Feature Distributions
Histograms show per-feature alignment. Minor scale variance indicates good distribution diversity.

<img width="700" height="1100" alt="distributions" src="https://github.com/user-attachments/assets/b701b7c9-70e0-4223-b09b-e8412f95b21c" />

---

### Correlation Heatmaps
Correlation matrices of real and synthetic datasets are almost identical, confirming strong structural fidelity.

<img width="1000" height="400" alt="correlations" src="https://github.com/user-attachments/assets/8253c295-67d2-4db8-8a34-333984a024cb" />

---

## How to Run

### Step 1, Install
The package uses a `src/` layout, so install it (editable) to put `autocurator`
on your path. This also pulls in the dependencies from `pyproject.toml`.
```bash
pip install -e .
```
Add the tooling used by CI with `pip install -e ".[dev]"`.

### Step 2, Run Benchmark
```bash
autocurator \
  --real data/real.csv \
  --synthetic data/synthetic.csv \
  --target target \
  --task classification \
  --out_dir outputs/runs/example_run \
  --report reports/example_run.html
```
`python -m autocurator.cli ...` works too once the package is installed.

**Optional flags**

- `--k N` — neighborhood size for the PRDC coverage metrics (default 5).
- `--utility_model {linear,rf}` — utility estimator: linear baseline (default) or
  a random forest that captures non-linear structure.
- `--holdout path.csv` — a disjoint real set the generator never saw; enables the
  stronger holdout-based membership-inference attack (`mia_auc_holdout`).

**Config file.** Any option can be set in a YAML config; CLI arguments override it.
```bash
autocurator --config configs/example.yaml
```

### Step 3, View Results
Open `reports/example_run.html` in a browser (image links resolve relative to the
report's location, so the report and its `--out_dir` plots can live in different folders).

---

## Outputs

| File | Description |
|:--|:--|
| `metrics.json` | Numeric benchmark results |
| `pca.png` | PCA projection plot |
| `distributions.png` | Overlapping feature histograms |
| `correlations.png` | Correlation heatmaps |
| `example_run.html` | Full report (HTML) |

---

## Interpretation Summary

| Aspect | Rating | Insights |
|:--|:--:|:--|
| **Fidelity** | ★★★★☆ | Minor histogram divergence; strong structural alignment |
| **Coverage** | ★★★★★ | Real and synthetic fully overlap |
| **Privacy** | ★★☆☆☆ | Bundled example is near-duplicative; corrected MIA flags the overlap |
| **Utility** | ★★★★★ | Predictive patterns preserved (example sets are nearly identical) |

---

## Use Cases

1. **Synthetic Data Validation**, verify realism and structure before model training.  
2. **Data Sharing Compliance**, ensure privacy before external release.  
3. **AI Governance Audits**, demonstrate data safety quantitatively.  
4. **Academic Research**, evaluate synthetic generators (VAE, GAN, Copula, Diffusion).  
5. **Pipeline QA**, assess model robustness using synthetic inputs.

---

## Future Enhancements

- Add **CTGAN** and **Diffusion** benchmark support.  
- Implement **multivariate Wasserstein** and Earth Mover’s Distance.  
- Integrate **differential privacy accounting** for stronger guarantees.  
- Develop a **Streamlit dashboard** for real-time visual inspection.  
- Extend to **mixed categorical embeddings** for complex datasets.

---

## Example Metrics JSON

```json
{
  "fidelity": {
    "per_feature_mean_jsd": 0.475,
    "per_feature_mean_ks": 0.12,
    "per_feature_mean_wasserstein": 290.8,
    "correlation_distance": 0.0065
  },
  "coverage": {"precision": 1.0, "recall": 1.0, "density": 1.16, "coverage": 1.0},
  "privacy": {"syn_to_real_mean_nnd": 0.29, "syn_to_real_min_nnd": 0.13, "mia_auc_distance": 0.04},
  "utility": {"TSTR_AUC": 1.0, "TRTS_AUC": 1.0}
}
```

---

## Tech Stack

- **Python 3.10+**
- **NumPy**, **Pandas**, **SciPy**
- **Scikit-learn** for statistical modeling  
- **Matplotlib** + **Seaborn** for visualization  
- **Jinja2** for templated HTML reports  
- **PyYAML** for optional `--config` files (`configs/example.yaml`)
