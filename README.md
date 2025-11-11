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
| **Coverage** | Precision-like | 1.0 | Synthetic points fall within real distribution |
|  | Recall-like | 1.0 | Real points represented in synthetic manifold |
| **Privacy** | Mean NN Distance | 0.22 | No synthetic data overlaps real points |
|  | Min NN Distance | 0.11 | Safe minimal similarity |
|  | MIA AUC | 1.0 | Excellent privacy protection |
| **Utility** | TSTR AUC | 1.0 | Synthetic training transfers perfectly to real |
|  | TRTS AUC | 1.0 | Real training transfers perfectly to synthetic |

---

## Metric Definitions

### **Fidelity**
Measures how close synthetic data is to real data.
- **Jensen Shannon Divergence (JSD):** Overlap between distributions, lower is better.
- **Kolmogorov Smirnov (KS):** Difference in cumulative distributions.
- **Wasserstein Distance:** "Effort" needed to transform one distribution into another.
- **Correlation Distance:** Difference between correlation matrices of real and synthetic data.

### **Coverage**
PRDC-like metrics describing manifold overlap.
- **Precision:** Synthetic data within the real data’s neighborhood.
- **Recall:** How well synthetic data represents all real patterns.

### **Privacy**
Quantifies distance and distinguishability.
- **Nearest Neighbor Distance:** Large values indicate privacy.
- **Membership Inference Attack (MIA):** Measures re-identification risk (AUC = 1.0 → perfectly safe).

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
├── src/
│   └── autocurator/
│       ├── cli.py
│       ├── loaders.py
│       ├── preprocess.py
│       ├── viz.py
│       └── metrics/
│           ├── fidelity.py
│           ├── coverage.py
│           ├── utility.py
│           └── privacy.py
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

### Step 1, Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2, Run Benchmark
```bash
python -m autocurator.cli   --real data/real.csv   --synthetic data/synthetic.csv   --target target   --task classification   --out_dir outputs/runs/example_run   --report reports/example_run.html
```

### Step 3, View Results
```bash
start reports/example_run.html
```

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
| **Privacy** | ★★★★★ | No re-identification risk |
| **Utility** | ★★★★★ | Predictive patterns perfectly preserved |

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
  "coverage": {"precision_like": 1.0, "recall_like": 1.0},
  "privacy": {"syn_to_real_mean_nnd": 0.22, "syn_to_real_min_nnd": 0.11, "mia_auc_distance": 1.0},
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
- **PyYAML** for configuration management  
