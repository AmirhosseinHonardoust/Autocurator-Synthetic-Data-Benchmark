import numpy as np, pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

def _hist(p, bins=30):
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return np.array([1.0]), np.array([0.0, 1.0])
    h, edges = np.histogram(p, bins=bins, density=True)
    h = h / (h.sum() + 1e-12)
    return h, edges

def per_feature_similarity(df_r: pd.DataFrame, df_s: pd.DataFrame, bins: int = 30):
    results = {}
    for col in df_r.columns:
        if not pd.api.types.is_numeric_dtype(df_r[col]):
            continue
        hr, _ = _hist(df_r[col].to_numpy(), bins=bins)
        hs, _ = _hist(df_s[col].to_numpy(), bins=bins)
        m = min(len(hr), len(hs))
        jsd = float(jensenshannon(hr[:m], hs[:m]))
        ks = float(ks_2samp(df_r[col].to_numpy(), df_s[col].to_numpy()).statistic)
        wd = float(wasserstein_distance(df_r[col].to_numpy(), df_s[col].to_numpy()))
        results[col] = {"JSD": jsd, "KS": ks, "Wasserstein": wd}
    return results

def correlation_distance(df_r: pd.DataFrame, df_s: pd.DataFrame):
    num_cols = [c for c in df_r.columns if pd.api.types.is_numeric_dtype(df_r[c])]
    if len(num_cols) < 2: return None
    cr = df_r[num_cols].corr().to_numpy()
    cs = df_s[num_cols].corr().to_numpy()
    return float(np.mean(np.abs(cr - cs)))
