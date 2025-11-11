import matplotlib.pyplot as plt, seaborn as sns, numpy as np, pandas as pd
from sklearn.decomposition import PCA

def pca_scatter(df_r: pd.DataFrame, df_s: pd.DataFrame, out_path: str):
    num_cols = [c for c in df_r.columns if pd.api.types.is_numeric_dtype(df_r[c])]
    if len(num_cols) < 2: return
    Xr = df_r[num_cols].fillna(df_r[num_cols].mean()).to_numpy()
    Xs = df_s[num_cols].fillna(df_s[num_cols].mean()).to_numpy()
    Z = np.vstack([Xr, Xs])
    pca = PCA(n_components=2, random_state=42).fit(Z)
    R = pca.transform(Xr); S = pca.transform(Xs)
    plt.figure(figsize=(6,5))
    plt.scatter(R[:,0], R[:,1], s=10, alpha=0.6, label="real")
    plt.scatter(S[:,0], S[:,1], s=10, alpha=0.6, label="synthetic")
    plt.title("PCA: real vs synthetic"); plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def per_feature_hist(df_r: pd.DataFrame, df_s: pd.DataFrame, out_path: str, cols: list[str] | None = None):
    cols = cols or [c for c in df_r.columns if pd.api.types.is_numeric_dtype(df_r[c])][:6]
    if not cols: return
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.2*n))
    if n == 1: axes = [axes]
    for ax, c in zip(axes, cols):
        ax.hist(df_r[c].dropna(), bins=30, alpha=0.6, label="real", density=True)
        ax.hist(df_s[c].dropna(), bins=30, alpha=0.6, label="synthetic", density=True)
        ax.set_title(f"Distribution: {c}"); ax.legend()
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def corr_heatmaps(df_r: pd.DataFrame, df_s: pd.DataFrame, out_path: str):
    num_cols = [c for c in df_r.columns if pd.api.types.is_numeric_dtype(df_r[c])]
    if len(num_cols) < 2: return
    cr = df_r[num_cols].corr(); cs = df_s[num_cols].corr()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    sns.heatmap(cr, ax=axes[0], vmin=-1, vmax=1, cmap="coolwarm", cbar=False); axes[0].set_title("Real correlation")
    sns.heatmap(cs, ax=axes[1], vmin=-1, vmax=1, cmap="coolwarm", cbar=False); axes[1].set_title("Synthetic correlation")
    plt.tight_layout(); plt.savefig(out_path); plt.close()
