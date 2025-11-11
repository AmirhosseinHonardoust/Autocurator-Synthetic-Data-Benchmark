import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def to_feature_space(df: pd.DataFrame, num_cols, cat_cols):
    scal = StandardScaler()
    # version-safe OneHotEncoder
    if hasattr(OneHotEncoder, "sparse_output"):
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    else:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Xn = scal.fit_transform(df[num_cols]) if num_cols else np.empty((len(df),0))
    Xc = enc.fit_transform(df[cat_cols]) if cat_cols else np.empty((len(df),0))
    X = np.concatenate([Xn, Xc], axis=1).astype(float)
    meta = {"scaler": scal, "encoder": enc, "num_cols": num_cols, "cat_cols": cat_cols}
    return X, meta
