import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def to_feature_space(df: pd.DataFrame, num_cols, cat_cols):
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Xn = scaler.fit_transform(df[num_cols]) if num_cols else np.empty((len(df), 0))
    Xc = encoder.fit_transform(df[cat_cols]) if cat_cols else np.empty((len(df), 0))
    X = np.concatenate([Xn, Xc], axis=1).astype(float)
    meta = {"scaler": scaler, "encoder": encoder, "num_cols": num_cols, "cat_cols": cat_cols}
    return X, meta
