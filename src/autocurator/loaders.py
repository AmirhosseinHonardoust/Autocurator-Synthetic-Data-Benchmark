import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    return df

def split_schema(df: pd.DataFrame, target: str | None = None):
    feat_cols = [c for c in df.columns if c != target] if target else df.columns.tolist()
    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feat_cols if c not in num_cols]
    return num_cols, cat_cols, feat_cols
