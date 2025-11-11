import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score

def tstr_trts_scores(df_r: pd.DataFrame, df_s: pd.DataFrame, target: str, task: str = "classification"):
    if target not in df_r.columns or target not in df_s.columns:
        return None
    Xr, yr = df_r.drop(columns=[target]), df_r[target]
    Xs, ys = df_s.drop(columns=[target]), df_s[target]

    if task == "classification":
        model_r = LogisticRegression(max_iter=200)
        model_s = LogisticRegression(max_iter=200)
        # TSTR: train on synthetic, test on real
        model_s.fit(Xs, ys); ypr = model_s.predict_proba(Xr)[:,1]
        # TRTS: train on real, test on synthetic
        model_r.fit(Xr, yr); yps = model_r.predict_proba(Xs)[:,1]
        tstr_auc = roc_auc_score(yr, ypr)
        trts_auc = roc_auc_score(ys, yps)
        return {"TSTR_AUC": float(tstr_auc), "TRTS_AUC": float(trts_auc)}
    else:
        model_r = Ridge(); model_s = Ridge()
        model_s.fit(Xs, ys); ypr = model_s.predict(Xr)
        model_r.fit(Xr, yr); yps = model_r.predict(Xs)
        tstr_r2 = r2_score(yr, ypr); trts_r2 = r2_score(ys, yps)
        return {"TSTR_R2": float(tstr_r2), "TRTS_R2": float(trts_r2)}
