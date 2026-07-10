import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score

from ..preprocess import to_feature_space


def _feature_matrices(df_r: pd.DataFrame, df_s: pd.DataFrame, target: str):
    """Encode real and synthetic features into an aligned numeric space.

    The encoder is fit on the union of both feature sets so that one-hot
    columns line up between real and synthetic (required for TSTR/TRTS, where a
    model trained on one is evaluated on the other).
    """
    feat_cols = [c for c in df_r.columns if c != target]
    num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df_r[c])]
    cat_cols = [c for c in feat_cols if c not in num_cols]

    combined = pd.concat([df_r[feat_cols], df_s[feat_cols]], axis=0, ignore_index=True)
    X_all, _ = to_feature_space(combined, num_cols, cat_cols)
    n_r = len(df_r)
    return X_all[:n_r], X_all[n_r:]


def _classification_auc(model, X_train, y_train, X_test, y_test) -> float:
    model.fit(X_train, y_train)
    classes = list(model.classes_)
    try:
        if len(classes) <= 2:
            proba = model.predict_proba(X_test)[:, 1]
            return float(roc_auc_score(y_test, proba))
        proba = model.predict_proba(X_test)
        return float(roc_auc_score(y_test, proba, multi_class="ovr", labels=classes))
    except ValueError:
        return float("nan")


def tstr_trts_scores(
    df_r: pd.DataFrame, df_s: pd.DataFrame, target: str, task: str = "classification"
) -> dict[str, float] | None:
    if target not in df_r.columns or target not in df_s.columns:
        return None

    Xr, Xs = _feature_matrices(df_r, df_s, target)
    yr, ys = df_r[target], df_s[target]

    if task == "classification":
        # TSTR: train on synthetic, test on real. TRTS: train on real, test on synthetic.
        tstr_auc = _classification_auc(LogisticRegression(max_iter=1000), Xs, ys, Xr, yr)
        trts_auc = _classification_auc(LogisticRegression(max_iter=1000), Xr, yr, Xs, ys)
        return {"TSTR_AUC": tstr_auc, "TRTS_AUC": trts_auc}

    model_s = Ridge().fit(Xs, ys)
    model_r = Ridge().fit(Xr, yr)
    tstr_r2 = r2_score(yr, model_s.predict(Xr))
    trts_r2 = r2_score(ys, model_r.predict(Xs))
    return {"TSTR_R2": float(tstr_r2), "TRTS_R2": float(trts_r2)}
