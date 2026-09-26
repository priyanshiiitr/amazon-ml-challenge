"""Gradient-boosting backend: LightGBM or XGBoost, selected by config.

Both are histogram-based GBDTs over the same features, so the pipeline treats
them interchangeably. XGBoost is worth having because its GPU path
(`device="cuda"`) is simple and fast, which matters on a GPU box where the
LightGBM CUDA build is awkward to get working.

Missing values matter here: the empty-address fix emits NaN rather than 0 so
the model can treat "no address information" differently from "the addresses
disagree". Both backends handle NaN natively -- LightGBM via a missing branch,
XGBoost via a learned default direction -- so the feature layer is unchanged.

Set the backend with AML_GBM=xgboost (or lightgbm), and AML_GPU=1 to use CUDA.
"""
from __future__ import annotations

import os

import numpy as np

BACKEND = os.environ.get("AML_GBM", "lightgbm").lower()
USE_GPU = os.environ.get("AML_GPU", "0") == "1"


def _xgb_params(lgb_params: dict) -> dict:
    """Translate the tuned LightGBM settings into XGBoost's names."""
    leaves = lgb_params.get("num_leaves", 255)
    p = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "max_leaves": leaves,
        "grow_policy": "lossguide",      # matches LightGBM's leaf-wise growth
        "max_depth": 0,                  # unlimited, as leaf-wise implies
        "eta": lgb_params.get("learning_rate", 0.05),
        "min_child_weight": lgb_params.get("min_data_in_leaf", 100) / 100.0,
        "colsample_bytree": lgb_params.get("feature_fraction", 0.85),
        "subsample": lgb_params.get("bagging_fraction", 0.85),
        "reg_lambda": lgb_params.get("lambda_l2", 1.0),
        "max_bin": lgb_params.get("max_bin", 255),
        "seed": lgb_params.get("seed", 42),
        "nthread": 0,
    }
    if USE_GPU:
        p["device"] = "cuda"
    return p


def train(params: dict, X, y, X_val=None, y_val=None, rounds=3000,
          early_stop=200, feature_names=None, log_every=200):
    """Train and return a backend-specific booster."""
    if BACKEND == "xgboost":
        import xgboost as xgb

        cols = feature_names or list(X.columns)
        dtr = xgb.DMatrix(X, label=y, feature_names=cols, nthread=-1)
        evals = [(dtr, "train")]
        dva = None
        if X_val is not None:
            dva = xgb.DMatrix(X_val, label=y_val, feature_names=cols, nthread=-1)
            evals = [(dva, "valid")]
        bst = xgb.train(_xgb_params(params), dtr, num_boost_round=rounds,
                        evals=evals,
                        early_stopping_rounds=early_stop if dva is not None else None,
                        verbose_eval=log_every)
        return bst

    import lightgbm as lgb

    cols = feature_names or list(X.columns)
    dtr = lgb.Dataset(X, label=y, feature_name=cols)
    cb = [lgb.log_evaluation(log_every)]
    valid = []
    if X_val is not None:
        valid = [lgb.Dataset(X_val, label=y_val, feature_name=cols,
                             reference=dtr)]
        cb.append(lgb.early_stopping(early_stop, verbose=False))
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=valid,
                     callbacks=cb)


def predict(booster, X, feature_names=None) -> np.ndarray:
    if BACKEND == "xgboost":
        import xgboost as xgb

        d = xgb.DMatrix(X, feature_names=feature_names or list(X.columns),
                        nthread=-1)
        it = getattr(booster, "best_iteration", None)
        rng = (0, it + 1) if it is not None else None
        out = (booster.predict(d, iteration_range=rng) if rng
               else booster.predict(d))
        return out.astype(np.float32)
    it = getattr(booster, "best_iteration", None)
    return booster.predict(X, num_iteration=it).astype(np.float32)


def best_iteration(booster) -> int:
    v = getattr(booster, "best_iteration", None)
    if v is not None:
        return int(v)
    return int(getattr(booster, "current_iteration", lambda: 0)())


def save(booster, path):
    booster.save_model(str(path))


def load(path):
    if BACKEND == "xgboost":
        import xgboost as xgb

        b = xgb.Booster()
        b.load_model(str(path))
        return b
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))


def importance(booster, cols, top=15):
    """-> [(feature, gain)] sorted descending, backend-independent."""
    if BACKEND == "xgboost":
        sc = booster.get_score(importance_type="gain")
        pairs = [(c, float(sc.get(c, 0.0))) for c in cols]
    else:
        pairs = list(zip(cols, booster.feature_importance("gain")))
    return sorted(pairs, key=lambda x: -x[1])[:top]


def model_suffix() -> str:
    """Keep the two backends' model files apart."""
    return "json" if BACKEND == "xgboost" else "txt"
