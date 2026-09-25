"""Pairwise matcher: LightGBM binary classifier over candidate pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .features import NUMERIC_FEATURES


def label_candidates(cands: pd.DataFrame, gt: pd.DataFrame) -> np.ndarray:
    """1 if the candidate pair is a true match in the ground truth."""
    truth = set()
    for sid, ms in zip(gt["source1_entity_id"], gt["matches"]):
        for m in ms:
            truth.add((sid, m))
    return np.fromiter(
        ((1 if (a, b) in truth else 0)
         for a, b in zip(cands["source1_entity_id"], cands["cand_entity_id"])),
        dtype=np.int8, count=len(cands),
    )


def split_entities(s1: pd.DataFrame, frac: float = None, seed: int = None):
    """Split by *entity*, never by pair — otherwise the same entity's candidates
    straddle the split and the validation score is optimistic."""
    frac = config.VALID_FRACTION if frac is None else frac
    seed = config.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    ids = s1["entity_id"].to_numpy()
    mask = rng.random(len(ids)) < frac
    return set(ids[~mask]), set(ids[mask])


def train(train_df: pd.DataFrame, valid_df: pd.DataFrame,
          features: list[str] = None, verbose: bool = True):
    import lightgbm as lgb

    features = features or NUMERIC_FEATURES
    dtrain = lgb.Dataset(train_df[features], label=train_df["y"], free_raw_data=True)
    dvalid = lgb.Dataset(valid_df[features], label=valid_df["y"], reference=dtrain)

    booster = lgb.train(
        config.LGB_PARAMS,
        dtrain,
        num_boost_round=config.LGB_ROUNDS,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(config.LGB_EARLY_STOP, verbose=verbose),
            lgb.log_evaluation(100 if verbose else 0),
        ],
    )
    if verbose:
        imp = pd.Series(booster.feature_importance("gain"), index=features)
        print("\n[model] top features by gain:")
        print(imp.sort_values(ascending=False).head(25).to_string())
    return booster


def predict(booster, df: pd.DataFrame, features: list[str] = None,
            chunk: int = 2_000_000) -> np.ndarray:
    features = features or NUMERIC_FEATURES
    out = np.empty(len(df), np.float32)
    for s in range(0, len(df), chunk):
        e = min(s + chunk, len(df))
        out[s:e] = booster.predict(
            df[features].iloc[s:e], num_iteration=booster.best_iteration
        ).astype(np.float32)
    return out


def save(booster, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    return path


def load(path: Path):
    import lightgbm as lgb
    return lgb.Booster(model_file=str(path))
