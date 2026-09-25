"""Turning per-pair probabilities into per-entity ID lists.

This module is where a surprising share of the score lives. A globally optimal
probability threshold is *not* optimal for a macro-averaged, precision-heavy,
singleton-aware metric:

* Each singleton is worth a full 1.0 for predicting nothing, and a flat 0.0 for
  predicting anything. With a high singleton rate, the "should this entity have
  any matches at all" gate matters more than ranking within an entity.
* Because the average is per entity, an entity with one candidate at p=0.55 and
  an entity with eight candidates at p=0.55 want different treatment. The
  relative threshold (alpha * best) handles that; a global cut cannot.

So we tune three knobs jointly:
    tau_empty : if the best candidate for an entity scores below this, predict []
    tau_abs   : absolute floor a candidate must clear to be kept
    alpha     : keep candidates scoring >= alpha * best_for_this_entity
and optionally a one-to-one assignment pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .evaluate import macro_f_beta


def apply_decision(df: pd.DataFrame, tau_empty: float, tau_abs: float,
                   alpha: float, one_to_one: bool = False,
                   prob_col: str = "prob") -> dict:
    """df: [source1_entity_id, cand_entity_id, prob]. Returns {sid -> set(ids)}."""
    if df.empty:
        return {}
    d = df[[  "source1_entity_id", "cand_entity_id", prob_col]].copy()
    best = d.groupby("source1_entity_id")[prob_col].transform("max")
    thr = np.maximum(tau_abs, alpha * best.to_numpy())
    keep = (d[prob_col].to_numpy() >= thr) & (best.to_numpy() >= tau_empty)
    d = d[keep]
    if d.empty:
        return {}

    if one_to_one:
        # Each source2/3 record describes one real business, so it should belong
        # to at most one source1 entity. Keep only its highest-scoring claim.
        d = d.sort_values(prob_col, ascending=False)
        d = d.drop_duplicates(subset="cand_entity_id", keep="first")

    return {k: set(v) for k, v in
            d.groupby("source1_entity_id")["cand_entity_id"].apply(list).items()}


def _score_mask(mask, ent_code, y, n_true, n_entities) -> float:
    """Macro F0.5 for one candidate-keep mask, as pure bincount arithmetic.

    Rebuilding python dicts per configuration costs seconds per config, which is
    hours over a full grid at competition scale. Everything here is O(n_pairs)
    numpy, so a grid of thousands of configs finishes in minutes.
    """
    kept = ent_code[mask]
    n_pred = np.bincount(kept, minlength=n_entities).astype(np.float64)
    tp = np.bincount(kept, weights=y[mask], minlength=n_entities)

    f = np.zeros(n_entities, np.float64)
    both_empty = (n_pred == 0) & (n_true == 0)
    f[both_empty] = 1.0
    live = (n_pred > 0) & (n_true > 0) & (tp > 0)
    p = tp[live] / n_pred[live]
    r = tp[live] / n_true[live]
    f[live] = 1.25 * p * r / (0.25 * p + r)
    return float(f.mean())


def tune_decision(val_df: pd.DataFrame, true_map: dict, verbose: bool = True,
                  prob_col: str = "prob") -> dict:
    """Grid-search the decision knobs against the real metric on a held-out split.

    `true_map` must cover every held-out source1 entity, including ones that
    produced no candidates at all — those are scored too, and each of them is a
    guaranteed 1.0 (correct empty prediction) that the average must include.
    """
    entities = list(true_map.keys())
    ent_index = {e: i for i, e in enumerate(entities)}
    n_entities = len(entities)
    n_true = np.array([len(true_map[e]) for e in entities], np.float64)

    d = val_df[val_df["source1_entity_id"].isin(ent_index)]
    if d.empty:
        return {"score": float((n_true == 0).mean()), "tau_empty": 1.0,
                "tau_abs": 1.0, "alpha": 1.0, "one_to_one": False}

    ent_code = d["source1_entity_id"].map(ent_index).to_numpy(np.int64)
    cid = d["cand_entity_id"].to_numpy()
    prob = d[prob_col].to_numpy(np.float32)
    y = np.array([1.0 if c in true_map[entities[e]] else 0.0
                  for e, c in zip(ent_code, cid)], np.float64)
    bst = pd.Series(prob).groupby(ent_code).transform("max").to_numpy(np.float32)

    grid = [(te, ta, al)
            for te in config.TAU_EMPTY_GRID
            for ta in config.TAU_ABS_GRID
            for al in config.ALPHA_GRID
            if ta <= te]
    if verbose:
        print(f"[decision] searching {len(grid):,} threshold configurations "
              f"over {len(d):,} pairs / {n_entities:,} entities")

    results = []
    for te, ta, al in grid:
        mask = (prob >= np.maximum(ta, al * bst)) & (bst >= te)
        results.append((_score_mask(mask, ent_code, y, n_true, n_entities),
                        te, ta, al))
    results.sort(reverse=True)
    best_score, te, ta, al = results[0]
    best = {"score": best_score, "tau_empty": te, "tau_abs": ta,
            "alpha": al, "one_to_one": False}
    if verbose:
        print(f"  best without one-to-one: {best_score:.5f} "
              f"tau_empty={te} tau_abs={ta} alpha={al}")

    # One-to-one is exact but needs a per-config dedup, so only the strongest
    # threshold settings are re-tested with it.
    if True in config.ONE_TO_ONE_OPTIONS:
        order = np.argsort(-prob, kind="stable")
        for score, te, ta, al in results[:10]:
            mask = (prob >= np.maximum(ta, al * bst)) & (bst >= te)
            idx = order[mask[order]]
            _, first = np.unique(cid[idx], return_index=True)
            keep = np.zeros(len(prob), bool)
            keep[idx[np.sort(first)]] = True
            s = _score_mask(keep, ent_code, y, n_true, n_entities)
            if s > best["score"]:
                best = {"score": s, "tau_empty": te, "tau_abs": ta,
                        "alpha": al, "one_to_one": True}
        if verbose and best["one_to_one"]:
            print(f"  one-to-one assignment improves it to {best['score']:.5f}")

    if verbose:
        on_edge = []
        if best["tau_empty"] in (config.TAU_EMPTY_GRID[0], config.TAU_EMPTY_GRID[-1]):
            on_edge.append("tau_empty")
        if best["alpha"] in (config.ALPHA_GRID[0], config.ALPHA_GRID[-1]):
            on_edge.append("alpha")
        if on_edge:
            print(f"  [warn] optimum sits on the grid boundary for {on_edge} — "
                  f"widen the grid in config.py")
    return best


def tune_per_country(val_df: pd.DataFrame, true_map: dict, country_of: dict,
                     verbose: bool = True) -> dict:
    """Separate knobs per country.

    Worth trying because US and India have very different address conventions,
    but be careful: France is absent from training, so it has no tuned knobs of
    its own and must fall back to the global setting. `apply_per_country` does
    that automatically.
    """
    out = {}
    countries = sorted({country_of.get(s, "") for s in true_map})
    for c in countries:
        ids = {s for s in true_map if country_of.get(s, "") == c}
        sub = val_df[val_df["source1_entity_id"].isin(ids)]
        tm = {s: true_map[s] for s in ids}
        if len(tm) < 200:
            continue
        out[c] = tune_decision(sub, tm, verbose=False)
        if verbose:
            print(f"  [{c}] F0.5={out[c]['score']:.5f} "
                  f"tau_empty={out[c]['tau_empty']} tau_abs={out[c]['tau_abs']} "
                  f"alpha={out[c]['alpha']} one_to_one={out[c]['one_to_one']}")
    return out


def apply_per_country(df: pd.DataFrame, per_country: dict, fallback: dict,
                      country_of: dict, prob_col: str = "prob") -> dict:
    """Unseen countries (France) transparently use `fallback`."""
    if df.empty:
        return {}
    d = df.copy()
    d["_country"] = d["source1_entity_id"].map(country_of).fillna("")
    pred = {}
    for c, sub in d.groupby("_country"):
        knobs = per_country.get(c, fallback)
        pred.update(apply_decision(
            sub, knobs["tau_empty"], knobs["tau_abs"], knobs["alpha"],
            knobs["one_to_one"], prob_col=prob_col))
    return pred
