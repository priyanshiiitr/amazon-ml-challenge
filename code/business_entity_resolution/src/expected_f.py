"""Choose each entity's prediction set by maximising expected F_0.5.

The tuned thresholds (tau_empty, tau_abs, alpha) are three global numbers. They
cannot express "this entity looks like it has four matches and that one looks
like a singleton", yet that is exactly what a macro-averaged, per-entity metric
rewards. Since the model emits calibrated probabilities, the set size can be
chosen analytically instead.

For an entity whose candidates have probabilities p_1 >= p_2 >= ... >= p_n:

    predicting the top k   ->  E[TP] = sum_{i<=k} p_i
                               E[T]  = gamma * sum_i p_i
                               P = E[TP]/k,  R = E[TP]/E[T]
                               F_k = 1.25 P R / (0.25 P + R)

    predicting nothing     ->  F_0 = prod_i (1 - p_i)
                               (the probability the entity really is a singleton,
                                which is exactly what an empty prediction scores)

and we take the k with the highest F_k. `gamma` (>= 1) inflates the expected
true-match count to account for matches blocking never retrieved -- without it
recall is measured against only what we found, which biases the rule toward
small sets. It is the single knob tuned on held-out data.

Everything is vectorised over the flat candidate table; a Python loop over
millions of entities would dominate the run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import macro_f_beta


def decide(df: pd.DataFrame, gamma: float = 1.15, min_p: float = 0.02,
           prob_col: str = "prob", tau_empty: float | None = None) -> dict:
    """df: [source1_entity_id, cand_entity_id, prob] -> {sid: set(ids)}.

    `tau_empty` replaces the analytic abstention test with the tuned gate. The
    two rules each win one half of the metric: measured on 60,000 held-out
    entities, the analytic rule scores 0.90911 on non-singletons against the
    threshold rule's 0.90213, but only 0.74315 on singletons against 0.86841
    (853 false merges vs 437). Deciding *whether* to predict by the gate and
    *how many* to predict analytically takes the better half of each.
    """
    if df.empty:
        return {}
    ent = df["source1_entity_id"].to_numpy()
    cand = df["cand_entity_id"].to_numpy()
    p = df[prob_col].to_numpy(np.float64)

    keep = p >= min_p
    ent, cand, p = ent[keep], cand[keep], p[keep]
    if len(ent) == 0:
        return {}

    codes, uniq = pd.factorize(ent)
    order = np.lexsort((-p, codes))
    codes, cand, p = codes[order], cand[order], p[order]

    n_ent = len(uniq)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)]
    gstart = np.repeat(starts, ends - starts)
    rank = np.arange(len(codes)) - gstart          # 0-based rank within entity

    # cumulative expected true positives within each entity
    csum = np.cumsum(p)
    base = np.r_[0.0, csum[starts[1:] - 1]] if n_ent > 1 else np.array([0.0])
    tp = csum - np.repeat(base, ends - starts)

    total = np.zeros(n_ent)
    np.add.at(total, codes, p)
    t_hat = np.maximum(gamma * total[codes], 1e-9)

    k = rank + 1.0
    prec = tp / k
    rec = tp / t_hat
    with np.errstate(divide="ignore", invalid="ignore"):
        fk = 1.25 * prec * rec / (0.25 * prec + rec)
    fk = np.nan_to_num(fk)

    # best k per entity
    best_f = np.zeros(n_ent)
    np.maximum.at(best_f, codes, fk)

    # F for predicting nothing = P(entity is truly a singleton)
    log1m = np.log(np.maximum(1.0 - p, 1e-12))
    s = np.zeros(n_ent)
    np.add.at(s, codes, log1m)
    f_empty = np.exp(s)

    if tau_empty is None:
        take_some = best_f > f_empty
    else:
        top_p = np.zeros(n_ent)
        np.maximum.at(top_p, codes, p)
        take_some = top_p >= tau_empty
    # keep every candidate ranked at or above the argmax position
    is_best = fk >= best_f[codes] - 1e-12
    best_rank = np.full(n_ent, -1.0)
    np.maximum.at(best_rank, codes[is_best], rank[is_best])
    sel = (rank <= best_rank[codes]) & take_some[codes]

    out: dict = {}
    for c, r in zip(codes[sel].tolist(), cand[sel].tolist()):
        out.setdefault(uniq[c], set()).add(r)
    return out


def tune_gamma(val: pd.DataFrame, true_map: dict, grid=None, verbose=True):
    """Pick gamma on held-out entities against the real metric."""
    grid = grid or [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0]
    best = (-1.0, None)
    for g in grid:
        pred = decide(val, gamma=g)
        s = macro_f_beta(pred, true_map)
        if verbose:
            print(f"    gamma={g:<5} macro_F0.5={s:.5f}", flush=True)
        if s > best[0]:
            best = (s, g)
    if verbose:
        print(f"  best gamma={best[1]} -> {best[0]:.5f}", flush=True)
    return {"score": best[0], "gamma": best[1]}
