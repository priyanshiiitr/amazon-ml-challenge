"""The competition metric, implemented exactly as specified.

F_0.5 = (1.25 * P * R) / (0.25 * P + R), computed per source1 entity and then
macro-averaged over *all* source1 entities in the evaluation set.

Singleton convention (from the problem statement):
  true empty & predicted empty -> 1.0
  true empty & predicted non-empty -> 0.0
  true non-empty & predicted empty -> 0.0
"""
from __future__ import annotations

import numpy as np


def f_beta_pair(pred: set, true: set, beta: float = 0.5) -> float:
    if not true and not pred:
        return 1.0
    if not true or not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f_beta(pred_map: dict, true_map: dict, beta: float = 0.5) -> float:
    """pred_map/true_map: {source1_entity_id -> set(ids)}. Averaged over true_map keys."""
    if not true_map:
        return 0.0
    total = 0.0
    for sid, true in true_map.items():
        total += f_beta_pair(pred_map.get(sid, set()), true, beta)
    return total / len(true_map)


def detailed_report(pred_map: dict, true_map: dict) -> dict:
    """Breakdown that tells you *where* score is being lost."""
    n = len(true_map)
    scores = np.empty(n, np.float32)
    sing_idx, nonsing_idx = [], []
    fp_on_singletons = 0
    missed_all = 0
    tp = fp = fn = 0

    for i, (sid, true) in enumerate(true_map.items()):
        pred = pred_map.get(sid, set())
        scores[i] = f_beta_pair(pred, true)
        if not true:
            sing_idx.append(i)
            if pred:
                fp_on_singletons += 1
        else:
            nonsing_idx.append(i)
            if not pred:
                missed_all += 1
        tp += len(pred & true)
        fp += len(pred - true)
        fn += len(true - pred)

    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    micro_f = (1.25 * micro_p * micro_r / max(0.25 * micro_p + micro_r, 1e-9))

    return {
        "macro_f0.5": float(scores.mean()),
        "n_entities": n,
        "n_singletons": len(sing_idx),
        "singleton_score": float(scores[sing_idx].mean()) if sing_idx else float("nan"),
        "nonsingleton_score": float(scores[nonsing_idx].mean()) if nonsing_idx else float("nan"),
        "false_merges_on_singletons": fp_on_singletons,
        "nonsingletons_predicted_empty": missed_all,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f0.5": micro_f,
        "link_tp": tp, "link_fp": fp, "link_fn": fn,
    }


def print_report(rep: dict, title: str = "validation"):
    print(f"\n--- {title} ---")
    print(f"  MACRO F0.5            : {rep['macro_f0.5']:.5f}   <- leaderboard metric")
    print(f"  entities              : {rep['n_entities']:,} "
          f"({rep['n_singletons']:,} singletons = "
          f"{rep['n_singletons'] / max(rep['n_entities'], 1):.1%})")
    print(f"  score on singletons   : {rep['singleton_score']:.5f}")
    print(f"  score on non-single   : {rep['nonsingleton_score']:.5f}")
    print(f"  false merges on singl : {rep['false_merges_on_singletons']:,} "
          f"(each costs a full 1.0)")
    print(f"  non-single left empty : {rep['nonsingletons_predicted_empty']:,}")
    print(f"  link P / R            : {rep['micro_precision']:.4f} / {rep['micro_recall']:.4f}")
