"""Two-stage stacking: cross-fitted stage-1 probabilities feed collective features.

The cluster-support features in `pairfeat` anchor on `sim_block`, the blocking
cosine, because that is the only per-candidate score available when features are
built. That is a weak anchor. What we actually want is "how much do this
entity's *probable* matches agree with this candidate" -- which needs model
probabilities at feature time, and using in-sample probabilities would leak.

Cross-fitting solves it: split entities into folds, train stage 1 on the other
folds, predict the held-out one, rotate. Every training pair then carries an
out-of-fold probability, and sibling support can be weighted by it.

This matters because an entity's true matches are corrupted copies of one
business and resemble each other far more than anything else -- max(skeleton,
address) similarity averages 97.2 over co-matched pairs versus 46.7 over random
right-right pairs. Applying that as a hard rule cost precision (0.978 -> 0.927)
and lost; as probability-weighted features the model can use it selectively.

  python -m src.stack --cands work/union_train_comp.parquet --folds 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process

from . import config, gbm
from .pairfeat import build_features
from .run_model import ATTR_COLS, enc_ids, load_left, load_right, read_ground_truth

TOPM = 12
EDGE = 85.0


def fold_of(entity_ids, n_folds: int) -> np.ndarray:
    """Deterministic fold per entity, from the numeric part of its id."""
    return np.fromiter((int(e[3:]) % n_folds for e in entity_ids),
                       np.int8, len(entity_ids))


def sibling_features(pairs: pd.DataFrame, attrs: pd.DataFrame,
                     prob: np.ndarray, verbose=True) -> pd.DataFrame:
    """Probability-weighted agreement between a candidate and its siblings."""
    t0 = time.time()
    n = len(pairs)
    ent = pairs["source1_entity_id"].to_numpy()
    cand = pairs["cand_id"].to_numpy()
    codes, _ = pd.factorize(ent)

    R = attrs.reindex(cand)
    rs = R["s_name"].fillna("").to_numpy()
    ra = R["n_addr"].fillna("").to_numpy()
    del R

    # rank within entity by stage-1 probability, keep the top M
    order = np.lexsort((-prob, codes))
    cs = codes[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    gs = np.repeat(starts, np.diff(np.r_[starts, len(cs)]))
    rank = np.empty(n, np.int64)
    rank[order] = np.arange(n) - gs
    keep = np.flatnonzero(rank < TOPM)
    keep = keep[np.argsort(codes[keep], kind="stable")]
    ck = codes[keep]
    gstart = np.flatnonzero(np.r_[True, ck[1:] != ck[:-1]])
    gend = np.r_[gstart[1:], len(ck)]

    ai_l, bi_l = [], []
    for s_, e_ in zip(gstart.tolist(), gend.tolist()):
        m = e_ - s_
        if m < 2:
            continue
        u, v = np.triu_indices(m, 1)
        ai_l.append(keep[s_ + u])
        bi_l.append(keep[s_ + v])

    sib_max = np.zeros(n, np.float32)
    sib_sum = np.zeros(n, np.float32)
    sib_cnt = np.zeros(n, np.float32)
    sib_sim = np.zeros(n, np.float32)
    if ai_l:
        ai = np.concatenate(ai_l)
        bi = np.concatenate(bi_l)
        s_s = rf_process.cpdist(rs[ai], rs[bi], scorer=fuzz.token_set_ratio,
                                workers=-1, dtype=np.float32)
        s_a = rf_process.cpdist(ra[ai], ra[bi], scorer=fuzz.token_set_ratio,
                                workers=-1, dtype=np.float32)
        mx = np.maximum(s_s, s_a)
        del s_s, s_a
        w_ab = (mx / 100.0) * prob[bi]     # support b gives a
        w_ba = (mx / 100.0) * prob[ai]
        np.maximum.at(sib_max, ai, w_ab)
        np.maximum.at(sib_max, bi, w_ba)
        np.add.at(sib_sum, ai, w_ab)
        np.add.at(sib_sum, bi, w_ba)
        hit = (mx >= EDGE) & (prob[bi] >= 0.5)
        np.add.at(sib_cnt, ai[hit], 1.0)
        hit2 = (mx >= EDGE) & (prob[ai] >= 0.5)
        np.add.at(sib_cnt, bi[hit2], 1.0)
        np.maximum.at(sib_sim, ai, mx)
        np.maximum.at(sib_sim, bi, mx)
        del ai, bi, mx, w_ab, w_ba

    out = pd.DataFrame(index=pairs.index)
    out["p1"] = prob.astype(np.float32)
    out["sib_max"] = sib_max
    out["sib_sum"] = sib_sum
    out["sib_cnt"] = sib_cnt
    out["sib_sim"] = sib_sim
    # how this candidate's own probability compares with its entity's best
    best = np.zeros(codes.max() + 1, np.float32)
    np.maximum.at(best, codes, prob.astype(np.float32))
    out["p1_rel"] = prob / np.maximum(best[codes], 1e-6)
    out["p1_gap"] = best[codes] - prob
    tot = np.zeros(codes.max() + 1, np.float32)
    np.add.at(tot, codes, prob.astype(np.float32))
    out["p1_mass"] = tot[codes]
    if verbose:
        print(f"    sibling features [{time.time() - t0:.0f}s]", flush=True)
    return out.astype(np.float32)


SIB_COLS = ["p1", "sib_max", "sib_sum", "sib_cnt", "sib_sim",
            "p1_rel", "p1_gap", "p1_mass"]


def main(argv=None):
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--split", default="train")
    a = ap.parse_args(argv)
    t0 = time.time()

    pairs = pd.read_parquet(a.cands)
    ents = set(pairs["source1_entity_id"].unique())
    truth = read_ground_truth(ents)
    left = load_left(a.split)
    right = load_right(a.split, keep=pairs["cand_id"].unique())
    feats = build_features(pairs, left, right, split=a.split)
    del left
    print(f"[stack] base features {feats.shape} [{time.time() - t0:.0f}s]",
          flush=True)

    truth_enc = {sid: (set(enc_ids(sorted(ms)).tolist()) if ms else set())
                 for sid, ms in truth.items()}
    sids = pairs["source1_entity_id"].to_numpy()
    cids = pairs["cand_id"].to_numpy()
    y = np.fromiter((c in truth_enc.get(s, ()) for s, c in zip(sids, cids)),
                    np.int8, len(pairs))
    print(f"[stack] positive rate {y.mean():.4f}", flush=True)

    folds = fold_of(sids, a.folds)
    cols = list(feats.columns)
    oof = np.zeros(len(pairs), np.float32)
    print(f"[stack] backend={gbm.BACKEND} gpu={gbm.USE_GPU}", flush=True)
    for k in range(a.folds):
        tr = folds != k
        b = gbm.train(config.LGB_PARAMS, feats[tr], y[tr], rounds=900,
                      feature_names=cols, log_every=300)
        oof[~tr] = gbm.predict(b, feats[~tr], cols)
        print(f"[stack] fold {k + 1}/{a.folds} done [{time.time() - t0:.0f}s]",
              flush=True)
        del b

    # full stage-1 model, for use at test time
    full = gbm.train(config.LGB_PARAMS, feats, y, rounds=900,
                     feature_names=cols, log_every=300)
    gbm.save(full, config.MODEL_DIR / f"stage1.{gbm.model_suffix()}")
    json.dump(cols, open(config.MODEL_DIR / "features_stage1.json", "w"))
    print(f"[stack] stage-1 full model saved [{time.time() - t0:.0f}s]",
          flush=True)

    sib = sibling_features(pairs, right, oof)
    del right
    feats2 = pd.concat([feats, sib], axis=1)
    del feats, sib
    print(f"[stack] stage-2 features {feats2.shape} [{time.time() - t0:.0f}s]",
          flush=True)

    uniq = np.array(sorted(ents))
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(uniq)
    val_ents = set(uniq[:int(len(uniq) * config.VALID_FRACTION)].tolist())
    is_val = np.fromiter((s in val_ents for s in sids), bool, len(sids))

    cols2 = list(feats2.columns)
    booster = gbm.train(config.LGB_PARAMS, feats2[~is_val], y[~is_val],
                        feats2[is_val], y[is_val], rounds=config.LGB_ROUNDS,
                        early_stop=config.LGB_EARLY_STOP, feature_names=cols2)
    gbm.save(booster, config.MODEL_DIR / f"model.{gbm.model_suffix()}")
    json.dump(cols2, open(config.MODEL_DIR / "features.json", "w"))
    print(f"[stack] stage-2 best_iter={gbm.best_iteration(booster)}", flush=True)
    imp = gbm.importance(booster, cols2, top=12)
    for nm, g in imp:
        print(f"    {nm:22s} {g:12.0f}")

    val = pd.DataFrame({
        "source1_entity_id": sids[is_val], "cand_entity_id": cids[is_val],
        "prob": gbm.predict(booster, feats2[is_val], cols2)})
    val.to_parquet(config.WORK_DIR / "val_probs.parquet", index=False)
    import pickle
    tm = {s: truth_enc[s] for s in val_ents if s in truth_enc}
    pickle.dump(tm, open(config.WORK_DIR / "val_truth.pkl", "wb"))
    print(f"[stack] wrote val probs; total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
