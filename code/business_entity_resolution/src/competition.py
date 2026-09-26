"""Competition features: what else is claiming this candidate?

Every Source 2/3 record belongs to at most one Source 1 entity -- verified on
the training ground truth, where 7,638,365 links used 7,638,365 *distinct* ids
with zero reuse. The pair model currently scores each (entity, candidate) pair
in isolation and so cannot see that a candidate it likes is also the runaway
favourite of a different entity, which is strong evidence against this claim.

These features restore that view. They are computed once over the whole
candidate table and written back into it, because the prediction pass streams
the table in chunks and a per-chunk groupby would see only part of each
candidate's claimants.

Columns added:
  cand_n_claims    how many Source 1 entities list this candidate at all
  cand_max_sim     the best blocking score any entity achieves on it
  sim_rel_cand     this pair's score as a fraction of that best
  cand_is_best     1 when this entity is the candidate's strongest claimant
  cand_rank        this pair's rank among the candidate's claimants (0 = best)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def add_competition_features(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """Augment a candidate table in place-ish. Requires cand_id and sim_block."""
    t0 = time.time()
    cand = df["cand_id"].to_numpy()
    sim = df["sim_block"].to_numpy(np.float32)

    # dense code per candidate id, via sort rather than pandas factorize:
    # at test scale this runs over hundreds of millions of rows
    order = np.argsort(cand, kind="stable")
    cs = cand[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    codes_sorted = np.zeros(len(cs), np.int64)
    codes_sorted[starts[1:]] = 1
    np.cumsum(codes_sorted, out=codes_sorted)
    codes = np.empty(len(cs), np.int64)
    codes[order] = codes_sorted
    n_cand = int(codes_sorted[-1]) + 1 if len(cs) else 0
    del cs, codes_sorted, starts

    n_claims = np.bincount(codes, minlength=n_cand).astype(np.float32)
    cand_max = np.zeros(n_cand, np.float32)
    np.maximum.at(cand_max, codes, sim)

    df["cand_n_claims"] = n_claims[codes]
    cmax = cand_max[codes]
    df["cand_max_sim"] = cmax
    df["sim_rel_cand"] = sim / np.maximum(cmax, 1e-6)
    df["cand_is_best"] = (sim >= cmax - 1e-6).astype(np.int8)

    # rank within each candidate's claimant list, best first
    o2 = np.lexsort((-sim, codes))
    cc = codes[o2]
    st = np.flatnonzero(np.r_[True, cc[1:] != cc[:-1]])
    gs = np.repeat(st, np.diff(np.r_[st, len(cc)]))
    rank = np.empty(len(cc), np.float32)
    rank[o2] = np.arange(len(cc)) - gs
    df["cand_rank"] = rank

    if verbose:
        print(f"[competition] {len(df):,} pairs over {n_cand:,} distinct "
              f"candidates; mean claims {n_claims.mean():.2f}, "
              f"max {int(n_claims.max())} [{time.time() - t0:.0f}s]", flush=True)
    return df


COMP_COLS = ["cand_n_claims", "cand_max_sim", "sim_rel_cand", "cand_is_best",
             "cand_rank"]


def add_name_competition(df: pd.DataFrame, split: str, verbose=True):
    """Rank each candidate's claimants by NAME similarity, not blocking score.

    When a Source 2/3 record has no address -- about 3.3% of them -- the only
    evidence available is the name, and the blocking score for such a record is
    built from name keys alone, so it ties easily. Error analysis found true
    pairs with an exactly matching name and a blank address still scoring ~0.1.

    Exclusivity makes the right question "is this the entity whose name matches
    this candidate best?", which no per-pair feature can answer. This computes
    it once over the whole table, like the other competition features.
    """
    import time

    from rapidfuzz import fuzz
    from rapidfuzz import process as rf_process

    from .run_model import load_left, load_right

    t0 = time.time()
    left = load_left(split)
    right = load_right(split, keep=df["cand_id"].unique())
    ls = left.reindex(df["source1_entity_id"].to_numpy())["s_name"].fillna("").to_numpy()
    rsn = right.reindex(df["cand_id"].to_numpy())["s_name"].fillna("").to_numpy()
    del left, right
    sim = rf_process.cpdist(ls, rsn, scorer=fuzz.token_set_ratio, workers=-1,
                            dtype=np.float32)
    del ls, rsn
    if verbose:
        print(f"[competition] name sims computed [{time.time() - t0:.0f}s]",
              flush=True)

    cand = df["cand_id"].to_numpy()
    order = np.argsort(cand, kind="stable")
    cs = cand[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    codes_sorted = np.zeros(len(cs), np.int64)
    codes_sorted[starts[1:]] = 1
    np.cumsum(codes_sorted, out=codes_sorted)
    codes = np.empty(len(cs), np.int64)
    codes[order] = codes_sorted
    n_cand = int(codes_sorted[-1]) + 1 if len(cs) else 0
    del cs, codes_sorted, starts

    cmax = np.zeros(n_cand, np.float32)
    np.maximum.at(cmax, codes, sim)
    df["cand_name_sim"] = sim
    df["cand_name_max"] = cmax[codes]
    df["cand_name_rel"] = sim / np.maximum(cmax[codes], 1e-6)
    df["cand_name_is_best"] = (sim >= cmax[codes] - 1e-6).astype(np.int8)

    o2 = np.lexsort((-sim, codes))
    cc = codes[o2]
    st = np.flatnonzero(np.r_[True, cc[1:] != cc[:-1]])
    gs = np.repeat(st, np.diff(np.r_[st, len(cc)]))
    rank = np.empty(len(cc), np.float32)
    rank[o2] = np.arange(len(cc)) - gs
    df["cand_name_rank"] = rank
    if verbose:
        print(f"[competition] name competition done [{time.time() - t0:.0f}s]",
              flush=True)
    return df


NAME_COLS = ["cand_name_sim", "cand_name_max", "cand_name_rel",
             "cand_name_is_best", "cand_name_rank"]


def main(path_in, path_out=None, split="train"):
    df = pd.read_parquet(path_in)
    add_competition_features(df)
    add_name_competition(df, split)
    df.to_parquet(path_out or path_in, index=False)
    print(f"[competition] wrote {path_out or path_in}", flush=True)


if __name__ == "__main__":
    import sys
    main(sys.argv[1],
         sys.argv[2] if len(sys.argv) > 2 else None,
         sys.argv[3] if len(sys.argv) > 3 else "train")
