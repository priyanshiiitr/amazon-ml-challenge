"""Union the forward and reverse candidate sets, keeping both provenances.

Measured on 300,000 training entities against the full 10.3M right side:

    forward K=120                recall 0.9172   120.0 candidates/entity
    union forward K=60 + rev top-3  0.9120    65.9
    union forward K=40 + rev top-3  0.9044    46.6

So the union is markedly more *efficient* per candidate, which matters now that
a smaller candidate set is an explicit ranking criterion, even though reverse
retrieval alone recalls less than forward (0.856) and the union adds only 0.008
at equal size.

The retrieval path is kept as a feature, not discarded. `rev_rank = 0` means
this entity is that record's single best owner across all 2.2M Source 1
entities; given that exclusivity holds exactly in the ground truth (7,638,365
links over 7,638,365 distinct ids), that is strong evidence no pairwise feature
can express.

  python -m src.merge_cands --fwd work/cands_train.parquet \
      --rev work/rev_train_full.parquet --fwd-k 60 --rev-k 3 --out work/union.parquet
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd


def merge(fwd: pd.DataFrame, rev: pd.DataFrame, fwd_k: int, rev_k: int,
          verbose=True) -> pd.DataFrame:
    t0 = time.time()
    if fwd_k:
        rk = fwd.groupby("source1_entity_id")["sim_block"].rank(
            ascending=False, method="first")
        fwd = fwd[rk <= fwd_k]
    rev = rev[rev["rev_rank"] < rev_k]

    f = fwd[["source1_entity_id", "cand_id", "sim_block"]].copy()
    f["fwd_hit"] = np.int8(1)
    r = rev[["source1_entity_id", "cand_id", "rev_score", "rev_rank"]].copy()
    r["rev_hit"] = np.int8(1)

    out = f.merge(r, on=["source1_entity_id", "cand_id"], how="outer")
    out["fwd_hit"] = out["fwd_hit"].fillna(0).astype(np.int8)
    out["rev_hit"] = out["rev_hit"].fillna(0).astype(np.int8)
    # a pair found from both directions is a reciprocal hit -- the single
    # strongest provenance signal, and the reason to keep both columns
    out["both_hit"] = (out["fwd_hit"] & out["rev_hit"]).astype(np.int8)
    out["sim_block"] = out["sim_block"].fillna(0.0).astype(np.float32)
    out["rev_score"] = out["rev_score"].fillna(0.0).astype(np.float32)
    # rank 99 = "never retrieved in reverse", kept distinct from rank 0
    out["rev_rank"] = out["rev_rank"].fillna(99).astype(np.int8)

    if verbose:
        n_ent = out["source1_entity_id"].nunique()
        print(f"[merge] fwd_k={fwd_k} rev_k={rev_k} -> {len(out):,} pairs "
              f"({len(out) / max(n_ent, 1):.2f} per entity); "
              f"fwd-only {int((out.fwd_hit & ~out.rev_hit.astype(bool)).sum()):,}, "
              f"rev-only {int((out.rev_hit & ~out.fwd_hit.astype(bool)).sum()):,}, "
              f"both {int(out.both_hit.sum()):,} [{time.time() - t0:.0f}s]",
              flush=True)
    return out.reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwd", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--fwd-k", type=int, default=60)
    ap.add_argument("--rev-k", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fwd = pd.read_parquet(a.fwd)
    rev = pd.read_parquet(a.rev)
    # reverse runs over every Source 1 entity; keep only those the forward set
    # covers, so the two sides describe the same entity population
    ents = pd.Index(fwd["source1_entity_id"].unique())
    rev = rev[rev["source1_entity_id"].isin(ents)]
    merge(fwd, rev, a.fwd_k, a.rev_k).to_parquet(a.out, index=False)
    print(f"[merge] wrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
