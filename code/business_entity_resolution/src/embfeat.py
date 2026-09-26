"""Attach embedding cosine to a candidate table.

Computed once per candidate file and stored as a column, rather than inside
`build_features`, because prediction streams the table twice (stage 1 then
stage 2) and the gather is the expensive part — 100M+ random reads into a
10M x 384 matrix.

Lookup is positional: `embed.py` writes vectors in parquet row order, so row i
of the .npy is row i of the parquet. Left ids map through a dict; right ids are
int64-encoded and map through a sorted array plus searchsorted.

  python -m src.embfeat --cands work/union_test_comp.parquet --split test
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .run_model import enc_ids

CHUNK = 4_000_000


def _left_index(split):
    ids = pq.read_table(config.WORK_DIR / f"{split}_source1.parquet",
                        columns=["entity_id"]).to_pandas()["entity_id"]
    return {e: i for i, e in enumerate(ids)}


def _right_index(split):
    """-> (sorted encoded ids, row offset into the concatenated matrix)."""
    keys, offs, base = [], [], 0
    for nm in ("source2", "source3"):
        ids = pq.read_table(config.WORK_DIR / f"{split}_{nm}.parquet",
                            columns=["entity_id"]).to_pandas()["entity_id"]
        keys.append(enc_ids(ids.to_numpy()))
        offs.append(np.arange(len(ids), dtype=np.int64) + base)
        base += len(ids)
    k = np.concatenate(keys)
    o = np.concatenate(offs)
    order = np.argsort(k, kind="stable")
    return k[order], o[order], base


def add(cands_path, split, out_path=None, verbose=True):
    t0 = time.time()
    lmap = _left_index(split)
    rkeys, rrows, n_right = _right_index(split)

    L = np.load(config.WORK_DIR / f"emb_{split}_source1.npy", mmap_mode="r")
    R2 = np.load(config.WORK_DIR / f"emb_{split}_source2.npy", mmap_mode="r")
    R3 = np.load(config.WORK_DIR / f"emb_{split}_source3.npy", mmap_mode="r")
    if verbose:
        print(f"[emb] left {L.shape} right {R2.shape}+{R3.shape} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    n2 = R2.shape[0]

    df = pd.read_parquet(cands_path)
    n = len(df)
    lid = df["source1_entity_id"].to_numpy()
    cid = df["cand_id"].to_numpy()
    cos = np.empty(n, np.float32)

    for lo in range(0, n, CHUNK):
        hi = min(lo + CHUNK, n)
        li = np.fromiter((lmap[s] for s in lid[lo:hi]), np.int64, hi - lo)
        pos = np.searchsorted(rkeys, cid[lo:hi])
        np.clip(pos, 0, len(rkeys) - 1, out=pos)
        ok = rkeys[pos] == cid[lo:hi]
        ri = np.where(ok, rrows[pos], 0)

        a = np.asarray(L[li], dtype=np.float32)
        b = np.empty((hi - lo, a.shape[1]), np.float32)
        in2 = ri < n2
        if in2.any():
            b[in2] = np.asarray(R2[ri[in2]], dtype=np.float32)
        if (~in2).any():
            b[~in2] = np.asarray(R3[ri[~in2] - n2], dtype=np.float32)
        c = np.einsum("ij,ij->i", a, b, optimize=True).astype(np.float32)
        c[~ok] = np.nan          # candidate not found: missing, not dissimilar
        cos[lo:hi] = c
        del a, b, c, li, pos, ok, ri
        if verbose:
            print(f"[emb] {hi:,}/{n:,} [{time.time() - t0:.0f}s]", flush=True)

    df["emb_cos"] = cos
    dst = out_path or cands_path
    df.to_parquet(dst, index=False)
    good = np.isfinite(cos)
    if verbose:
        print(f"[emb] emb_cos mean {cos[good].mean():.4f} "
              f"(n={int(good.sum()):,}) -> {dst} [{time.time() - t0:.0f}s]",
              flush=True)
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    add(a.cands, a.split, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
