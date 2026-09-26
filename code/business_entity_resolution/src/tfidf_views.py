"""Character n-gram TF-IDF retrieval, several views, each with its own top-K.

This exists because the blocking oracle is the binding constraint. Ours is
0.96585; a comparable published pipeline reports 0.99645 from char n-gram TF-IDF
views. No amount of feature work or stacking can pass our own oracle, so
retrieval is where the remaining score lives.

Why our discrete-key blocker falls short: keys are whole skeleton tokens,
bigrams and *subsampled* 4-gram shingles capped at df 250. Character 3-grams are
far denser and degrade gracefully under typos, truncation and transliteration
drift, where whole-token equality fails outright. The discrete-key design was
chosen when the machine had 8 GB; with 216 GB the constraint that forced it is
gone.

Each view retrieves its own top-K and the results are unioned, so a pair that is
strong on address alone is not forced to outrank the field on a blended score --
the failure mode behind 68.6% of unreachable links.

Views:
    name      char_wb 3-gram over the normalised name
    nospace   char 3-gram over the name with spaces removed (word-order proof)
    addr      word-level over the normalised address
    hybrid    char_wb 3-gram over name + address

  python -m src.tfidf_views train --sample 50000 --out work/tf50k.parquet
"""
from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .run_blocking import COLS, enc_ids

try:
    from sparse_dot_topn import sp_matmul_topn
    HAVE_TOPN = True
except Exception:                                    # pragma: no cover
    HAVE_TOPN = False

VIEW_SPEC = {
    "name":    dict(field="n_name", analyzer="char_wb", ngram=(3, 3)),
    "nospace": dict(field="_nospace", analyzer="char", ngram=(3, 3)),
    "addr":    dict(field="n_addr", analyzer="word", ngram=(1, 1)),
    "hybrid":  dict(field="_blob", analyzer="char_wb", ngram=(3, 3)),
}
MAX_FEATURES = 300_000
LEFT_CHUNK = 20_000


def _field(df, name):
    if name == "_nospace":
        return df["n_name"].fillna("").str.replace(" ", "", regex=False)
    if name == "_blob":
        return (df["n_name"].fillna("") + " " + df["n_addr"].fillna("")).str.strip()
    return df[name].fillna("")


def _topk(Lm, Rt, k, min_sim=0.05):
    """Top-k columns of Rt for each row of Lm. Returns (row, col, score)."""
    rows, cols, vals = [], [], []
    for lo in range(0, Lm.shape[0], LEFT_CHUNK):
        hi = min(lo + LEFT_CHUNK, Lm.shape[0])
        sub = Lm[lo:hi]
        if HAVE_TOPN:
            C = sp_matmul_topn(sub, Rt, top_n=k, threshold=min_sim, sort=True)
        else:                                        # pragma: no cover
            C = (sub @ Rt).tocsr()
        C = C.tocoo()
        m = C.data >= min_sim
        rows.append(C.row[m] + lo)
        cols.append(C.col[m])
        vals.append(C.data[m])
        del C, sub
    if not rows:
        return (np.empty(0, np.int32),) * 2 + (np.empty(0, np.float32),)
    return (np.concatenate(rows).astype(np.int32),
            np.concatenate(cols).astype(np.int32),
            np.concatenate(vals).astype(np.float32))


def run_view(left, right, view, k, verbose=True):
    spec = VIEW_SPEC[view]
    t0 = time.time()
    lt = _field(left, spec["field"]).to_numpy()
    rt = _field(right, spec["field"]).to_numpy()

    vec = TfidfVectorizer(analyzer=spec["analyzer"], ngram_range=spec["ngram"],
                          min_df=2, max_features=MAX_FEATURES,
                          sublinear_tf=True, dtype=np.float32)
    vec.fit(np.concatenate([lt, rt[::7]]))     # subsample the right for the fit
    Lm = vec.transform(lt)
    Rm = vec.transform(rt)
    del lt, rt
    Rt = Rm.T.tocsr()
    del Rm
    gc.collect()
    if verbose:
        print(f"    [{view}] vocab {len(vec.vocabulary_):,}, "
              f"L {Lm.shape} nnz {Lm.nnz:,}, Rt nnz {Rt.nnz:,} "
              f"[{time.time() - t0:.0f}s]", flush=True)

    li, ri, sc = _topk(Lm, Rt, k)
    del Lm, Rt, vec
    gc.collect()
    if verbose:
        print(f"    [{view}] {len(li):,} pairs "
              f"({len(li) / max(len(left), 1):.1f}/entity) "
              f"[{time.time() - t0:.0f}s]", flush=True)
    if len(li) == 0:
        return None
    return pd.DataFrame({
        "source1_entity_id": left["entity_id"].to_numpy()[li],
        "cand_id": enc_ids(right["entity_id"].to_numpy())[ri],
        f"tf_{view}": sc,
    })


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--k-name", type=int, default=40)
    ap.add_argument("--k-nospace", type=int, default=20)
    ap.add_argument("--k-addr", type=int, default=40)
    ap.add_argument("--k-hybrid", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    ks = {"name": a.k_name, "nospace": a.k_nospace,
          "addr": a.k_addr, "hybrid": a.k_hybrid}
    print(f"[tfidf] split={a.split} K={ks} topn_backend={HAVE_TOPN}", flush=True)

    s1 = pq.read_table(config.WORK_DIR / f"{a.split}_source1.parquet",
                       columns=COLS).to_pandas()
    if a.sample:
        s1 = s1.head(a.sample).reset_index(drop=True)

    out = []
    for country in sorted(s1["country_key"].unique()):
        L = s1[s1["country_key"] == country].reset_index(drop=True)
        R = []
        for nm in ("source2", "source3"):
            t = pq.read_table(config.WORK_DIR / f"{a.split}_{nm}.parquet",
                              columns=COLS).to_pandas()
            R.append(t[t["country_key"] == country])
            del t
        R = pd.concat(R, ignore_index=True)
        print(f"  country={country!r}: {len(L):,} x {len(R):,}", flush=True)

        frames = [f for f in
                  (run_view(L, R, v, k) for v, k in ks.items() if k > 0)
                  if f is not None]
        del R
        gc.collect()
        if not frames:
            continue
        m = frames[0]
        for f in frames[1:]:
            m = m.merge(f, on=["source1_entity_id", "cand_id"], how="outer")
        out.append(m)
        del L, frames, m
        gc.collect()

    res = pd.concat(out, ignore_index=True)
    for v in ks:
        c = f"tf_{v}"
        res[c] = (res[c].fillna(0.0).astype(np.float32) if c in res.columns
                  else np.float32(0.0))
    res["sim_block"] = res[[f"tf_{v}" for v in ks]].max(axis=1).astype(np.float32)
    res["n_views"] = sum((res[f"tf_{v}"] > 0).astype(np.int8) for v in ks)
    res.to_parquet(a.out, index=False)
    print(f"[tfidf] {len(res):,} pairs "
          f"({len(res) / max(len(s1), 1):.2f} per entity) -> {a.out}", flush=True)
    print(f"[tfidf] total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
