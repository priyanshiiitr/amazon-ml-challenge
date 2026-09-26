"""Document frequency of skeleton name tokens, for a name-rarity feature.

Error analysis showed the model's confident false merges are concentrated on
*generic* names -- "new delhi india", "solan", "blue trading", "step brothers" --
where an exact name match carries almost no information because thousands of
records share it. The pair features treat every matching token alike, so the
model cannot tell that apart from a match on "garware educational society".

The blocking index already weights keys by IDF, but that weight is consumed
inside a single cosine and never reaches the classifier. This exposes it.

Built once per split over that split's Source 2/3 records, so train and test
each use their own statistics and nothing leaks between them.

  python -m src.tokidf train
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

from . import config


def build(split: str) -> dict:
    df: Counter = Counter()
    n_docs = 0
    for nm in ("source2", "source3"):
        pf = pq.ParquetFile(config.WORK_DIR / f"{split}_{nm}.parquet")
        for batch in pf.iter_batches(batch_size=500_000, columns=["s_name"]):
            col = batch.column(0).to_pylist()
            n_docs += len(col)
            for s in col:
                if s:
                    df.update(set(s.split()))
    out = {"n_docs": n_docs, "df": dict(df)}
    path = config.MODEL_DIR / f"tokidf_{split}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print(f"[tokidf] {split}: {n_docs:,} docs, {len(df):,} distinct tokens "
          f"-> {path}", flush=True)
    return out


_CACHE: dict = {}


def load(split: str):
    if split not in _CACHE:
        path = config.MODEL_DIR / f"tokidf_{split}.pkl"
        if not path.exists():
            _CACHE[split] = None
        else:
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            n = max(d["n_docs"], 1)
            _CACHE[split] = (n, d["df"])
    return _CACHE[split]


def idf_stats(names, split: str):
    """-> (min_idf, mean_idf, max_idf) per name; NaN when unknown."""
    tbl = load(split)
    n = len(names)
    out = np.full((3, n), np.nan, np.float32)
    if tbl is None:
        return out
    n_docs, df = tbl
    ln = np.log
    for i, s in enumerate(names):
        if not s:
            continue
        vals = [ln(1.0 + n_docs / max(df.get(t, 1), 1)) for t in set(s.split())]
        if vals:
            out[0, i] = min(vals)
            out[1, i] = sum(vals) / len(vals)
            out[2, i] = max(vals)
    return out


if __name__ == "__main__":
    sys.exit(0 if build(sys.argv[1] if len(sys.argv) > 1 else "train") else 1)
