"""Reverse retrieval: query from Source 2/3 into a Source 1 index.

Forward retrieval asks each Source 1 entity to rank its true matches above ~10M
records. A record whose name was replaced ("NYLADREX") carries only address
evidence, so it competes with every other business on that street plus the 26%
unmatched noise, and sinks far below any affordable K. That is why reachability
is 0.993 while recall at K=1000 is only 0.950 -- the pairs are in the index and
are being out-ranked.

Reversing the direction changes the competition set. Each Source 2/3 record has
at most one owner (verified: 7,638,365 links over 7,638,365 distinct ids), and
it only has to rank that owner into its own top 1-3 against the 1.73M
*deduplicated* Source 1 entities -- typically a handful on the same street.
Noise records become queries instead of competitors.

Budget: reverse top-3 over 9.97M records is ~30M pairs, about 17 per Source 1
entity, so the union with a reduced forward K stays small. That matters now
that a smaller candidate set is itself a ranking criterion.

  python -m src.reverse test --top-k 3 --out work/rev_test.parquet
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .retrieve import ShardIndex, extract_keys, query_shard, topk_per_row
from .run_blocking import COLS, enc_ids, iter_shards


def _query_norms(qrows, qhash, index, n_q):
    """Per-channel IDF norms for the query side, using the index's weights."""
    pos = np.searchsorted(index.uniq, qhash)
    np.clip(pos, 0, len(index.uniq) - 1, out=pos)
    ok = index.uniq[pos] == qhash
    w2 = np.zeros(len(qhash), np.float64)
    w2[ok] = index.idf[pos[ok]] ** 2
    ch = np.full(len(qhash), -1, np.int8)
    ch[ok] = index.chan[pos[ok]]
    out = np.empty((2, n_q), np.float32)
    for c in (0, 1):
        nrm = np.sqrt(np.bincount(qrows, weights=w2 * (ch == c),
                                  minlength=n_q)).astype(np.float32)
        np.maximum(nrm, 1e-6, out=nrm)
        out[c] = nrm
    return out


def reverse_country(s1: pd.DataFrame, right_paths, country, k, batch_rows,
                    verbose=True):
    """Top-k Source 1 entities for each Source 2/3 record of one country."""
    t0 = time.time()
    index = ShardIndex(s1.reset_index(drop=True), verbose=False)
    if index.empty:
        return pd.DataFrame(columns=["source1_entity_id", "cand_id",
                                     "rev_score", "rev_rank"])
    s1_ids = s1["entity_id"].to_numpy()
    if verbose:
        print(f"    S1 index: {len(s1):,} entities, {len(index.uniq):,} keys",
              flush=True)

    parts = []
    n_batch = 0
    for batch in iter_shards(right_paths, country, batch_rows):
        n_batch += 1
        qrows, qhash, _ = extract_keys(batch)
        if len(qrows) == 0:
            continue
        qnorm = _query_norms(qrows, qhash, index, len(batch))
        qi, si, sc = query_shard(qrows, qhash, qnorm, index, k,
                                 config.BLOCK_CHUNK)
        if len(qi) == 0:
            continue
        # rank of each S1 owner within this record's own shortlist
        order = np.lexsort((-sc, qi))
        qi, si, sc = qi[order], si[order], sc[order]
        starts = np.flatnonzero(np.r_[True, qi[1:] != qi[:-1]]).astype(np.int64)
        gs = np.repeat(starts, np.diff(np.r_[starts, np.int64(len(qi))]))
        rank = (np.arange(len(qi), dtype=np.int64) - gs).astype(np.int8)

        parts.append(pd.DataFrame({
            "source1_entity_id": s1_ids[si],
            "cand_id": enc_ids(batch["entity_id"].to_numpy())[qi],
            "rev_score": sc,
            "rev_rank": rank,
        }))
        if verbose and n_batch % 4 == 0:
            print(f"    batch {n_batch}: {sum(len(p) for p in parts):,} pairs "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    del index
    if not parts:
        return pd.DataFrame(columns=["source1_entity_id", "cand_id",
                                     "rev_score", "rev_rank"])
    out = pd.concat(parts, ignore_index=True)
    if verbose:
        print(f"    {country}: {len(out):,} reverse pairs in "
              f"{time.time() - t0:.0f}s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--sample", type=int, default=0,
                    help="restrict the Source 1 index to the first N entities")
    ap.add_argument("--batch-rows", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    batch_rows = a.batch_rows or config.SHARD_SIZE
    t0 = time.time()

    s1 = pq.read_table(config.WORK_DIR / f"{a.split}_source1.parquet",
                       columns=COLS).to_pandas()
    if a.sample:
        s1 = s1.head(a.sample).reset_index(drop=True)
    right_paths = [config.WORK_DIR / f"{a.split}_source2.parquet",
                   config.WORK_DIR / f"{a.split}_source3.parquet"]
    print(f"[reverse] split={a.split} S1={len(s1):,} top_k={a.top_k}", flush=True)

    parts = []
    for country in sorted(s1["country_key"].unique()):
        L = s1[s1["country_key"] == country]
        print(f"  country={country!r}: {len(L):,} S1 entities", flush=True)
        parts.append(reverse_country(L, right_paths, country, a.top_k,
                                     batch_rows))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(a.out, index=False)
    print(f"[reverse] {len(out):,} pairs "
          f"({len(out) / max(len(s1), 1):.2f} per S1 entity) -> {a.out}",
          flush=True)
    print(f"[reverse] total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
