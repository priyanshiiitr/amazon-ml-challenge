"""Full-scale blocking: stream right-hand shards from parquet, keep top-K.

Nothing here ever holds the whole right side. Shards are read from parquet in
row batches, filtered to one country, indexed, queried against the entire left
side, and dropped. Candidate ids are carried as int64 (source tag in the high
bits) rather than as Python strings -- at test scale there are ~43M of them and
the string form alone would cost several GB.

  python -m src.run_blocking train --sample 50000
  python -m src.run_blocking test
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .retrieve import (ShardIndex, extract_keys, query_shard, rescore_pairs,
                       topk_per_row)

COLS = ["entity_id", "country_key", "n_name", "s_name", "n_addr", "postal"]


def enc_ids(ids) -> np.ndarray:
    """'S2-681193310' -> int64 keeping the source separable."""
    a = np.asarray(ids, dtype=object)
    src = np.fromiter((int(s[1]) for s in a), dtype=np.int64, count=len(a))
    num = np.fromiter((int(s[3:]) for s in a), dtype=np.int64, count=len(a))
    return (src << 32) | num


def dec_ids(v: np.ndarray) -> np.ndarray:
    src = (v >> 32).astype(np.int64)
    num = (v & 0xFFFFFFFF).astype(np.int64)
    return np.asarray([f"S{s}-{n}" for s, n in zip(src.tolist(), num.tolist())],
                      dtype=object)


def iter_shards(paths, country, batch_rows):
    """Yield country-filtered right-hand shards, one batch at a time."""
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=batch_rows, columns=COLS):
            df = batch.to_pandas()
            df = df[df["country_key"] == country]
            if len(df):
                yield df.reset_index(drop=True)
            del batch


def _first_shard_norm(paths, country, batch_rows, lrows, lhash, n_left):
    """Per-channel L2 norms of each left row in IDF space.

    Returns a 2 x n_left array: row 0 over name-derived keys, row 1 over
    address-derived keys, matching the channel split in query_shard.

    IDF is log(1 + N/df) and both N and df scale with shard size, so a single
    shard is an unbiased estimate of the global weight; what matters is that
    every shard then uses the *same* one, or scores would not be comparable
    across the shards we merge.
    """
    for shard in iter_shards(paths, country, batch_rows):
        idx = ShardIndex(shard, verbose=False)
        if idx.empty:
            continue
        pos = np.searchsorted(idx.uniq, lhash)
        np.clip(pos, 0, len(idx.uniq) - 1, out=pos)
        ok = idx.uniq[pos] == lhash
        w2 = np.zeros(len(lhash), dtype=np.float64)
        w2[ok] = idx.idf[pos[ok]] ** 2
        ch = np.full(len(lhash), -1, dtype=np.int8)
        ch[ok] = idx.chan[pos[ok]]
        out = np.empty((2, n_left), np.float32)
        for c in (0, 1):
            nrm = np.sqrt(np.bincount(lrows, weights=w2 * (ch == c),
                                      minlength=n_left)).astype(np.float32)
            np.maximum(nrm, 1e-6, out=nrm)
            out[c] = nrm
        return out
    return np.ones((2, n_left), np.float32)


def _block_country_parallel(left, paths, country, k, batch_rows,
                            lrows, lhash, verbose, workers, wide=0):
    import multiprocessing as mp

    t0 = time.time()
    lnorm = _first_shard_norm(paths, country, batch_rows, lrows, lhash, len(left))

    best_l = np.empty(0, np.int32)
    best_r = np.empty(0, np.int64)
    best_s = np.empty(0, np.float32)
    ctx = mp.get_context("fork")
    done = 0
    l_skel = left["s_name"].fillna("").to_numpy()
    l_addr = left["n_addr"].fillna("").to_numpy()
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(lrows, lhash, lnorm, k, config.BLOCK_CHUNK,
                            wide, l_skel, l_addr)) as pool:
        for sl, sr, ss in pool.imap_unordered(
                _query_one, iter_shards(paths, country, batch_rows)):
            done += 1
            if len(sl):
                best_l = np.concatenate([best_l, sl])
                best_r = np.concatenate([best_r, sr])
                best_s = np.concatenate([best_s, ss])
                best_l, best_r, best_s = topk_per_row(best_l, best_r, best_s, k)
            if verbose and done % 4 == 0:
                print(f"    shard {done}: kept {len(best_l):,}  "
                      f"[{time.time() - t0:.0f}s]", flush=True)
    if verbose:
        print(f"    {country}: {done} shards, {len(best_l):,} pairs "
              f"in {time.time() - t0:.0f}s ({workers} workers)", flush=True)
    return best_l, best_r, best_s


_W = {}     # worker state, inherited through fork


def _init_worker(lrows, lhash, lnorm, k, chunk, wide, l_skel, l_addr):
    _W.update(lrows=lrows, lhash=lhash, lnorm=lnorm, k=k, chunk=chunk,
              wide=wide, l_skel=l_skel, l_addr=l_addr)


def _query_one(shard):
    """Run in a worker: index one shard, query the whole left side against it.

    When `wide > k`, the shard is queried for `wide` candidates by the cheap IDF
    cosine and those are then rescored with real string similarity before being
    cut to `k`. Rescoring here rather than in the parent is what keeps it
    affordable: the shard's text is already in memory, so no lookup of
    right-hand attributes by id is needed anywhere.
    """
    idx = ShardIndex(shard, verbose=False)
    if idx.empty:
        return (np.empty(0, np.int32), np.empty(0, np.int64),
                np.empty(0, np.float32))
    wide = max(_W["wide"], _W["k"])
    sl, sr, ss = query_shard(_W["lrows"], _W["lhash"], _W["lnorm"], idx,
                             wide, _W["chunk"])
    if wide > _W["k"] and len(sl):
        r_skel = shard["s_name"].fillna("").to_numpy()
        r_addr = shard["n_addr"].fillna("").to_numpy()
        ss = rescore_pairs(_W["l_skel"], _W["l_addr"], r_skel, r_addr,
                           sl, sr, ss)
        sl, sr, ss = topk_per_row(sl, sr, ss, _W["k"])
    ids = enc_ids(shard["entity_id"].to_numpy())
    return sl, ids[sr], ss


def block_country(left: pd.DataFrame, paths, country, k, batch_rows,
                  verbose=True, workers=1, wide=0):
    """Top-k candidates for every left row of one country. Returns int64 ids.

    Each shard is independent -- index, query, discard -- so with `workers > 1`
    they run in parallel and only the top-K merge happens in the parent. The
    left-side norm is computed once here and inherited by every worker, so
    scores stay comparable across shards; letting each worker derive its own
    would rescale one left row differently per shard and corrupt the merge.
    """
    t0 = time.time()
    lrows, lhash, _ = extract_keys(left)
    if verbose:
        print(f"    left keys: {len(lhash):,} "
              f"({len(lhash) / max(len(left), 1):.1f} per entity)", flush=True)

    if workers > 1:
        return _block_country_parallel(left, paths, country, k, batch_rows,
                                       lrows, lhash, verbose, workers, wide)

    # same per-channel norms as the parallel path, so both agree
    lnorm = _first_shard_norm(paths, country, batch_rows, lrows, lhash, len(left))
    best_l = np.empty(0, np.int32)
    best_r = np.empty(0, np.int64)
    best_s = np.empty(0, np.float32)
    n_shard = 0
    touched = 0

    for shard in iter_shards(paths, country, batch_rows):
        n_shard += 1
        idx = ShardIndex(shard, verbose=False)
        if idx.empty:
            continue

        sl, sr, ss = query_shard(lrows, lhash, lnorm, idx, k, config.BLOCK_CHUNK)
        shard_ids = enc_ids(shard["entity_id"].to_numpy())
        del idx, shard
        touched += len(sl)
        if len(sl):
            best_l = np.concatenate([best_l, sl])
            best_r = np.concatenate([best_r, shard_ids[sr]])
            best_s = np.concatenate([best_s, ss])
            best_l, best_r, best_s = topk_per_row(best_l, best_r, best_s, k)
        del shard_ids
        if verbose and n_shard % 2 == 0:
            print(f"    shard {n_shard}: kept {len(best_l):,}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    if verbose:
        print(f"    {country}: {n_shard} shards, {len(best_l):,} pairs "
              f"in {time.time() - t0:.0f}s", flush=True)
    return best_l, best_r, best_s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--sample", type=int, default=0,
                    help="use only the first N source1 entities")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--batch-rows", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel shard queries; each holds one shard index")
    ap.add_argument("--wide", type=int, default=0,
                    help="retrieve this many per shard by IDF, rescore with "
                         "string similarity, then cut to --top-k (0 = off)")
    a = ap.parse_args(argv)

    k = a.top_k or config.BLOCK_TOPK_FORWARD
    batch_rows = a.batch_rows or config.SHARD_SIZE
    t0 = time.time()

    s1 = pq.read_table(config.WORK_DIR / f"{a.split}_source1.parquet",
                       columns=COLS).to_pandas()
    if a.sample:
        s1 = s1.head(a.sample).reset_index(drop=True)
    right_paths = [config.WORK_DIR / f"{a.split}_source2.parquet",
                   config.WORK_DIR / f"{a.split}_source3.parquet"]
    print(f"[blocking] split={a.split} left={len(s1):,} k={k} "
          f"shard={batch_rows:,} workers={a.workers} wide={a.wide}", flush=True)

    parts = []
    for country in sorted(s1["country_key"].unique()):
        L = s1[s1["country_key"] == country].reset_index(drop=True)
        print(f"  country={country!r}: {len(L):,} entities", flush=True)
        li, ri, sc = block_country(L, right_paths, country, k, batch_rows,
                                   workers=a.workers, wide=a.wide)
        if len(li) == 0:
            continue
        parts.append(pd.DataFrame({
            "source1_entity_id": L["entity_id"].to_numpy()[li],
            "cand_id": ri,
            "sim_block": sc,
        }))
        del L

    out = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=["source1_entity_id", "cand_id", "sim_block"]))
    dst = a.out or (config.WORK_DIR / f"cands_{a.split}.parquet")
    out.to_parquet(dst, index=False)
    print(f"[blocking] {len(out):,} pairs "
          f"({len(out) / max(len(s1), 1):.2f} per entity) -> {dst}", flush=True)
    print(f"[blocking] total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
