"""Multi-view retrieval: each view gets its own top-K quota, then union.

Our single-ranking retrieval blends name keys and address keys into one IDF
cosine and takes the top K. That dilutes exactly the pairs we most need: when a
business name has been replaced by an unrelated token, the address evidence has
to outrank every same-street competitor *inside one shared ranking*, and it
loses. 68.6% of links no key could reach were of that form.

Combining the two channels differently does not fix it -- scoring name and
address as separate cosines and taking a max-dominant combination measured
0.8498 against 0.9073, because it still yields a single ranking and lets
same-street pairs flood in.

Giving each view its own quota does fix it. The address view contributes its
top-K regardless of the name, so a name-destroyed true pair competes only
against other addresses, not against the whole blended field. The union is then
larger per entity but strictly more diverse.

Views (tags refer to retrieve.extract_keys):
    name    F, T, B, G      whole-name skeleton, tokens, bigrams, shingles
    addr    PA, NA, AB      postal/number/word address keys only
    both    all of the above (the previous single-view behaviour)

Reverse retrieval is a fourth view, produced separately by reverse.py.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .retrieve import (TAG_AB, TAG_B, TAG_F, TAG_G, TAG_NA, TAG_PA, TAG_T,
                       ShardIndex, extract_keys, query_shard, topk_per_row)
from .run_blocking import COLS, enc_ids, iter_shards

VIEWS = {
    "name": (TAG_F, TAG_T, TAG_B, TAG_G),
    "addr": (TAG_PA, TAG_NA, TAG_AB),
    "both": (TAG_F, TAG_T, TAG_B, TAG_G, TAG_PA, TAG_NA, TAG_AB),
}


def _filter_keys(rows, hashes, tags, keep_tags):
    m = np.isin(tags, list(keep_tags))
    return rows[m], hashes[m], tags[m]


def _left_norms(rows, hashes, index, n_left):
    pos = np.searchsorted(index.uniq, hashes)
    np.clip(pos, 0, len(index.uniq) - 1, out=pos)
    ok = index.uniq[pos] == hashes
    w2 = np.zeros(len(hashes), np.float64)
    w2[ok] = index.idf[pos[ok]] ** 2
    ch = np.full(len(hashes), -1, np.int8)
    ch[ok] = index.chan[pos[ok]]
    out = np.empty((2, n_left), np.float32)
    for c in (0, 1):
        n = np.sqrt(np.bincount(rows, weights=w2 * (ch == c),
                                minlength=n_left)).astype(np.float32)
        np.maximum(n, 1e-6, out=n)
        out[c] = n
    return out


def retrieve_view(left, right_paths, country, view, k, batch_rows, verbose=True):
    """Top-k per left row using only this view's keys."""
    t0 = time.time()
    keep = VIEWS[view]
    lrows, lhash, ltags = extract_keys(left)
    lrows, lhash, _ = _filter_keys(lrows, lhash, ltags, keep)
    if len(lrows) == 0:
        return None
    if verbose:
        print(f"    [{view}] {len(lhash):,} left keys "
              f"({len(lhash) / max(len(left), 1):.1f}/entity)", flush=True)

    lnorm = None
    bl = np.empty(0, np.int32); br = np.empty(0, np.int64)
    bs = np.empty(0, np.float32)
    n = 0
    for shard in iter_shards(right_paths, country, batch_rows):
        idx = ShardIndex(shard, verbose=False, keep_tags=keep)
        if idx.empty:
            continue
        if lnorm is None:
            lnorm = _left_norms(lrows, lhash, idx, len(left))
        sl, sr, ss = query_shard(lrows, lhash, lnorm, idx, k,
                                 config.BLOCK_CHUNK)
        ids = enc_ids(shard["entity_id"].to_numpy())
        del idx, shard
        if len(sl):
            bl = np.concatenate([bl, sl])
            br = np.concatenate([br, ids[sr]])
            bs = np.concatenate([bs, ss])
            bl, br, bs = topk_per_row(bl, br, bs, k)
        n += 1
    if verbose:
        print(f"    [{view}] {country}: {len(bl):,} pairs, {n} shards "
              f"[{time.time() - t0:.0f}s]", flush=True)
    if len(bl) == 0:
        return None
    return pd.DataFrame({
        "source1_entity_id": left["entity_id"].to_numpy()[bl],
        "cand_id": br,
        f"sim_{view}": bs,
        f"rank_{view}": np.zeros(len(bl), np.int16),
    })


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--k-name", type=int, default=40)
    ap.add_argument("--k-addr", type=int, default=40)
    ap.add_argument("--k-both", type=int, default=40)
    ap.add_argument("--batch-rows", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    batch = a.batch_rows or config.SHARD_SIZE
    t0 = time.time()

    s1 = pq.read_table(config.WORK_DIR / f"{a.split}_source1.parquet",
                       columns=COLS).to_pandas()
    if a.sample:
        s1 = s1.head(a.sample).reset_index(drop=True)
    paths = [config.WORK_DIR / f"{a.split}_source2.parquet",
             config.WORK_DIR / f"{a.split}_source3.parquet"]
    ks = {"name": a.k_name, "addr": a.k_addr, "both": a.k_both}
    print(f"[multiview] split={a.split} S1={len(s1):,} K={ks}", flush=True)

    per_country = []
    for country in sorted(s1["country_key"].unique()):
        L = s1[s1["country_key"] == country].reset_index(drop=True)
        print(f"  country={country!r}: {len(L):,}", flush=True)
        frames = []
        for view, k in ks.items():
            if k <= 0:
                continue
            f = retrieve_view(L, paths, country, view, k, batch)
            if f is not None:
                frames.append(f)
        if not frames:
            continue
        out = frames[0]
        for f in frames[1:]:
            out = out.merge(f, on=["source1_entity_id", "cand_id"], how="outer")
        per_country.append(out)
        del L, frames

    res = pd.concat(per_country, ignore_index=True)
    for v in ks:
        c = f"sim_{v}"
        if c in res.columns:
            res[c] = res[c].fillna(0.0).astype(np.float32)
        else:
            res[c] = np.float32(0.0)
    # a single blended score so downstream pruning and features keep working
    res["sim_block"] = res[[f"sim_{v}" for v in ks]].max(axis=1).astype(np.float32)
    res["n_views"] = sum((res[f"sim_{v}"] > 0).astype(np.int8) for v in ks)
    res.to_parquet(a.out, index=False)
    print(f"[multiview] {len(res):,} pairs "
          f"({len(res) / max(len(s1), 1):.2f} per entity) -> {a.out}", flush=True)
    print(f"[multiview] total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
