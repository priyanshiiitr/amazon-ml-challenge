"""Memory-bounded candidate retrieval at full scale.

Two changes make this fit in 8 GB where the straightforward version does not.

*Hashed keys.* The obvious implementation maps key strings to ids with a dict.
At test scale that dict holds ~10M distinct strings and costs well over 1.5 GB
by itself. Instead every key is hashed to an int64 as soon as it is built, the
string is discarded, and lookup becomes `np.searchsorted` over a sorted array.
Everything downstream is plain numpy.

*Right-side sharding.* The right side (~10M records) is indexed one shard at a
time. Each shard is queried against the whole left side, its top-K written to
disk, and the shards merged at the end. Peak memory is therefore set by the
shard size, not by the size of the data.

Hashes come from Python's `hash()`, which is SipHash-64 and is stable within a
single process -- index construction and querying always happen in the same
run, so this never crosses a process boundary.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import config

# tag ids, parallel to the key arrays, so each channel keeps its own df cap
TAG_F, TAG_T, TAG_B, TAG_G, TAG_PA, TAG_NA, TAG_AB = range(7)
TAG_NAMES = {TAG_F: "F", TAG_T: "T", TAG_B: "B", TAG_G: "G",
             TAG_PA: "P", TAG_NA: "N", TAG_AB: "A"}

_GENERIC_ADDR = {
    "street", "road", "avenue", "lane", "drive", "court", "place", "square",
    "boulevard", "circle", "terrace", "trail", "parkway", "highway", "floor",
    "suite", "apartment", "unit", "building", "room", "number", "near",
    "opposite", "behind", "block", "phase", "sector", "nagar", "colony",
    "north", "south", "east", "west", "new", "old", "post", "office", "rue",
}

_MASK = (1 << 62) - 1
SHINGLE_KEEP = 3          # keep a shingle when (hash & 3) == 0, i.e. ~25%


def _h(s: str) -> int:
    return hash(s) & _MASK


def extract_keys(df: pd.DataFrame):
    """-> (row_idx int32, key_hash int64, tag int8), one entry per key.

    Shingles are deterministically subsampled to a quarter. Both sides sample
    by the same hash predicate, so a shared shingle is kept on both sides or
    neither; this cuts the widest channel's volume 4x without biasing it.
    """
    rows: list[int] = []
    hashes: list[int] = []
    tags: list[int] = []
    ra, ha, ta = rows.append, hashes.append, tags.append

    s_names = df["s_name"].to_numpy()
    postals = df["postal"].to_numpy()
    addrs = df["n_addr"].to_numpy()

    for i in range(len(s_names)):
        sn = s_names[i]
        if sn:
            toks = sn.split()
            if toks:
                if len(toks) <= 8:
                    ra(i); ha(_h("F" + "".join(sorted(toks)))); ta(TAG_F)
                seen = set()
                for t in toks:
                    if len(t) >= 2 and t not in seen:
                        seen.add(t)
                        ra(i); ha(_h("T" + t)); ta(TAG_T)
                for a, b in zip(toks, toks[1:]):
                    ra(i); ha(_h("B" + a + b)); ta(TAG_B)
                flat = sn.replace(" ", "")
                if len(flat) >= 4:
                    for j in range(len(flat) - 3):
                        g = _h("G" + flat[j:j + 4])
                        if not (g & SHINGLE_KEEP):
                            ra(i); ha(g); ta(TAG_G)

        ad = addrs[i]
        if not ad:
            continue
        atoks = ad.split()
        nums = [t for t in atoks if t[0].isdigit()]
        words = [t for t in atoks
                 if not t[0].isdigit() and len(t) >= 3 and t not in _GENERIC_ADDR]
        if not words:
            continue
        p = postals[i]
        if p:
            for w in words[:5]:
                ra(i); ha(_h("P" + p + "|" + w)); ta(TAG_PA)
        for num in nums[:2]:
            base = num.rstrip("abcdefghijklmnopqrstuvwxyz")
            if base:
                for w in words[:4]:
                    ra(i); ha(_h("N" + base + "|" + w)); ta(TAG_NA)
        if not nums:
            for a, b in zip(words, words[1:]):
                ra(i); ha(_h("A" + a + "|" + b)); ta(TAG_AB)

    return (np.asarray(rows, dtype=np.int32),
            np.asarray(hashes, dtype=np.int64),
            np.asarray(tags, dtype=np.int8))


class ShardIndex:
    """Postings over one right-hand shard, keyed by sorted unique key hash."""

    def __init__(self, right: pd.DataFrame, verbose=False):
        rows, hashes, tags = extract_keys(right)
        n_right = len(right)
        if len(rows) == 0:
            self.empty = True
            return
        self.empty = False

        uniq, inv = np.unique(hashes, return_inverse=True)
        inv = inv.astype(np.int32)
        df = np.bincount(inv, minlength=len(uniq)).astype(np.int32)

        # per-channel frequency cap; a key's tag is constant across its rows
        cap_of = np.full(len(uniq), config.DF_CAP, dtype=np.int32)
        first_tag = np.zeros(len(uniq), dtype=np.int8)
        first_tag[inv[::-1]] = tags[::-1]
        for tid, nm in TAG_NAMES.items():
            cap_of[first_tag == tid] = config.DF_CAP_BY_TAG.get(nm, config.DF_CAP)

        keep = df <= cap_of
        mask = keep[inv]
        rows, inv = rows[mask], inv[mask]
        del hashes, tags, mask

        order = np.argsort(inv, kind="stable")
        self.postings = rows[order]
        sorted_keys = inv[order]
        counts = np.bincount(sorted_keys, minlength=len(uniq))
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.uniq = uniq

        self.idf = np.log(1.0 + n_right / np.maximum(df, 1)).astype(np.float32)
        w2 = self.idf[sorted_keys] ** 2
        self.norm = np.sqrt(
            np.bincount(self.postings, weights=w2, minlength=n_right)
        ).astype(np.float32)
        np.maximum(self.norm, 1e-6, out=self.norm)
        self.n_right = n_right
        if verbose:
            print(f"    [shard] {n_right:,} rows, {len(uniq):,} keys, "
                  f"{len(self.postings):,} postings", flush=True)


def topk_per_row(lrow, rcol, score, k):
    """Keep the k best rcol per lrow. Arrays need not be sorted."""
    if len(lrow) == 0:
        return lrow, rcol, score
    order = np.lexsort((-score, lrow))
    lrow, rcol, score = lrow[order], rcol[order], score[order]
    starts = np.flatnonzero(np.r_[True, lrow[1:] != lrow[:-1]]).astype(np.int64)
    group_start = np.repeat(starts, np.diff(np.r_[starts, np.int64(len(lrow))]))
    keep = (np.arange(len(lrow), dtype=np.int64) - group_start) < k
    return lrow[keep], rcol[keep], score[keep]


def query_shard(lrows, lhash, lnorm, index: ShardIndex, k: int, chunk: int):
    """Top-k rows of the shard for each left row, by cosine over IDF weights."""
    if index.empty:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float32))

    pos = np.searchsorted(index.uniq, lhash)
    np.clip(pos, 0, len(index.uniq) - 1, out=pos)
    ok = index.uniq[pos] == lhash
    lr, lk = lrows[ok], pos[ok].astype(np.int32)
    if len(lr) == 0:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float32))

    order = np.argsort(lr, kind="stable")
    lr, lk = lr[order], lk[order]
    n_left = int(lrows.max()) + 1 if len(lrows) else 0
    bounds = np.searchsorted(lr, np.arange(0, n_left + 1))

    out_l, out_r, out_s = [], [], []
    for lo in range(0, n_left, chunk):
        hi = min(lo + chunk, n_left)
        a, b = bounds[lo], bounds[hi]
        if a == b:
            continue
        ck_rows, ck_keys = lr[a:b], lk[a:b]
        starts = index.indptr[ck_keys]
        lens = (index.indptr[ck_keys + 1] - starts).astype(np.int64)
        total = int(lens.sum())
        if total == 0:
            continue

        left_rep = np.repeat(ck_rows, lens)
        w_rep = np.repeat(index.idf[ck_keys], lens)
        offs = (np.arange(total, dtype=np.int64)
                - np.repeat(np.cumsum(np.r_[0, lens[:-1]]), lens))
        right_rep = index.postings[np.repeat(starts, lens) + offs]
        del offs

        comb = left_rep.astype(np.int64) * index.n_right + right_rep
        del left_rep, right_rep
        o = np.argsort(comb, kind="stable")
        comb, w_rep = comb[o], w_rep[o]
        del o
        bnd = np.flatnonzero(np.r_[True, comb[1:] != comb[:-1]])
        pair_score = np.add.reduceat(w_rep, bnd).astype(np.float32)
        pc = comb[bnd]
        del comb, w_rep, bnd

        pl = (pc // index.n_right).astype(np.int32)
        pr = (pc % index.n_right).astype(np.int32)
        del pc
        pair_score /= lnorm[pl] * index.norm[pr]
        pl, pr, ps = topk_per_row(pl, pr, pair_score, k)
        out_l.append(pl); out_r.append(pr); out_s.append(ps)

    if not out_l:
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float32))
    return (np.concatenate(out_l), np.concatenate(out_r),
            np.concatenate(out_s))


def retrieve(left: pd.DataFrame, right: pd.DataFrame, k: int,
             shard_size: int | None = None, chunk: int | None = None,
             verbose=True):
    """Top-k right rows per left row, sharding the right side.

    Returns (left_idx, right_idx, score) as positions into `left` and `right`.
    """
    shard_size = shard_size or config.SHARD_SIZE
    chunk = chunk or config.BLOCK_CHUNK
    t0 = time.time()

    lrows, lhash, _ = extract_keys(left)
    lnorm = None
    best_l = np.empty(0, np.int32)
    best_r = np.empty(0, np.int32)
    best_s = np.empty(0, np.float32)

    n_shards = (len(right) + shard_size - 1) // shard_size
    for si in range(n_shards):
        lo, hi = si * shard_size, min((si + 1) * shard_size, len(right))
        idx = ShardIndex(right.iloc[lo:hi], verbose=verbose)
        if idx.empty:
            continue
        if lnorm is None:
            # IDF weights differ per shard; the first shard's are a good enough
            # normaliser and keeping it fixed makes scores comparable across
            # shards, which is what the merge below requires
            pos = np.searchsorted(idx.uniq, lhash)
            np.clip(pos, 0, len(idx.uniq) - 1, out=pos)
            ok = idx.uniq[pos] == lhash
            w = np.zeros(len(lhash), dtype=np.float32)
            w[ok] = idx.idf[pos[ok]]
            lnorm = np.sqrt(
                np.bincount(lrows, weights=w.astype(np.float64),
                            minlength=len(left))
            ).astype(np.float32)
            np.maximum(lnorm, 1e-6, out=lnorm)
            del w, pos, ok

        sl, sr, ss = query_shard(lrows, lhash, lnorm, idx, k, chunk)
        del idx
        if len(sl):
            best_l = np.concatenate([best_l, sl])
            best_r = np.concatenate([best_r, sr + lo])
            best_s = np.concatenate([best_s, ss])
            best_l, best_r, best_s = topk_per_row(best_l, best_r, best_s, k)
        if verbose:
            print(f"  [retrieve] shard {si + 1}/{n_shards} "
                  f"({lo:,}-{hi:,})  running pairs {len(best_l):,}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    return best_l, best_r, best_s


def generate_candidates(s1: pd.DataFrame, right: pd.DataFrame,
                        top_k: int | None = None, verbose=True) -> pd.DataFrame:
    """Candidates for every source1 entity, retrieved within each country."""
    top_k = top_k or config.BLOCK_TOPK_FORWARD
    frames = []
    for g in sorted(set(s1["country_key"])):
        L = s1[s1["country_key"] == g].reset_index(drop=True)
        R = right[right["country_key"] == g].reset_index(drop=True)
        if verbose:
            print(f"[blocking] country={g!r}: {len(L):,} x {len(R):,}", flush=True)
        if len(R) == 0:
            if verbose:
                print("           no right-hand rows — forced singletons")
            continue
        li, ri, sc = retrieve(L, R, top_k, verbose=verbose)
        if len(li) == 0:
            continue
        frames.append(pd.DataFrame({
            "source1_entity_id": L["entity_id"].to_numpy()[li],
            "cand_entity_id": R["entity_id"].to_numpy()[ri],
            "sim_block": sc,
        }))
        del L, R
    if not frames:
        return pd.DataFrame(columns=["source1_entity_id", "cand_entity_id",
                                     "sim_block"])
    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"[blocking] {len(out):,} pairs "
              f"({len(out) / max(len(s1), 1):.1f} per entity)", flush=True)
    return out
