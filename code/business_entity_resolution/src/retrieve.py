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

import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process

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

    def __init__(self, right: pd.DataFrame, verbose=False, keep_tags=None):
        """`keep_tags` restricts the index to one retrieval view's channels."""
        rows, hashes, tags = extract_keys(right)
        if keep_tags is not None:
            m = np.isin(tags, list(keep_tags))
            rows, hashes, tags = rows[m], hashes[m], tags[m]
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

        # Channel of each key: 0 = derived from the name, 1 = from the address.
        # The two are scored and normalised separately (see query_shard). With a
        # single mixed vector, a pair whose name was replaced but whose address
        # survived gets its cosine crushed: the name keys contribute nothing to
        # the numerator yet still inflate both norms. That is exactly the 68.6%
        # of hard links measured as "name destroyed, address intact".
        self.chan = np.isin(first_tag, [TAG_PA, TAG_NA, TAG_AB]).astype(np.int8)

        w2 = self.idf[sorted_keys] ** 2
        ch = self.chan[sorted_keys]
        self.norm_n = np.sqrt(np.bincount(self.postings, weights=w2 * (ch == 0),
                                          minlength=n_right)).astype(np.float32)
        self.norm_a = np.sqrt(np.bincount(self.postings, weights=w2 * (ch == 1),
                                          minlength=n_right)).astype(np.float32)
        np.maximum(self.norm_n, 1e-6, out=self.norm_n)
        np.maximum(self.norm_a, 1e-6, out=self.norm_a)
        self.norm = np.sqrt(
            np.bincount(self.postings, weights=w2, minlength=n_right)
        ).astype(np.float32)
        np.maximum(self.norm, 1e-6, out=self.norm)
        self.n_right = n_right
        if verbose:
            print(f"    [shard] {n_right:,} rows, {len(uniq):,} keys, "
                  f"{len(self.postings):,} postings", flush=True)


def rescore_pairs(l_skel, l_addr, r_skel, r_addr, lrow, rcol, block_score,
                  w_min=0.35, w_block=0.05):
    """Replace the bag-of-keys score with real string similarity.

    The IDF cosine ranks poorly: against the full 10.3M right side, reachability
    at K=inf is 0.993 but recall at K=120 is only 0.907, so ~9% of true links are
    retrieved and then out-ranked.

    The combination is max-dominant, not a weighted sum, because the corruption
    is *disjunctive*. Measured on 138,165 true pairs, 99.78% have address
    similarity >=70 OR skeleton-name similarity >=70, but frequently not both:
    one field is destroyed while the other survives. A symmetric weighted sum
    scores (addr=100, name=20) -- a real match with a replaced name -- the same
    as (addr=60, name=60), which is usually noise. Taking the max as the primary
    term and the min only as a bonus preserves that asymmetry.

    An earlier symmetric version (0.45/0.45) scored 0.9069 against a 0.9073
    baseline, i.e. no better than the cosine it replaced.
    """
    if len(lrow) == 0:
        return np.empty(0, np.float32)
    a_s = l_skel[lrow]; b_s = r_skel[rcol]
    a_a = l_addr[lrow]; b_a = r_addr[rcol]
    s_skel = rf_process.cpdist(a_s, b_s, scorer=fuzz.token_set_ratio,
                               workers=1, dtype=np.float32)
    s_addr = rf_process.cpdist(a_a, b_a, scorer=fuzz.token_set_ratio,
                               workers=1, dtype=np.float32)
    hi = np.maximum(s_skel, s_addr)
    lo = np.minimum(s_skel, s_addr)
    return (hi + w_min * lo + w_block * (100.0 * block_score)).astype(np.float32)


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


def query_shard(lrows, lhash, lnorm, index: ShardIndex, k: int, chunk: int,
                split_channels: bool = False, w_min: float = 0.35):
    """Top-k rows of the shard for each left row, by cosine over IDF weights.

    With `split_channels`, the name-derived and address-derived keys are scored
    as two separate cosines, each normalised by its own channel's norm, and
    combined max-dominant. The corruption is disjunctive -- 99.78% of true pairs
    have address OR name similarity >=70, often not both -- so a single mixed
    vector penalises precisely the pairs where one field was destroyed.

    `lnorm` is then a 2-row array: [name norms, address norms].
    """
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
        ch_rep = np.repeat(index.chan[ck_keys], lens)
        offs = (np.arange(total, dtype=np.int64)
                - np.repeat(np.cumsum(np.r_[0, lens[:-1]]), lens))
        right_rep = index.postings[np.repeat(starts, lens) + offs]
        del offs

        comb = left_rep.astype(np.int64) * index.n_right + right_rep
        del left_rep, right_rep
        o = np.argsort(comb, kind="stable")
        comb, w_rep, ch_rep = comb[o], w_rep[o], ch_rep[o]
        del o
        bnd = np.flatnonzero(np.r_[True, comb[1:] != comb[:-1]])
        if split_channels:
            sum_n = np.add.reduceat(w_rep * (ch_rep == 0), bnd).astype(np.float32)
            sum_a = np.add.reduceat(w_rep * (ch_rep == 1), bnd).astype(np.float32)
        else:
            pair_score = np.add.reduceat(w_rep, bnd).astype(np.float32)
        pc = comb[bnd]
        del comb, w_rep, ch_rep, bnd

        pl = (pc // index.n_right).astype(np.int32)
        pr = (pc % index.n_right).astype(np.int32)
        del pc
        if split_channels:
            cos_n = sum_n / (lnorm[0][pl] * index.norm_n[pr])
            cos_a = sum_a / (lnorm[1][pl] * index.norm_a[pr])
            del sum_n, sum_a
            hi = np.maximum(cos_n, cos_a)
            lo_ = np.minimum(cos_n, cos_a)
            pair_score = (hi + w_min * lo_).astype(np.float32)
            del cos_n, cos_a, hi, lo_
        else:
            # lnorm is (2, n_left): per-channel norms. The mixed norm is their
            # quadrature sum, since norm^2 is just the total of idf^2 over all
            # of a row's keys. Indexing the 2-row array directly would broadcast
            # into an (n_pairs, n_left) array.
            ln2 = (np.sqrt(lnorm[0][pl] ** 2 + lnorm[1][pl] ** 2)
                   if lnorm.ndim == 2 else lnorm[pl])
            pair_score /= ln2 * index.norm[pr]
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
