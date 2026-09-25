"""Candidate generation — an IDF-weighted inverted index over discrete keys.

This stage sets the recall ceiling for the whole pipeline: a true match that
never becomes a candidate can never be recovered downstream.

Why not char n-gram TF-IDF + sparse matmul (the obvious choice, and what this
file used to do): at full scale it is 1.7M x 10M per country. That approach
materialises `B.T` for a ~5M x 400k sparse matrix, runs three such passes, and
accumulates every hit into pandas frames that then have to be merged with
`how="outer"` across 150M+ rows. It runs out of memory long before it runs out
of time.

Instead each record emits a handful of discrete *keys*; a candidate is any pair
sharing at least one key, scored by the summed IDF of the keys they share. That
is the same cosine-like ranking, but computed only over pairs that actually
share something, so the cost is proportional to the number of real collisions
rather than to the product of the two sides.

Keys per record, in rough order of selectivity:

  skel_full     whole consonant-skeleton name        mean bucket 1.33
  skel_bigram   adjacent skeleton-token pairs        very selective
  skel_token    individual skeleton tokens           the broad net
  postal_tok    postal code + first skeleton token   survives name drift
  num_tok       street number + first skeleton token for missing postal codes

Every key is a consonant skeleton rather than raw text, because ~23% of Indian
Source 2/3 names are in an Indic script while Source 1 is always Latin, and the
skeleton is what makes those two comparable (see normalize.skel).

Keys whose document frequency exceeds DF_CAP are dropped: they are stopwords,
they contribute almost no information, and they dominate the cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# key extraction
# ---------------------------------------------------------------------------
_GENERIC_ADDR = {
    "street", "road", "avenue", "lane", "drive", "court", "place", "square",
    "boulevard", "circle", "terrace", "trail", "parkway", "highway", "floor",
    "suite", "apartment", "unit", "building", "room", "number", "near",
    "opposite", "behind", "block", "phase", "sector", "nagar", "colony",
    "north", "south", "east", "west", "new", "old", "post", "office", "rue",
}


def _shingles(s: str, n: int = 4) -> list[str]:
    """Character n-grams of a whitespace-stripped skeleton.

    Whole-token equality is too rigid: 'Tech Constructions' and its Devanagari
    counterpart skeletonise to *similar* but not identical token sets, so an
    exact-token index misses them. Shingles degrade gracefully instead.
    """
    t = s.replace(" ", "")
    if len(t) < n:
        return [t] if t else []
    return [t[i:i + n] for i in range(len(t) - n + 1)]


def _keys_for(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Emit (row_index, key_string) for every key of every row.

    Returned as two parallel arrays so the caller can hash the keys once.

    Crucially this emits *address-only* keys as well as name keys. Measured on
    6,000 entities, 68.6% of the links no key could reach were cases where the
    right-hand name had been replaced by an unrelated token ('NYLADREX',
    '#sjace', 'dprobst.com') while the address stayed nearly identical. Every
    key anchored on the name is blind to those by construction.
    """
    rows: list[np.ndarray] = []
    keys: list[np.ndarray] = []

    s_name = df["s_name"].to_numpy()
    postal = df["postal"].to_numpy()
    n_addr = df["n_addr"].to_numpy()

    buf: dict[str, tuple[list, list]] = {}

    def emit(tag, i, k):
        r, kk = buf.setdefault(tag, ([], []))
        r.append(i)
        kk.append(tag + "|" + k)

    for i in range(len(s_name)):
        sn = s_name[i]
        toks = sn.split() if sn else []

        # ---- name keys -------------------------------------------------
        if toks:
            if len(toks) <= 8:
                emit("F", i, "".join(sorted(toks)))
            seen = set()
            for t in toks:
                if len(t) >= 2 and t not in seen:
                    seen.add(t)
                    emit("T", i, t)
            for a, b in zip(toks, toks[1:]):
                emit("B", i, a + b)
            # shingles rescue near-miss skeletons that exact tokens split apart
            for g in set(_shingles(sn, 4)):
                emit("G", i, g)

        # ---- address keys, independent of the name ---------------------
        ad = n_addr[i]
        if not ad:
            continue
        atoks = ad.split()
        nums = [t for t in atoks if t[0].isdigit()]
        words = [t for t in atoks
                 if not t[0].isdigit() and len(t) >= 3 and t not in _GENERIC_ADDR]
        p = postal[i]

        # postal code paired with a distinctive street word
        if p:
            for w in words[:6]:
                emit("PA", i, p + "|" + w)
        # street number paired with a distinctive street word: survives a
        # missing postal code, reordering, and '31' vs '31D'
        for num in nums[:3]:
            base = num.rstrip("abcdefghijklmnopqrstuvwxyz")
            if not base:
                continue
            for w in words[:5]:
                emit("NA", i, base + "|" + w)
        # adjacent distinctive address words, for addresses with no number
        if not nums:
            for a, b in zip(words, words[1:]):
                emit("AB", i, a + "|" + b)

    for r, k in buf.values():
        rows.append(np.asarray(r, dtype=np.int32))
        keys.append(np.asarray(k, dtype=object))

    if not rows:
        return np.empty(0, np.int32), np.empty(0, dtype=object)
    return np.concatenate(rows), np.concatenate(keys)


# ---------------------------------------------------------------------------
# inverted index over the right-hand side
# ---------------------------------------------------------------------------
class KeyIndex:
    """Postings lists keyed by key-id, with IDF weights and a df cap."""

    def __init__(self, right: pd.DataFrame, df_cap: int, verbose=True):
        rows, keys = _keys_for(right)
        if len(rows) == 0:
            self.empty = True
            return
        self.empty = False

        uniq, key_ids = np.unique(keys, return_inverse=True)
        key_ids = key_ids.astype(np.int32)
        df = np.bincount(key_ids, minlength=len(uniq)).astype(np.int32)

        # each channel gets its own frequency cap; the tag is the key's first
        # character, so this costs one pass over the vocabulary
        tags = np.frombuffer("".join(k[0] for k in uniq.tolist()).encode("ascii"),
                             dtype="S1")
        caps = np.full(len(uniq), df_cap, dtype=np.int32)
        for tag, cap in config.DF_CAP_BY_TAG.items():
            caps[tags == tag.encode("ascii")] = cap

        keep = df <= caps
        mask = keep[key_ids]
        rows, key_ids = rows[mask], key_ids[mask]

        order = np.argsort(key_ids, kind="stable")
        self.postings = rows[order]
        sorted_keys = key_ids[order]
        counts = np.bincount(sorted_keys, minlength=len(uniq))
        self.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        n_docs = max(len(right), 1)
        with np.errstate(divide="ignore"):
            self.idf = np.log(1.0 + n_docs / np.maximum(df, 1)).astype(np.float32)
        self.vocab = {k: i for i, k in enumerate(uniq.tolist())}
        self.n_right = len(right)

        # L2 norm of each right row in IDF space, so the score below is a
        # cosine rather than a raw sum. Without this, records with many keys
        # (long names, long addresses) outrank the true match on every query.
        w2 = self.idf[sorted_keys] ** 2
        self.norm = np.sqrt(
            np.bincount(self.postings, weights=w2, minlength=len(right))
        ).astype(np.float32)
        np.maximum(self.norm, 1e-6, out=self.norm)

        if verbose:
            print(f"[index] {len(uniq):,} keys, {len(self.postings):,} postings "
                  f"({len(df) - int(keep.sum()):,} keys dropped over df_cap={df_cap})")

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """Key strings -> key ids, -1 where unknown."""
        v = self.vocab
        return np.fromiter((v.get(k, -1) for k in keys), dtype=np.int32,
                           count=len(keys))


def _topk_per_row(left_row, right_idx, score, k):
    """Keep the k highest-scoring right_idx for each left_row.

    Input need not be sorted. Returns the three arrays filtered in place.
    """
    if len(left_row) == 0:
        return left_row, right_idx, score
    # sort by (left_row asc, score desc) so each group's best come first
    order = np.lexsort((-score, left_row))
    left_row, right_idx, score = left_row[order], right_idx[order], score[order]
    # rank within each contiguous group; int32 throughout, since these arrays
    # can reach tens of millions of rows per chunk
    starts = np.flatnonzero(np.r_[True, left_row[1:] != left_row[:-1]]).astype(np.int32)
    group_start = np.repeat(starts, np.diff(np.r_[starts, np.int32(len(left_row))]))
    rank = np.arange(len(left_row), dtype=np.int32) - group_start
    keep = rank < k
    return left_row[keep], right_idx[keep], score[keep]


def _retrieve(left: pd.DataFrame, index: KeyIndex, k: int, chunk: int,
              verbose=True):
    """Top-k right rows per left row by summed IDF of shared keys."""
    rows, keys = _keys_for(left)
    if len(rows) == 0 or index.empty:
        return (np.empty(0, np.int32),) * 2 + (np.empty(0, np.float32),)

    key_ids = index.lookup(keys)
    ok = key_ids >= 0
    rows, key_ids = rows[ok], key_ids[ok]

    order = np.argsort(rows, kind="stable")
    rows, key_ids = rows[order], key_ids[order]
    row_starts = np.searchsorted(rows, np.arange(0, len(left) + 1))

    left_norm = np.sqrt(
        np.bincount(rows, weights=index.idf[key_ids] ** 2, minlength=len(left))
    ).astype(np.float32)
    np.maximum(left_norm, 1e-6, out=left_norm)

    out_l, out_r, out_s = [], [], []
    n_left = len(left)
    for lo in range(0, n_left, chunk):
        hi = min(lo + chunk, n_left)
        a, b = row_starts[lo], row_starts[hi]
        if a == b:
            continue
        ck_rows = rows[a:b]
        ck_keys = key_ids[a:b]

        starts = index.indptr[ck_keys]
        ends = index.indptr[ck_keys + 1]
        lens = (ends - starts).astype(np.int64)
        total = int(lens.sum())
        if total == 0:
            continue

        # expand each (left row, key) into its postings list
        left_rep = np.repeat(ck_rows, lens)
        w_rep = np.repeat(index.idf[ck_keys], lens)
        # build the concatenated posting indices without a Python loop
        pos = np.repeat(starts, lens) + (
            np.arange(total, dtype=np.int64)
            - np.repeat(np.cumsum(np.r_[0, lens[:-1]]), lens)
        )
        right_rep = index.postings[pos]

        # sum the weights for each (left, right) pair
        comb = left_rep.astype(np.int64) * index.n_right + right_rep
        o = np.argsort(comb, kind="stable")
        comb, w_rep = comb[o], w_rep[o]
        bnd = np.flatnonzero(np.r_[True, comb[1:] != comb[:-1]])
        pair_score = np.add.reduceat(w_rep, bnd).astype(np.float32)
        pair_comb = comb[bnd]

        pl = (pair_comb // index.n_right).astype(np.int32)
        pr = (pair_comb % index.n_right).astype(np.int32)
        pair_score /= left_norm[pl] * index.norm[pr]
        pl, pr, ps = _topk_per_row(pl, pr, pair_score, k)
        out_l.append(pl)
        out_r.append(pr)
        out_s.append(ps)

        if verbose and (lo // chunk) % 20 == 0:
            print(f"[retrieve] {hi:,}/{n_left:,} left rows, "
                  f"{sum(len(x) for x in out_l):,} pairs so far", flush=True)

    if not out_l:
        return (np.empty(0, np.int32),) * 2 + (np.empty(0, np.float32),)
    return (np.concatenate(out_l), np.concatenate(out_r),
            np.concatenate(out_s).astype(np.float32))


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
SIM_COLS = ["sim_block", "n_keys"]


def generate_candidates(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                        top_k: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Candidate pairs as [source1_entity_id, cand_entity_id, cand_src, sim_block]."""
    top_k = top_k or config.BLOCK_TOPK_FORWARD
    right = pd.concat([s2.assign(cand_src=np.int8(2)),
                       s3.assign(cand_src=np.int8(3))], ignore_index=True)

    groups = (sorted(set(s1["country_key"]) | set(right["country_key"]))
              if config.BLOCK_WITHIN_COUNTRY else [None])

    frames = []
    for g in groups:
        L = s1 if g is None else s1[s1["country_key"] == g]
        R = right if g is None else right[right["country_key"] == g]
        if len(L) == 0:
            continue
        if len(R) == 0:
            if verbose:
                print(f"[blocking] country={g!r}: {len(L):,} source1 rows but no "
                      f"source2/3 rows — forced singletons")
            continue
        L = L.reset_index(drop=True)
        R = R.reset_index(drop=True)
        if verbose:
            print(f"[blocking] country={g!r}: {len(L):,} x {len(R):,}")

        index = KeyIndex(R, config.DF_CAP, verbose=verbose)
        li, ri, sc = _retrieve(L, index, top_k, config.BLOCK_CHUNK, verbose=verbose)
        if len(li) == 0:
            continue
        frames.append(pd.DataFrame({
            "source1_entity_id": L["entity_id"].to_numpy()[li],
            "cand_entity_id": R["entity_id"].to_numpy()[ri],
            "cand_src": R["cand_src"].to_numpy()[ri],
            "sim_block": sc,
        }))
        del index

    if not frames:
        return pd.DataFrame(columns=["source1_entity_id", "cand_entity_id",
                                     "cand_src", "sim_block"])
    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"[blocking] {len(out):,} candidate pairs "
              f"({len(out) / max(len(s1), 1):.1f} per source1 entity)")
    return out


def recall_ceiling(cands: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """What fraction of true links survived blocking? This caps final recall."""
    have = set(zip(cands["source1_entity_id"].tolist(),
                   cands["cand_entity_id"].tolist()))
    total = hit = per_entity_full = n_nonsingleton = 0
    covered_entities = 0
    for sid, ms in zip(gt["source1_entity_id"], gt["matches"]):
        if not ms:
            continue
        n_nonsingleton += 1
        h = sum(1 for m in ms if (sid, m) in have)
        total += len(ms)
        hit += h
        if h == len(ms):
            per_entity_full += 1
        if h:
            covered_entities += 1
    return {
        "link_recall": hit / max(total, 1),
        "entities_fully_covered": per_entity_full / max(n_nonsingleton, 1),
        "entities_partly_covered": covered_entities / max(n_nonsingleton, 1),
        "true_links": total,
        "recovered": hit,
    }
