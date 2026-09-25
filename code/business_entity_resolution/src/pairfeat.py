"""Pair features for the candidate set.

Every string comparison goes through `rapidfuzz.process.cpdist`, which compares
two equal-length sequences elementwise across threads. A Python loop over tens
of millions of pairs is the difference between minutes and hours.

Two deliberate omissions:

*No country feature.* The test set contains France, absent from training. A
model given `country` would learn US/India-specific thresholds and apply
something arbitrary to a quarter-million French entities. Every feature here is
computed the same way whatever the country label says.

*Rank-relative features included.* The metric is macro-averaged per entity, so
an entity with 2 candidates and one with 120 are worth the same. Features like
"how far below this entity's best candidate is this one" let the model express
a per-entity decision that an absolute similarity cannot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process

_SCORERS = {
    "tsr": fuzz.token_set_ratio,
    "tso": fuzz.token_sort_ratio,
    "rat": fuzz.ratio,
    "par": fuzz.partial_ratio,
}


def _cpdist(a, b, scorer, workers=-1):
    if len(a) == 0:
        return np.empty(0, np.float32)
    return rf_process.cpdist(a, b, scorer=scorer, workers=workers,
                             dtype=np.float32)


def _tokset(s):
    return set(s.split()) if s else set()


def build_features(pairs: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame,
                   verbose=True) -> pd.DataFrame:
    """pairs: [source1_entity_id, cand_id, sim_block]; left/right indexed by id.

    `left` must be indexed by entity_id (string), `right` by the int64 encoded
    id, both carrying n_name / s_name / n_addr / postal / name_acr.
    """
    L = left.reindex(pairs["source1_entity_id"].to_numpy())
    R = right.reindex(pairs["cand_id"].to_numpy())

    ln = L["n_name"].fillna("").to_numpy()
    rn = R["n_name"].fillna("").to_numpy()
    ls = L["s_name"].fillna("").to_numpy()
    rs = R["s_name"].fillna("").to_numpy()
    la = L["n_addr"].fillna("").to_numpy()
    ra = R["n_addr"].fillna("").to_numpy()
    lp = L["postal"].fillna("").to_numpy()
    rp = R["postal"].fillna("").to_numpy()
    lc = L["name_acr"].fillna("").to_numpy()
    rc = R["name_acr"].fillna("").to_numpy()

    f = pd.DataFrame(index=pairs.index)
    f["sim_block"] = pairs["sim_block"].to_numpy(np.float32)

    if verbose:
        print("    name similarities...", flush=True)
    for tag, sc in _SCORERS.items():
        f[f"name_{tag}"] = _cpdist(ln, rn, sc)
    if verbose:
        print("    skeleton similarities...", flush=True)
    for tag in ("tsr", "rat"):
        f[f"skel_{tag}"] = _cpdist(ls, rs, _SCORERS[tag])
    if verbose:
        print("    address similarities...", flush=True)
    for tag in ("tsr", "tso", "rat"):
        f[f"addr_{tag}"] = _cpdist(la, ra, _SCORERS[tag])

    # ---- discrete agreement -------------------------------------------
    if verbose:
        print("    discrete features...", flush=True)
    f["postal_eq"] = ((lp == rp) & (lp != "")).astype(np.int8)
    f["postal_both"] = ((lp != "") & (rp != "")).astype(np.int8)
    f["acr_eq"] = ((lc == rc) & (lc != "")).astype(np.int8)
    f["addr_empty"] = ((la == "") | (ra == "")).astype(np.int8)
    f["cand_src"] = (pairs["cand_id"].to_numpy() >> 32).astype(np.int8)

    # token overlap on name and address, and shared numeric tokens
    def overlap(a_arr, b_arr):
        inter = np.empty(len(a_arr), np.float32)
        union = np.empty(len(a_arr), np.float32)
        for i in range(len(a_arr)):
            sa, sb = _tokset(a_arr[i]), _tokset(b_arr[i])
            inter[i] = len(sa & sb)
            union[i] = len(sa | sb) or 1
        return inter, union

    ni, nu = overlap(ln, rn)
    f["name_tok_inter"] = ni
    f["name_tok_jacc"] = ni / nu
    ai, au = overlap(la, ra)
    f["addr_tok_inter"] = ai
    f["addr_tok_jacc"] = ai / au

    def numset(s):
        return {t for t in s.split() if t[:1].isdigit()} if s else set()

    num_i = np.empty(len(la), np.float32)
    num_u = np.empty(len(la), np.float32)
    for i in range(len(la)):
        sa, sb = numset(la[i]), numset(ra[i])
        num_i[i] = len(sa & sb)
        num_u[i] = len(sa | sb) or 1
    f["addr_num_inter"] = num_i
    f["addr_num_jacc"] = num_i / num_u

    # ---- length / shape ------------------------------------------------
    f["len_name_l"] = np.fromiter((len(x) for x in ln), np.float32, len(ln))
    f["len_name_r"] = np.fromiter((len(x) for x in rn), np.float32, len(rn))
    f["len_ratio"] = f["len_name_r"] / np.maximum(f["len_name_l"], 1)
    f["ntok_l"] = np.fromiter((x.count(" ") + 1 if x else 0 for x in ln),
                              np.float32, len(ln))
    f["ntok_r"] = np.fromiter((x.count(" ") + 1 if x else 0 for x in rn),
                              np.float32, len(rn))

    # ---- rank-relative, within each source1 entity ---------------------
    if verbose:
        print("    rank-relative features...", flush=True)
    ent = pairs["source1_entity_id"].to_numpy()
    codes, _ = pd.factorize(ent)
    n_ent = codes.max() + 1 if len(codes) else 0

    def add_rel(col):
        v = f[col].to_numpy(np.float32)
        best = np.zeros(n_ent, np.float32)
        np.maximum.at(best, codes, v)
        b = best[codes]
        f[f"{col}_rel"] = v / np.maximum(b, 1e-6)
        f[f"{col}_gap"] = b - v

    for col in ("sim_block", "name_tsr", "skel_tsr", "addr_tsr"):
        add_rel(col)

    cnt = np.bincount(codes, minlength=n_ent).astype(np.float32)
    f["n_cands"] = cnt[codes]
    # rank of this candidate within its entity by the combined blocking score
    order = np.lexsort((-f["skel_tsr"].to_numpy(), codes))
    rank = np.empty(len(codes), np.float32)
    starts = np.flatnonzero(np.r_[True, codes[order][1:] != codes[order][:-1]])
    gs = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank[order] = np.arange(len(order)) - gs
    f["skel_rank"] = rank

    return f.astype(np.float32)


FEATURE_COLS = None  # set on first build by the caller
