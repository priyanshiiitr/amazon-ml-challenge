"""Pairwise features for candidate (source1, source2/3) pairs.

Two ideas drive the feature set beyond plain string similarity:

* **Rank-relative features.** F_0.5 is macro-averaged *per source1 entity*, so the
  question is never "is this pair similar in absolute terms" but "is this pair
  good *relative to the other candidates for this same entity*". Ratio-to-best
  and rank features let the model express that directly.
* **IDF-weighted token overlap.** Sharing the token "medical" is worth far less
  than sharing "kanhaiya". Plain Jaccard cannot see the difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    _HAS_RF = True
except ImportError:  # pragma: no cover
    _HAS_RF = False

try:
    from rapidfuzz.process import cpdist as _cpdist
    _HAS_CPDIST = True
except ImportError:
    _HAS_CPDIST = False

from .blocking import SIM_COLS


def _pairwise(a: list[str], b: list[str], scorer, workers: int = -1) -> np.ndarray:
    """Element-wise similarity over two equal-length lists, 0..1."""
    if not a:
        return np.zeros(0, np.float32)
    if _HAS_CPDIST:
        try:
            return (_cpdist(a, b, scorer=scorer, workers=workers)
                    .astype(np.float32) / 100.0)
        except Exception:  # noqa: BLE001  (older signature / unsupported scorer)
            pass
    return np.fromiter((scorer(x, y) for x, y in zip(a, b)),
                       dtype=np.float32, count=len(a)) / 100.0


def _jw(a: list[str], b: list[str]) -> np.ndarray:
    if not a:
        return np.zeros(0, np.float32)
    if _HAS_CPDIST:
        try:
            return _cpdist(a, b, scorer=JaroWinkler.similarity,
                           workers=-1).astype(np.float32)
        except Exception:  # noqa: BLE001
            pass
    return np.fromiter((JaroWinkler.similarity(x, y) for x, y in zip(a, b)),
                       dtype=np.float32, count=len(a))


# ---------------------------------------------------------------------------
# token-set features, computed without materialising python sets per pair where
# possible
# ---------------------------------------------------------------------------
def _token_stats(left_tokens, right_tokens, idf: dict, default_idf: float):
    """Jaccard, containment and IDF-weighted overlap for aligned token lists."""
    n = len(left_tokens)
    jac = np.zeros(n, np.float32)
    cont = np.zeros(n, np.float32)
    widf = np.zeros(n, np.float32)
    nshared = np.zeros(n, np.float32)
    max_idf_shared = np.zeros(n, np.float32)

    for i in range(n):
        A = left_tokens[i]
        B = right_tokens[i]
        if not A or not B:
            continue
        inter = A & B
        union = A | B
        jac[i] = len(inter) / len(union)
        cont[i] = len(inter) / min(len(A), len(B))
        nshared[i] = len(inter)
        if inter:
            w = [idf.get(t, default_idf) for t in inter]
            tot = sum(idf.get(t, default_idf) for t in union)
            widf[i] = sum(w) / tot if tot else 0.0
            max_idf_shared[i] = max(w)
    return jac, cont, widf, nshared, max_idf_shared


def build_idf(series: pd.Series) -> tuple[dict, float]:
    """Document frequency over a token field -> idf lookup."""
    from collections import Counter

    df = Counter()
    n_docs = 0
    for s in series:
        n_docs += 1
        if s:
            df.update(set(s.split()))
    idf = {t: float(np.log(n_docs / (1 + c)) + 1.0) for t, c in df.items()}
    default = float(np.log(n_docs / 1.0) + 1.0)
    return idf, default


NUMERIC_FEATURES = [
    # blocking similarities carried through
    *SIM_COLS,
    # name
    "nm_ratio", "nm_tokset", "nm_toksort", "nm_partial", "nm_jw",
    "nm_jaccard", "nm_contain", "nm_widf", "nm_nshared", "nm_maxidf",
    "nm_len_l", "nm_len_r", "nm_len_ratio", "nm_first_tok_eq",
    "nm_acr_eq", "nm_exact", "nm_sorted_eq", "nm_prefix4",
    # address
    "ad_ratio", "ad_tokset", "ad_toksort", "ad_jw",
    "ad_jaccard", "ad_contain", "ad_widf", "ad_nshared",
    "ad_num_inter", "ad_num_jacc", "ad_postal_eq", "ad_postal_both",
    "ad_len_l", "ad_len_r", "ad_empty_l", "ad_empty_r",
    # combined / context
    "cand_src", "country_eq",
    "n_cands", "rank_name", "rank_combined",
    "ratio_best_name", "ratio_best_combined", "gap_to_best",
    "combined", "combined_minus_mean", "best_minus_second",
]


def compute_features(cands: pd.DataFrame, s1: pd.DataFrame,
                     right: pd.DataFrame, idf_name=None, idf_addr=None,
                     chunk: int = 2_000_000, verbose: bool = True) -> pd.DataFrame:
    """Attach NUMERIC_FEATURES to the candidate frame. Returns a new frame."""
    if not _HAS_RF:
        raise ImportError("rapidfuzz is required: pip install rapidfuzz")

    l_idx = pd.Index(s1["entity_id"])
    r_idx = pd.Index(right["entity_id"])
    li = l_idx.get_indexer(cands["source1_entity_id"])
    ri = r_idx.get_indexer(cands["cand_entity_id"])
    if (li < 0).any() or (ri < 0).any():
        raise ValueError("candidate frame references unknown entity ids")

    if idf_name is None:
        idf_name = build_idf(s1["n_name"])
    if idf_addr is None:
        idf_addr = build_idf(s1["n_addr"])
    idf_n, def_n = idf_name
    idf_a, def_a = idf_addr

    L = {c: s1[c].to_numpy() for c in
         ("n_name", "n_addr", "c_name", "name_sorted", "name_acr", "postal",
          "country_key")}
    R = {c: right[c].to_numpy() for c in
         ("n_name", "n_addr", "c_name", "name_sorted", "name_acr", "postal",
          "country_key")}
    L_ntok = [set(x.split()) for x in L["n_name"]]
    R_ntok = [set(x.split()) for x in R["n_name"]]
    L_atok = [set(x.split()) for x in L["n_addr"]]
    R_atok = [set(x.split()) for x in R["n_addr"]]
    from .normalize import addr_numbers
    L_num = [addr_numbers(x) for x in L["n_addr"]]
    R_num = [addr_numbers(x) for x in R["n_addr"]]

    out_blocks = []
    n = len(cands)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        if verbose and n > chunk:
            print(f"[features] {start:,} .. {stop:,} of {n:,}")
        a = li[start:stop]
        b = ri[start:stop]
        f = {}

        ln = list(L["n_name"][a]); rn = list(R["n_name"][b])
        la = list(L["n_addr"][a]); ra = list(R["n_addr"][b])

        f["nm_ratio"] = _pairwise(ln, rn, fuzz.ratio)
        f["nm_tokset"] = _pairwise(ln, rn, fuzz.token_set_ratio)
        f["nm_toksort"] = _pairwise(ln, rn, fuzz.token_sort_ratio)
        f["nm_partial"] = _pairwise(ln, rn, fuzz.partial_ratio)
        f["nm_jw"] = _jw(ln, rn)

        jac, cont, widf, nsh, mxi = _token_stats(
            [L_ntok[i] for i in a], [R_ntok[i] for i in b], idf_n, def_n)
        f["nm_jaccard"], f["nm_contain"] = jac, cont
        f["nm_widf"], f["nm_nshared"], f["nm_maxidf"] = widf, nsh, mxi

        lnl = np.array([len(x) for x in ln], np.float32)
        rnl = np.array([len(x) for x in rn], np.float32)
        f["nm_len_l"], f["nm_len_r"] = lnl, rnl
        f["nm_len_ratio"] = np.minimum(lnl, rnl) / np.maximum(np.maximum(lnl, rnl), 1)
        f["nm_first_tok_eq"] = np.array(
            [1.0 if (x.split()[:1] == y.split()[:1] and x) else 0.0
             for x, y in zip(ln, rn)], np.float32)
        f["nm_acr_eq"] = ((L["name_acr"][a] == R["name_acr"][b]) &
                          (L["name_acr"][a] != "")).astype(np.float32)
        f["nm_exact"] = ((L["n_name"][a] == R["n_name"][b]) &
                         (L["n_name"][a] != "")).astype(np.float32)
        f["nm_sorted_eq"] = ((L["name_sorted"][a] == R["name_sorted"][b]) &
                             (L["name_sorted"][a] != "")).astype(np.float32)
        f["nm_prefix4"] = np.array(
            [1.0 if (x[:4] and x[:4] == y[:4]) else 0.0 for x, y in zip(ln, rn)],
            np.float32)

        f["ad_ratio"] = _pairwise(la, ra, fuzz.ratio)
        f["ad_tokset"] = _pairwise(la, ra, fuzz.token_set_ratio)
        f["ad_toksort"] = _pairwise(la, ra, fuzz.token_sort_ratio)
        f["ad_jw"] = _jw(la, ra)

        jac, cont, widf, nsh, _ = _token_stats(
            [L_atok[i] for i in a], [R_atok[i] for i in b], idf_a, def_a)
        f["ad_jaccard"], f["ad_contain"] = jac, cont
        f["ad_widf"], f["ad_nshared"] = widf, nsh

        ni = np.zeros(stop - start, np.float32)
        nj = np.zeros(stop - start, np.float32)
        for k, (i, j) in enumerate(zip(a, b)):
            A, B = L_num[i], R_num[j]
            if A and B:
                inter = len(A & B)
                ni[k] = inter
                nj[k] = inter / len(A | B)
        f["ad_num_inter"], f["ad_num_jacc"] = ni, nj
        pl, pr = L["postal"][a], R["postal"][b]
        f["ad_postal_eq"] = ((pl == pr) & (pl != "")).astype(np.float32)
        f["ad_postal_both"] = ((pl != "") & (pr != "")).astype(np.float32)
        f["ad_len_l"] = np.array([len(x) for x in la], np.float32)
        f["ad_len_r"] = np.array([len(x) for x in ra], np.float32)
        f["ad_empty_l"] = (f["ad_len_l"] == 0).astype(np.float32)
        f["ad_empty_r"] = (f["ad_len_r"] == 0).astype(np.float32)

        f["country_eq"] = (L["country_key"][a] == R["country_key"][b]).astype(np.float32)

        out_blocks.append(pd.DataFrame(f))

    feats = pd.concat(out_blocks, ignore_index=True) if out_blocks else pd.DataFrame()
    del out_blocks

    res = cands.reset_index(drop=True).copy()
    for c in feats.columns:
        res[c] = feats[c].to_numpy()
    del feats
    res["cand_src"] = res["cand_src"].astype(np.float32)

    # ---- per-entity context features ------------------------------------
    res["combined"] = (
        0.40 * res["nm_tokset"] + 0.20 * res["nm_jw"] + 0.15 * res["nm_widf"]
        + 0.15 * res["ad_tokset"] + 0.10 * res["ad_widf"]
    ).astype(np.float32)

    g = res.groupby("source1_entity_id", sort=False)
    res["n_cands"] = g["combined"].transform("size").astype(np.float32)
    best_nm = g["nm_tokset"].transform("max")
    best_cb = g["combined"].transform("max")
    mean_cb = g["combined"].transform("mean")
    res["ratio_best_name"] = (res["nm_tokset"] / best_nm.clip(lower=1e-6)).astype(np.float32)
    res["ratio_best_combined"] = (res["combined"] / best_cb.clip(lower=1e-6)).astype(np.float32)
    res["gap_to_best"] = (best_cb - res["combined"]).astype(np.float32)
    res["combined_minus_mean"] = (res["combined"] - mean_cb).astype(np.float32)
    res["rank_name"] = g["nm_tokset"].rank(ascending=False, method="first").astype(np.float32)
    res["rank_combined"] = g["combined"].rank(ascending=False, method="first").astype(np.float32)

    # Second-best combined score per entity. rank_combined is already computed
    # above, so this is a masked max rather than a python-level transform
    # (which costs minutes once there are millions of groups).
    res["best_minus_second"] = (
        best_cb - res["combined"].where(res["rank_combined"] == 2, 0.0)
        .groupby(res["source1_entity_id"]).transform("max")
    ).astype(np.float32)

    for c in NUMERIC_FEATURES:
        if c not in res.columns:
            res[c] = np.float32(0.0)
        res[c] = res[c].astype(np.float32)
    return res
