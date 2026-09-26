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

from .tokidf import idf_stats

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
                   verbose=True, split: str = "train") -> pd.DataFrame:
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
    # competition features, if the candidate table carries them (see
    # competition.py). They encode what *else* is claiming this candidate,
    # which the one-to-one structure makes highly informative.
    for c in ("cand_n_claims", "cand_max_sim", "sim_rel_cand", "cand_is_best",
              "cand_rank", "cand_name_sim", "cand_name_max", "cand_name_rel",
              "cand_name_is_best", "cand_name_rank",
              # embedding cosine from a pretrained multilingual encoder;
              # information the string-similarity features cannot express
              "emb_cos",
              # retrieval provenance: rev_rank==0 means this entity is the
              # record's single best owner among all 2.2M Source 1 entities
              "rev_score", "rev_rank", "fwd_hit", "rev_hit", "both_hit"):
        if c in pairs.columns:
            f[c] = pairs[c].to_numpy(np.float32)

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
    # An empty address is MISSING data, not disagreement. token_set_ratio("x","")
    # returns 0, and because the address features carry the highest gain in the
    # model, a blank address was being read as "the addresses strongly differ".
    # Error analysis found true pairs with an exactly matching name and a blank
    # right-hand address scoring ~0.1. LightGBM handles NaN natively as missing,
    # so the trees can route these cases instead of penalising them.
    addr_missing = (la == "") | (ra == "")
    for tag in ("tsr", "tso", "rat"):
        v = _cpdist(la, ra, _SCORERS[tag])
        v[addr_missing] = np.nan
        f[f"addr_{tag}"] = v

    # ---- discrete agreement -------------------------------------------
    if verbose:
        print("    discrete features...", flush=True)
    # name rarity: an exact match on a generic name ("blue trading",
    # "new delhi india") is weak evidence, but the pair features treat every
    # matching token alike. The model's confident false merges were dominated by
    # exactly such names, so expose how rare the name actually is.
    st = idf_stats(ln, split)
    f["name_idf_min"] = st[0]
    f["name_idf_mean"] = st[1]
    f["name_idf_max"] = st[2]
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
    ai[addr_missing] = np.nan
    f["addr_tok_inter"] = ai
    f["addr_tok_jacc"] = np.where(addr_missing, np.nan, ai / au)

    def numset(s):
        # ANY token containing a digit, not just tokens starting with one.
        # Unit designators like "d1", "c2", "e50" are exactly what distinguishes
        # neighbouring businesses, and error analysis found confident false
        # merges that differed only there: "room no 6 404 d1" vs "404 d6"
        # scored 0.999 because neither token was being extracted.
        return {t for t in s.split() if any(c.isdigit() for c in t)} if s else set()

    num_i = np.empty(len(la), np.float32)
    num_u = np.empty(len(la), np.float32)
    num_c = np.empty(len(la), np.float32)
    for i in range(len(la)):
        sa, sb = numset(la[i]), numset(ra[i])
        num_i[i] = len(sa & sb)
        num_u[i] = len(sa | sb) or 1
        # tokens present on one side only: positive evidence AGAINST a match,
        # which an intersection-only view cannot express
        num_c[i] = len(sa ^ sb)
    num_i[addr_missing] = np.nan
    num_c[addr_missing] = np.nan
    f["addr_num_inter"] = num_i
    f["addr_num_jacc"] = np.where(addr_missing, np.nan, num_i / num_u)
    f["addr_num_conflict"] = num_c
    f["addr_num_conflict_rate"] = np.where(addr_missing, np.nan, num_c / num_u)

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

    # ---- transitivity: does this candidate agree with the entity's best? ----
    # Every true match of an entity is a corrupted copy of the *same* business,
    # so the true matches should resemble each other, while a distractor
    # typically resembles only the Source 1 record on one field. Comparing each
    # candidate against its entity's strongest candidate gives the model a view
    # of that agreement, which a pairwise-only feature set cannot express.
    if verbose:
        print("    transitivity features...", flush=True)
    anchor = np.zeros(n_ent, np.int64)
    best_v = np.full(n_ent, -np.inf, np.float32)
    sb = f["sim_block"].to_numpy(np.float32)
    np.maximum.at(best_v, codes, sb)
    is_best = sb >= best_v[codes] - 1e-9
    anchor[codes[is_best]] = np.flatnonzero(is_best)
    apos = anchor[codes]
    f["is_anchor"] = is_best.astype(np.int8)
    f["skel_vs_anchor"] = _cpdist(rs, rs[apos], _SCORERS["tsr"])
    f["addr_vs_anchor"] = _cpdist(ra, ra[apos], _SCORERS["tsr"])
    # agreement with the Source 1 record, relative to how well the anchor agrees
    f["anchor_gap_skel"] = f["skel_tsr"].to_numpy() - f["skel_tsr"].to_numpy()[apos]
    f["anchor_gap_addr"] = f["addr_tsr"].to_numpy() - f["addr_tsr"].to_numpy()[apos]

    # ---- cluster support among an entity's own candidates ------------------
    # An entity's true matches are several corrupted copies of one business, so
    # they resemble each other far more than anything else: over 127,802
    # co-matched pairs max(skeleton, address) similarity averages 97.2 and
    # clears 80 for 96.5% of them, versus 46.7 and 0.4% for random right-right
    # pairs. A candidate with several near-identical siblings in the same
    # entity's list is therefore very likely real.
    #
    # Applying this as a hard rule (predict the whole best cluster) measured
    # 0.884 -- it lifted link recall 0.807 -> 0.859 but cost precision
    # 0.978 -> 0.927, which F_0.5 punishes twice as hard. As features, the model
    # can decide when the evidence is worth acting on.
    if verbose:
        print("    cluster-support features...", flush=True)
    TOPM = 16
    rank_all = np.empty(len(codes), np.int64)
    o3 = np.lexsort((-f["sim_block"].to_numpy(), codes))
    st3 = np.flatnonzero(np.r_[True, codes[o3][1:] != codes[o3][:-1]])
    gs3 = np.repeat(st3, np.diff(np.r_[st3, len(o3)]))
    rank_all[o3] = np.arange(len(o3)) - gs3
    inm = rank_all < TOPM

    idx_in = np.flatnonzero(inm)
    ci = codes[idx_in]
    order_in = np.argsort(ci, kind="stable")
    idx_in = idx_in[order_in]
    ci = ci[order_in]
    gstart = np.flatnonzero(np.r_[True, ci[1:] != ci[:-1]])
    gend = np.r_[gstart[1:], len(ci)]

    ai_l, bi_l = [], []
    for s_, e_ in zip(gstart.tolist(), gend.tolist()):
        m = e_ - s_
        if m < 2:
            continue
        u, v = np.triu_indices(m, 1)
        ai_l.append(idx_in[s_ + u])
        bi_l.append(idx_in[s_ + v])
    deg = np.zeros(len(codes), np.float32)
    mass = np.zeros(len(codes), np.float32)
    if ai_l:
        ai = np.concatenate(ai_l); bi = np.concatenate(bi_l)
        sim_s = _cpdist(rs[ai], rs[bi], _SCORERS["tsr"])
        sim_a = _cpdist(ra[ai], ra[bi], _SCORERS["tsr"])
        mx = np.maximum(sim_s, sim_a)
        hit = mx >= 85.0
        sb_all = f["sim_block"].to_numpy(np.float32)
        np.add.at(deg, ai[hit], 1.0)
        np.add.at(deg, bi[hit], 1.0)
        np.add.at(mass, ai[hit], sb_all[bi[hit]])
        np.add.at(mass, bi[hit], sb_all[ai[hit]])
        del ai, bi, sim_s, sim_a, mx, hit
    f["clus_deg"] = deg
    f["clus_mass"] = mass

    cnt = np.bincount(codes, minlength=n_ent).astype(np.float32)
    f["n_cands"] = cnt[codes]
    f["clus_deg_rel"] = deg / np.maximum(np.minimum(cnt[codes], TOPM) - 1, 1)
    # rank of this candidate within its entity by the combined blocking score
    order = np.lexsort((-f["skel_tsr"].to_numpy(), codes))
    rank = np.empty(len(codes), np.float32)
    starts = np.flatnonzero(np.r_[True, codes[order][1:] != codes[order][:-1]])
    gs = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    rank[order] = np.arange(len(order)) - gs
    f["skel_rank"] = rank

    return f.astype(np.float32)


FEATURE_COLS = None  # set on first build by the caller
