"""Group an entity's candidates into clusters, then predict a whole cluster.

An entity's true matches are several independently corrupted copies of one
business, so they resemble *each other* far more than they resemble anything
else. Measured on 127,802 co-matched pairs, max(skeleton, address)
token_set_ratio averages 97.2 and clears 80 for 96.5% of them; over random
right-right pairs it averages 46.7 and clears 80 for 0.4%.

That separation is sharper than the Source1-to-candidate signal the pair model
scores (address 91.7, skeleton 93.7 on true pairs), which is why pairwise
scoring alone leaves recall on the table: we retrieve *some* match for 96.7% of
entities but fully cover only 75.9%. Clustering turns "found one" into
"found all" -- a weak individual link is carried by its cluster-mates.

Procedure, per entity:
  1. take the top candidates by model probability
  2. join any two whose mutual similarity clears `edge`
  3. score each connected component by its summed probability
  4. predict the whole winning component, including members whose own
     probability would not have survived a threshold
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process


def _components(n, edges):
    """Union-find over a handful of nodes."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return [find(i) for i in range(n)]


def decide(val: pd.DataFrame, attrs: pd.DataFrame, top_m: int = 12,
           edge: float = 85.0, tau_empty: float = 0.6, min_p: float = 0.05,
           carry: float = 0.15, prob_col: str = "prob") -> dict:
    """val: [source1_entity_id, cand_entity_id, prob]; attrs indexed by cand id.

    `carry` is the floor a cluster member must reach to be carried along by its
    cluster; it is far below the gate a lone candidate would have to clear.
    """
    if val.empty:
        return {}
    v = val[val[prob_col] >= min_p]
    if v.empty:
        return {}
    ent = v["source1_entity_id"].to_numpy()
    cand = v["cand_entity_id"].to_numpy()
    p = v[prob_col].to_numpy(np.float32)

    order = np.lexsort((-p, ent))
    ent, cand, p = ent[order], cand[order], p[order]
    starts = np.flatnonzero(np.r_[True, ent[1:] != ent[:-1]])
    ends = np.r_[starts[1:], len(ent)]

    skel = attrs["s_name"].to_dict()
    addr = attrs["n_addr"].to_dict()

    out: dict = {}
    for s, e in zip(starts.tolist(), ends.tolist()):
        ids = cand[s:e][:top_m]
        pr = p[s:e][:top_m]
        if pr[0] < tau_empty:
            continue                       # gate: this entity predicts nothing
        n = len(ids)
        if n == 1:
            out[ent[s]] = {ids[0]}
            continue
        sk = [skel.get(i, "") or "" for i in ids]
        ad = [addr.get(i, "") or "" for i in ids]
        ai, bi = np.triu_indices(n, 1)
        ss = rf_process.cpdist([sk[x] for x in ai], [sk[y] for y in bi],
                               scorer=fuzz.token_set_ratio, workers=1,
                               dtype=np.float32)
        sa = rf_process.cpdist([ad[x] for x in ai], [ad[y] for y in bi],
                               scorer=fuzz.token_set_ratio, workers=1,
                               dtype=np.float32)
        m = np.maximum(ss, sa)
        edges = [(int(ai[t]), int(bi[t])) for t in np.flatnonzero(m >= edge)]
        roots = _components(n, edges)

        score: dict = {}
        for i, r in enumerate(roots):
            score[r] = score.get(r, 0.0) + float(pr[i])
        best_root = max(score, key=score.get)
        sel = {ids[i] for i in range(n)
               if roots[i] == best_root and pr[i] >= carry}
        if sel:
            out[ent[s]] = sel
    return out
