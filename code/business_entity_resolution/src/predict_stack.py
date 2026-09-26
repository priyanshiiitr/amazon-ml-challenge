"""Two-pass stacked prediction over the test candidate set.

Sibling features need every candidate of an entity together, but the candidate
table is far too large to featurise in one piece, and chunk boundaries cut
across entities. So:

  pass 1  chunked: base features -> stage-1 probability, kept as one float32
          array over all pairs (~470 MB at test scale) and then discarded
  global: sibling support computed from those probabilities
  pass 2  chunked: base features rebuilt, joined to the sibling block,
          scored by stage 2, decided, written

Base features are built twice rather than cached because caching them would
cost ~28 GB on disk and more in memory; rebuilding is the cheaper trade.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config, gbm
from .pairfeat import build_features
from .run_model import CHUNK_PAIRS, dec_ids, load_left, load_right
from .stack import SIB_COLS, sibling_features


def main(argv=None):
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()

    s1 = gbm.load(config.MODEL_DIR / f"stage1.{gbm.model_suffix()}")
    cols1 = json.load(open(config.MODEL_DIR / "features_stage1.json"))
    s2 = gbm.load(config.MODEL_DIR / f"model.{gbm.model_suffix()}")
    cols2 = json.load(open(config.MODEL_DIR / "features.json"))
    hyb = json.load(open(config.MODEL_DIR / "hybrid.json"))
    print(f"[stack-predict] hybrid {hyb}", flush=True)

    left = load_left("test")
    right = load_right("test")
    print(f"[stack-predict] attrs loaded [{time.time() - t0:.0f}s]", flush=True)

    # ---- pass 1: stage-1 probabilities ---------------------------------
    pf = pq.ParquetFile(a.cands)
    p1_parts, done = [], 0
    for batch in pf.iter_batches(batch_size=CHUNK_PAIRS):
        pairs = batch.to_pandas()
        f = build_features(pairs, left, right, verbose=False, split="test")
        p1_parts.append(gbm.predict(s1, f[cols1], cols1))
        done += len(pairs)
        print(f"[stack-predict] pass1 {done:,} [{time.time() - t0:.0f}s]",
              flush=True)
        del f, pairs
    p1 = np.concatenate(p1_parts)
    del p1_parts
    print(f"[stack-predict] stage-1 done, {len(p1):,} probs "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # ---- global: sibling features --------------------------------------
    key = pq.read_table(a.cands, columns=["source1_entity_id",
                                          "cand_id"]).to_pandas()
    sib = sibling_features(key, right, p1)
    del p1, key
    print(f"[stack-predict] sibling block {sib.shape} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # ---- pass 2: stage-2 scoring ---------------------------------------
    all_s1 = pq.read_table(config.WORK_DIR / "test_source1.parquet",
                           columns=["entity_id"]).to_pandas()["entity_id"]
    code_of = {e: i for i, e in enumerate(all_s1)}
    n_ent = len(all_s1)
    floor = min(hyb["tau_empty"], 0.5) * 0.4
    keep_c, keep_r, keep_p, cand_c, cand_r = [], [], [], [], []
    off, done = 0, 0
    pf = pq.ParquetFile(a.cands)
    for batch in pf.iter_batches(batch_size=CHUNK_PAIRS):
        pairs = batch.to_pandas()
        n = len(pairs)
        f = build_features(pairs, left, right, verbose=False, split="test")
        blk = sib.iloc[off:off + n].reset_index(drop=True)
        f = pd.concat([f.reset_index(drop=True), blk], axis=1)
        prob = gbm.predict(s2, f[cols2], cols2)
        codes = np.fromiter((code_of[s] for s in pairs["source1_entity_id"]),
                            np.int32, n)
        cids = pairs["cand_id"].to_numpy()
        cand_c.append(codes); cand_r.append(cids)
        m = prob >= floor
        keep_c.append(codes[m]); keep_r.append(cids[m]); keep_p.append(prob[m])
        off += n; done += n
        print(f"[stack-predict] pass2 {done:,}, {int(m.sum()):,} above floor "
              f"[{time.time() - t0:.0f}s]", flush=True)
        del f, pairs, prob, codes, cids, m, blk
    del left, right, sib

    codes = np.concatenate(keep_c); cids = np.concatenate(keep_r)
    probs = np.concatenate(keep_p)
    del keep_c, keep_r, keep_p

    from .expected_f import decide as ef_decide
    tmp = pd.DataFrame({"source1_entity_id": codes, "cand_entity_id": cids,
                        "prob": probs})
    sel = ef_decide(tmp, gamma=hyb["gamma"], tau_empty=hyb["tau_empty"])
    del tmp, codes, cids, probs
    print(f"[stack-predict] {len(sel):,} entities with matches "
          f"[{time.time() - t0:.0f}s]", flush=True)

    all_codes = np.concatenate(cand_c); all_cids = np.concatenate(cand_r)
    del cand_c, cand_r
    o = np.argsort(all_codes, kind="stable")
    cs, cid_sorted = all_codes[o], all_cids[o]
    bounds = np.searchsorted(cs, np.arange(n_ent + 1))
    del all_codes, all_cids, o, cs

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mpath = config.OUTPUT_DIR / "matching_results.tsv"
    cpath = config.OUTPUT_DIR / "candidate_pairs.tsv"
    with open(mpath, "w", encoding="utf-8", newline="") as mf, \
         open(cpath, "w", encoding="utf-8", newline="") as cf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, sid in enumerate(all_s1):
            m = sel.get(i)
            mf.write(sid + "\t"
                     + (",".join(sorted(set(dec_ids(list(m))))) if m else "")
                     + "\n")
            c = cid_sorted[bounds[i]:bounds[i + 1]]
            cf.write(sid + "\t"
                     + (",".join(sorted(set(dec_ids(c)))) if len(c) else "")
                     + "\n")
    print(f"[stack-predict] wrote {mpath} [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
