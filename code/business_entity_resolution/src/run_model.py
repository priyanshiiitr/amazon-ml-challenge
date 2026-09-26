"""Train the pair model, tune the decision layer, write the submission.

  python -m src.run_model train
  python -m src.run_model predict

Both stages stream the candidate set in chunks: at test scale there are tens of
millions of pairs and featurising them all at once is the one place this
pipeline can still run out of memory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .decision import apply_decision, tune_decision
from .evaluate import detailed_report, print_report
from .pairfeat import build_features

ATTR_COLS = ["entity_id", "n_name", "s_name", "n_addr", "postal", "name_acr"]
MODEL_PATH = config.MODEL_DIR / "lgb.txt"
KNOBS_PATH = config.MODEL_DIR / "decision.json"
FEATS_PATH = config.MODEL_DIR / "features.json"
CHUNK_PAIRS = 4_000_000


def enc_ids(ids) -> np.ndarray:
    a = np.asarray(ids, dtype=object)
    src = np.fromiter((int(s[1]) for s in a), np.int64, len(a))
    num = np.fromiter((int(s[3:]) for s in a), np.int64, len(a))
    return (src << 32) | num


def dec_ids(v) -> list:
    return [f"S{int(x) >> 32}-{int(x) & 0xFFFFFFFF}" for x in v]


def load_left(split):
    df = pq.read_table(config.WORK_DIR / f"{split}_source1.parquet",
                       columns=ATTR_COLS).to_pandas()
    return df.set_index("entity_id")


def load_right(split, keep: np.ndarray | None = None):
    """Right-hand attributes indexed by encoded int64 id."""
    frames = []
    for nm in ("source2", "source3"):
        t = pq.read_table(config.WORK_DIR / f"{split}_{nm}.parquet",
                          columns=ATTR_COLS).to_pandas()
        t["cand_id"] = enc_ids(t["entity_id"].to_numpy())
        t = t.drop(columns=["entity_id"])
        if keep is not None:
            t = t[np.isin(t["cand_id"].to_numpy(), keep)]
        frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    return out.set_index("cand_id")


def read_ground_truth(ents: set | None = None):
    truth = {}
    with open(config.PATHS.train_ground_truth, encoding="utf-8",
              errors="replace") as fh:
        fh.readline()
        for line in fh:
            sid, _, rest = line.rstrip("\n").partition("\t")
            if ents is not None and sid not in ents:
                continue
            truth[sid] = {m.strip() for m in rest.split(",") if m.strip()}
    return truth


# ---------------------------------------------------------------------------
def cmd_train(a):
    import lightgbm as lgb

    t0 = time.time()
    pairs = pd.read_parquet(a.cands)
    print(f"[train] {len(pairs):,} candidate pairs", flush=True)

    ents = set(pairs["source1_entity_id"].unique())
    truth = read_ground_truth(ents)
    print(f"[train] {len(truth):,} entities with ground truth", flush=True)

    left = load_left("train")
    right = load_right("train", keep=pairs["cand_id"].unique())
    print(f"[train] attrs loaded [{time.time() - t0:.0f}s]", flush=True)

    feats = build_features(pairs, left, right, split="train")
    del left, right
    print(f"[train] features {feats.shape} [{time.time() - t0:.0f}s]", flush=True)

    # labels
    truth_enc = {sid: (enc_ids(sorted(ms)) if ms else np.empty(0, np.int64))
                 for sid, ms in truth.items()}
    y = np.zeros(len(pairs), np.int8)
    sids = pairs["source1_entity_id"].to_numpy()
    cids = pairs["cand_id"].to_numpy()
    for i in range(len(pairs)):
        t = truth_enc.get(sids[i])
        if t is not None and t.size and cids[i] in t:
            y[i] = 1
    print(f"[train] positive rate {y.mean():.4f}", flush=True)

    # split by ENTITY, never by pair
    uniq = np.array(sorted(ents))
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(uniq)
    n_val = int(len(uniq) * config.VALID_FRACTION)
    val_ents = set(uniq[:n_val].tolist())
    is_val = np.fromiter((s in val_ents for s in sids), bool, len(sids))
    print(f"[train] train pairs {int((~is_val).sum()):,}  "
          f"val pairs {int(is_val.sum()):,}", flush=True)

    cols = list(feats.columns)
    dtr = lgb.Dataset(feats[~is_val], label=y[~is_val], feature_name=cols)
    dva = lgb.Dataset(feats[is_val], label=y[is_val], feature_name=cols,
                      reference=dtr)
    booster = lgb.train(config.LGB_PARAMS, dtr, num_boost_round=config.LGB_ROUNDS,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(config.LGB_EARLY_STOP,
                                                      verbose=False),
                                   lgb.log_evaluation(100)])
    booster.save_model(str(MODEL_PATH))
    json.dump(cols, open(FEATS_PATH, "w"))
    print(f"[train] best_iter={booster.best_iteration} "
          f"[{time.time() - t0:.0f}s]", flush=True)

    imp = sorted(zip(cols, booster.feature_importance("gain")),
                 key=lambda x: -x[1])[:15]
    print("[train] top features by gain:")
    for nm, g in imp:
        print(f"    {nm:22s} {g:12.0f}")

    # ---- decision layer on the held-out entities ----------------------
    val = pd.DataFrame({
        "source1_entity_id": sids[is_val],
        "cand_entity_id": cids[is_val],
        "prob": booster.predict(feats[is_val],
                                num_iteration=booster.best_iteration),
    })
    true_map = {sid: truth_enc[sid] for sid in val_ents if sid in truth_enc}
    true_map = {k: set(v.tolist()) for k, v in true_map.items()}
    knobs = tune_decision(val, true_map)
    print(f"[train] best knobs: {knobs}", flush=True)
    json.dump({"global": knobs}, open(KNOBS_PATH, "w"), indent=2)

    pred = apply_decision(val, knobs["tau_empty"], knobs["tau_abs"],
                          knobs["alpha"], knobs["one_to_one"])
    print_report(detailed_report(pred, true_map), "tuned decision")
    print(f"[train] total {time.time() - t0:.0f}s", flush=True)
    return 0


# ---------------------------------------------------------------------------
def cmd_predict(a):
    import lightgbm as lgb

    t0 = time.time()
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    cols = json.load(open(FEATS_PATH))
    knobs = json.load(open(KNOBS_PATH))["global"]
    print(f"[predict] knobs {knobs}", flush=True)

    pf = pq.ParquetFile(a.cands)
    left = load_left("test")
    right = load_right("test")
    print(f"[predict] attrs loaded [{time.time() - t0:.0f}s]", flush=True)

    # Entities become int32 codes for the rest of this function. At test scale
    # there are ~200M candidate pairs, and carrying the id as a Python string
    # costs over 12 GB before any grouping happens.
    all_s1 = pq.read_table(config.WORK_DIR / "test_source1.parquet",
                           columns=["entity_id"]).to_pandas()["entity_id"]
    code_of = {e: i for i, e in enumerate(all_s1)}
    n_ent = len(all_s1)
    best = np.zeros(n_ent, np.float32)

    # Candidates scoring below this can never survive the tuned thresholds, so
    # they are dropped as they stream past instead of being materialised.
    floor = min(knobs["tau_abs"], knobs["tau_empty"]) * 0.5
    keep_c, keep_r, keep_p = [], [], []
    cand_c, cand_r = [], []
    done = 0
    for batch in pf.iter_batches(batch_size=CHUNK_PAIRS):
        pairs = batch.to_pandas()
        feats = build_features(pairs, left, right, verbose=False, split="test")
        prob = booster.predict(feats[cols],
                               num_iteration=booster.best_iteration
                               ).astype(np.float32)
        codes = np.fromiter((code_of[s] for s in pairs["source1_entity_id"]),
                            np.int32, len(pairs))
        cids = pairs["cand_id"].to_numpy()
        np.maximum.at(best, codes, prob)
        cand_c.append(codes)
        cand_r.append(cids)
        m = prob >= floor
        keep_c.append(codes[m]); keep_r.append(cids[m]); keep_p.append(prob[m])
        done += len(pairs)
        print(f"[predict] {done:,} pairs, {int(m.sum()):,} above floor "
              f"[{time.time() - t0:.0f}s]", flush=True)
        del feats, pairs, prob, codes, cids, m
    del left, right

    codes = np.concatenate(keep_c); cids = np.concatenate(keep_r)
    probs = np.concatenate(keep_p)
    del keep_c, keep_r, keep_p
    print(f"[predict] {len(codes):,} pairs survived the floor", flush=True)

    # ---- decision layer, in numpy ------------------------------------
    hyb_path = config.MODEL_DIR / "hybrid.json"
    hyb = json.load(open(hyb_path)) if hyb_path.exists() else None
    if hyb:
        # Gate by the tuned tau_empty, then size each entity's set by expected
        # F_0.5. Measured on 60,000 held-out entities, the analytic rule wins on
        # non-singletons (0.90911 vs 0.90213) and the gate wins on singletons
        # (0.86841 vs 0.74315); combining them beat either alone.
        from .expected_f import decide as ef_decide
        tmp = pd.DataFrame({"source1_entity_id": codes,
                            "cand_entity_id": cids, "prob": probs})
        sel = ef_decide(tmp, gamma=hyb["gamma"], tau_empty=hyb["tau_empty"])
        del tmp
        keep_c, keep_r = [], []
        for c, s in sel.items():
            keep_c.append(np.full(len(s), c, np.int32))
            keep_r.append(np.fromiter(s, np.int64, len(s)))
        if keep_c:
            codes = np.concatenate(keep_c)
            cids = np.concatenate(keep_r)
            probs = np.ones(len(codes), np.float32)
        else:
            codes = np.empty(0, np.int32); cids = np.empty(0, np.int64)
            probs = np.empty(0, np.float32)
        print(f"[predict] hybrid rule: gamma={hyb['gamma']} "
              f"tau_empty={hyb['tau_empty']}", flush=True)
    else:
        b = best[codes]
        thr = np.maximum(np.float32(knobs["tau_abs"]),
                         np.float32(knobs["alpha"]) * b)
        keep = (probs >= thr) & (b >= np.float32(knobs["tau_empty"]))
        codes, cids, probs = codes[keep], cids[keep], probs[keep]
    if knobs.get("one_to_one"):
        # each S2/S3 record belongs to at most one S1 entity (verified on the
        # training ground truth: zero of 7,638,365 ids were reused), so keep
        # only the highest-scoring claim on each candidate
        order = np.argsort(-probs, kind="stable")
        codes, cids = codes[order], cids[order]
        _, first = np.unique(cids, return_index=True)
        codes, cids = codes[first], cids[first]
    print(f"[predict] {len(codes):,} predicted links "
          f"[{time.time() - t0:.0f}s]", flush=True)

    def grouped(code_arr, id_arr):
        """Sort by entity code and return (ids_sorted, group_bounds).

        A dict of lists would box ~200M candidate ids into Python ints; sorted
        arrays plus searchsorted bounds do the same job at a tenth the memory.
        """
        o = np.argsort(code_arr, kind="stable")
        cs, ids = code_arr[o], id_arr[o]
        return ids, np.searchsorted(cs, np.arange(n_ent + 1))

    pred_ids, pred_bounds = grouped(codes, cids)
    all_codes = np.concatenate(cand_c)
    all_cids = np.concatenate(cand_r)
    del cand_c, cand_r
    cand_ids, cand_bounds = grouped(all_codes, all_cids)
    del all_codes, all_cids
    print(f"[predict] grouped [{time.time() - t0:.0f}s]", flush=True)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mpath = config.OUTPUT_DIR / "matching_results.tsv"
    cpath = config.OUTPUT_DIR / "candidate_pairs.tsv"
    with open(mpath, "w", encoding="utf-8", newline="") as mf, \
         open(cpath, "w", encoding="utf-8", newline="") as cf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, sid in enumerate(all_s1):
            m = pred_ids[pred_bounds[i]:pred_bounds[i + 1]]
            mf.write(sid + "\t"
                     + (",".join(sorted(set(dec_ids(m)))) if len(m) else "")
                     + "\n")
            c = cand_ids[cand_bounds[i]:cand_bounds[i + 1]]
            cf.write(sid + "\t"
                     + (",".join(sorted(set(dec_ids(c)))) if len(c) else "")
                     + "\n")
    print(f"[predict] wrote {mpath} and {cpath} [{time.time() - t0:.0f}s]",
          flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--cands", required=True)
    a = ap.parse_args(argv)
    return cmd_train(a) if a.cmd == "train" else cmd_predict(a)


if __name__ == "__main__":
    sys.exit(main())
