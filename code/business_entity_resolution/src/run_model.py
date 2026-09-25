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

    feats = build_features(pairs, left, right)
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

    parts = []
    done = 0
    for batch in pf.iter_batches(batch_size=CHUNK_PAIRS):
        pairs = batch.to_pandas()
        feats = build_features(pairs, left, right, verbose=False)
        prob = booster.predict(feats[cols],
                               num_iteration=booster.best_iteration)
        parts.append(pd.DataFrame({
            "source1_entity_id": pairs["source1_entity_id"].to_numpy(),
            "cand_entity_id": pairs["cand_id"].to_numpy(),
            "prob": prob.astype(np.float32),
        }))
        done += len(pairs)
        print(f"[predict] {done:,} pairs [{time.time() - t0:.0f}s]", flush=True)
        del feats, pairs
    del left, right

    scored = pd.concat(parts, ignore_index=True)
    del parts
    pred = apply_decision(scored, knobs["tau_empty"], knobs["tau_abs"],
                          knobs["alpha"], knobs["one_to_one"])
    print(f"[predict] {len(pred):,} entities with >=1 match "
          f"[{time.time() - t0:.0f}s]", flush=True)

    # ---- write both files, one row per test source1 entity ------------
    all_s1 = pq.read_table(config.WORK_DIR / "test_source1.parquet",
                           columns=["entity_id"]).to_pandas()["entity_id"]
    cand_map = (scored.groupby("source1_entity_id")["cand_entity_id"]
                .apply(list).to_dict())
    del scored

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mpath = config.OUTPUT_DIR / "matching_results.tsv"
    cpath = config.OUTPUT_DIR / "candidate_pairs.tsv"
    with open(mpath, "w", encoding="utf-8", newline="") as mf, \
         open(cpath, "w", encoding="utf-8", newline="") as cf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1:
            m = pred.get(sid)
            mf.write(sid + "\t" + (",".join(sorted(set(dec_ids(m)))) if m else "") + "\n")
            c = cand_map.get(sid)
            cf.write(sid + "\t" + (",".join(sorted(set(dec_ids(c)))) if c else "") + "\n")
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
