"""End-to-end driver.

  python -m src.run_pipeline inspect          # scale + ground-truth statistics
  python -m src.run_pipeline blocking         # candidates on train, recall ceiling
  python -m src.run_pipeline train            # features + LightGBM + tuned decision
  python -m src.run_pipeline predict          # test candidates -> output/*.tsv
  python -m src.run_pipeline all

Add --sample N to run everything on the first N source1 entities. Do this first:
at full scale a single pass is long, and almost every bug shows up on 20k rows.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

from . import config
from .blocking import generate_candidates, recall_ceiling
from .data_io import load_split, read_ground_truth
from .decision import apply_decision, apply_per_country, tune_decision, tune_per_country
from .evaluate import detailed_report, print_report
from .features import build_idf, compute_features
from .model import label_candidates, load as load_model, predict as model_predict
from .model import save as save_model, split_entities, train as train_model
from .submission import run_official_validator, write_submission

CAND_TRAIN = config.WORK_DIR / "cands_train.parquet"
FEAT_TRAIN = config.WORK_DIR / "feats_train.parquet"
MODEL_PATH = config.MODEL_DIR / "lgb.txt"
KNOBS_PATH = config.MODEL_DIR / "decision.json"
IDF_PATH = config.MODEL_DIR / "idf.pkl"


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        print(f"\n>>> {self.label}")
        return self

    def __exit__(self, *a):
        print(f"<<< {self.label}: {time.time() - self.t:.1f}s")


def _subsample(s1, s2, s3, gt, n):
    """Keep N source1 entities plus every source2/3 record they can match, and a
    similar volume of unrelated records so blocking still faces distractors."""
    s1 = s1.head(n).reset_index(drop=True)
    keep_ids = set()
    if gt is not None:
        gt = gt[gt["source1_entity_id"].isin(set(s1["entity_id"]))].reset_index(drop=True)
        for ms in gt["matches"]:
            keep_ids |= ms
    def cut(df):
        related = df[df["entity_id"].isin(keep_ids)]
        others = df[~df["entity_id"].isin(keep_ids)].head(max(n * 3, 20_000))
        return pd.concat([related, others], ignore_index=True).drop_duplicates("entity_id")
    return s1, cut(s2), cut(s3), gt


def prune_candidates(cands: pd.DataFrame, top_n: int = 12,
                     keep_above: float = 0.80) -> pd.DataFrame:
    """Second blocking stage: the set the model actually scores.

    This is what goes into candidate_pairs.tsv, per the spec ("the final
    candidate list just before the ML model scores them").
    """
    if cands.empty:
        return cands
    score = (0.5 * cands["sim_name"] + 0.2 * cands["sim_blob"]
             + 0.15 * cands["sim_addr"]
             + 0.15 * cands[["key_name_sorted", "key_acr", "key_postal_tok"]].max(axis=1))
    cands = cands.assign(_pre=score.astype(np.float32))
    rank = cands.groupby("source1_entity_id")["_pre"].rank(ascending=False, method="first")
    out = cands[(rank <= top_n) | (cands["_pre"] >= keep_above)]
    print(f"[prune] {len(cands):,} -> {len(out):,} candidate pairs "
          f"({len(out) / max(cands['source1_entity_id'].nunique(), 1):.1f} per entity)")
    return out.drop(columns=["_pre"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
def cmd_inspect(args):
    from .inspect_data import main as inspect_main
    return inspect_main()


def cmd_blocking(args):
    with Timer("load train"):
        s1, s2, s3 = load_split("train")
        gt = read_ground_truth(config.PATHS.train_ground_truth)
        if args.sample:
            s1, s2, s3, gt = _subsample(s1, s2, s3, gt, args.sample)
        print(f"    s1={len(s1):,} s2={len(s2):,} s3={len(s3):,} gt={len(gt):,}")

    with Timer("candidate generation"):
        cands = generate_candidates(s1, s2, s3)
        cands = prune_candidates(cands, args.top_n)

    with Timer("recall ceiling"):
        rc = recall_ceiling(cands, gt)
        print(f"    link recall           : {rc['link_recall']:.4f}  "
              f"({rc['recovered']:,} / {rc['true_links']:,})")
        print(f"    entities fully covered: {rc['entities_fully_covered']:.4f}")
        print(f"    reduction ratio       : "
              f"{1 - len(cands) / (len(s1) * (len(s2) + len(s3))):.10f}")
        if rc["link_recall"] < 0.95:
            print("    [warn] recall ceiling below 0.95 — raise BLOCK_TOPK_FORWARD "
                  "or top_n before investing in the model")

    cands.to_parquet(CAND_TRAIN, index=False)
    print(f"    saved {CAND_TRAIN}")
    return 0


def cmd_train(args):
    with Timer("load train"):
        s1, s2, s3 = load_split("train")
        gt = read_ground_truth(config.PATHS.train_ground_truth)
        if args.sample:
            s1, s2, s3, gt = _subsample(s1, s2, s3, gt, args.sample)
        right = pd.concat([s2.assign(cand_src=2), s3.assign(cand_src=3)],
                          ignore_index=True)

    if CAND_TRAIN.exists() and not args.refresh:
        cands = pd.read_parquet(CAND_TRAIN)
        print(f"    reusing {CAND_TRAIN} ({len(cands):,} pairs)")
    else:
        with Timer("candidate generation"):
            cands = prune_candidates(generate_candidates(s1, s2, s3), args.top_n)
            cands.to_parquet(CAND_TRAIN, index=False)
        print(f"    recall ceiling: {recall_ceiling(cands, gt)}")

    with Timer("features"):
        idf_name = build_idf(pd.concat([s1["n_name"], right["n_name"]]))
        idf_addr = build_idf(pd.concat([s1["n_addr"], right["n_addr"]]))
        with open(IDF_PATH, "wb") as fh:
            pickle.dump({"name": idf_name, "addr": idf_addr}, fh)
        feats = compute_features(cands, s1, right, idf_name, idf_addr)
        feats["y"] = label_candidates(feats, gt)
        print(f"    {len(feats):,} pairs, positive rate "
              f"{feats['y'].mean():.4f}")

    with Timer("train / valid split by entity"):
        tr_ids, va_ids = split_entities(s1)
        tr = feats[feats["source1_entity_id"].isin(tr_ids)]
        va = feats[feats["source1_entity_id"].isin(va_ids)]
        print(f"    train pairs={len(tr):,}  valid pairs={len(va):,}")

    with Timer("LightGBM"):
        booster = train_model(tr, va)
        save_model(booster, MODEL_PATH)

    with Timer("decision tuning"):
        va = va.copy()
        va["prob"] = model_predict(booster, va)
        # every held-out entity is scored, including ones blocking missed entirely
        gt_idx = dict(zip(gt["source1_entity_id"], gt["matches"]))
        true_map = {sid: gt_idx.get(sid, set())
                    for sid in s1["entity_id"] if sid in va_ids}
        knobs = tune_decision(va, true_map)
        print(f"    best global: {knobs}")

        country_of = dict(zip(s1["entity_id"], s1["country_key"]))
        per_country = tune_per_country(va, true_map, country_of)

        pred_global = apply_decision(va, knobs["tau_empty"], knobs["tau_abs"],
                                     knobs["alpha"], knobs["one_to_one"])
        rep_g = detailed_report(pred_global, true_map)
        print_report(rep_g, "global knobs")

        if per_country:
            pred_pc = apply_per_country(va, per_country, knobs, country_of)
            rep_pc = detailed_report(pred_pc, true_map)
            print_report(rep_pc, "per-country knobs")
            use_pc = rep_pc["macro_f0.5"] > rep_g["macro_f0.5"] + 1e-4
        else:
            use_pc = False
        print(f"\n    -> using {'per-country' if use_pc else 'global'} knobs")

    with open(KNOBS_PATH, "w") as fh:
        json.dump({"global": knobs, "per_country": per_country,
                   "use_per_country": bool(use_pc)}, fh, indent=2)
    print(f"    saved {MODEL_PATH} and {KNOBS_PATH}")
    return 0


def cmd_predict(args):
    with Timer("load test"):
        s1, s2, s3 = load_split("test")
        if args.sample:
            s1, s2, s3, _ = _subsample(s1, s2, s3, None, args.sample)
        right = pd.concat([s2.assign(cand_src=2), s3.assign(cand_src=3)],
                          ignore_index=True)
        print(f"    s1={len(s1):,} s2={len(s2):,} s3={len(s3):,}")
        print(f"    countries: {sorted(set(s1['country']))}")

    with Timer("candidate generation"):
        cands = prune_candidates(generate_candidates(s1, s2, s3), args.top_n)

    with Timer("features + scoring"):
        with open(IDF_PATH, "rb") as fh:
            idfs = pickle.load(fh)
        feats = compute_features(cands, s1, right, idfs["name"], idfs["addr"])
        booster = load_model(MODEL_PATH)
        feats["prob"] = model_predict(booster, feats)

    with Timer("decision + write"):
        knobs = json.loads(KNOBS_PATH.read_text())
        country_of = dict(zip(s1["entity_id"], s1["country_key"]))
        if knobs.get("use_per_country") and knobs.get("per_country"):
            pred_map = apply_per_country(feats, knobs["per_country"],
                                         knobs["global"], country_of)
        else:
            g = knobs["global"]
            pred_map = apply_decision(feats, g["tau_empty"], g["tau_abs"],
                                      g["alpha"], g["one_to_one"])
        cand_map = {k: set(v) for k, v in
                    feats.groupby("source1_entity_id")["cand_entity_id"]
                    .apply(list).items()}
        mp, cp = write_submission(s1, s2, s3, pred_map, cand_map)
        if not args.sample:
            run_official_validator(mp, cp)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inspect", "blocking", "train", "predict", "all"])
    ap.add_argument("--sample", type=int, default=0,
                    help="run on the first N source1 entities (smoke test)")
    ap.add_argument("--top-n", type=int, default=12,
                    help="candidates per entity after pruning")
    ap.add_argument("--refresh", action="store_true",
                    help="recompute cached candidates")
    args = ap.parse_args(argv)

    np.random.seed(config.SEED)
    if args.stage == "all":
        for fn in (cmd_blocking, cmd_train, cmd_predict):
            rc = fn(args)
            if rc:
                return rc
        return 0
    return {"inspect": cmd_inspect, "blocking": cmd_blocking,
            "train": cmd_train, "predict": cmd_predict}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
