"""Fine-tune a cross-encoder on labelled candidate pairs, then score with it.

Why a cross-encoder and not more features: the model sits at ~0.92 against a
~0.966 oracle, so it holds the right candidates and mis-ranks them. Every
existing feature is a string-similarity statistic over normalised text. A
cross-encoder reads both records jointly and can learn the corruption patterns
directly -- transliteration drift, token replacement, abbreviation semantics,
reordering -- which is genuinely different information.

Why it is affordable: scoring all ~118M test pairs would take hours, so it runs
only over the top candidates per entity (by stage-1 probability or blocking
score). At top-10 that is ~17M pairs, roughly an hour on an A100, and its output
becomes one more feature for the GBDT rather than a replacement for it.

Model: `intfloat/multilingual-e5-small` (118M, MIT) with a binary head --
comfortably inside the challenge's MIT/Apache and 8B limits. Only the
organisers' text is used; no identity is ever looked up externally.

  python -m src.crossenc fit   --cands work/union_train_comp.parquet --steps 6000
  python -m src.crossenc score --cands work/union_test_comp.parquet --split test --topm 10
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config
from .run_model import ATTR_COLS, enc_ids, load_left, load_right, read_ground_truth

MODEL = "intfloat/multilingual-e5-small"
MAXLEN = 96
CKPT = config.MODEL_DIR / "crossenc"


def _pair_texts(pairs, left, right):
    L = left.reindex(pairs["source1_entity_id"].to_numpy())
    R = right.reindex(pairs["cand_id"].to_numpy())
    a = (L["n_name"].fillna("") + " | " + L["n_addr"].fillna("")).to_numpy()
    b = (R["n_name"].fillna("") + " | " + R["n_addr"].fillna("")).to_numpy()
    return a, b


def fit(cands_path, steps=6000, batch=256, lr=2e-5, n_pos=800_000,
        n_neg=1_600_000, split="train"):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.time()
    pairs = pd.read_parquet(cands_path,
                            columns=["source1_entity_id", "cand_id", "sim_block"])
    ents = set(pairs["source1_entity_id"].unique())
    truth = read_ground_truth(ents)
    truth_enc = {s: (set(enc_ids(sorted(m)).tolist()) if m else set())
                 for s, m in truth.items()}
    sids = pairs["source1_entity_id"].to_numpy()
    cids = pairs["cand_id"].to_numpy()
    y = np.fromiter((c in truth_enc.get(s, ()) for s, c in zip(sids, cids)),
                    np.int8, len(pairs))
    print(f"[ce] {len(pairs):,} pairs, {int(y.sum()):,} positives "
          f"[{time.time() - t0:.0f}s]", flush=True)

    rng = np.random.default_rng(config.SEED)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) > n_pos:
        pos = rng.choice(pos, n_pos, replace=False)
    # hard negatives: the ones the blocking score already likes
    sb = pairs["sim_block"].to_numpy()
    order = np.argsort(-sb[neg])
    neg = neg[order[:n_neg]]
    sel = np.concatenate([pos, neg])
    rng.shuffle(sel)
    print(f"[ce] training on {len(sel):,} pairs "
          f"({len(pos):,} pos / {len(neg):,} hard neg)", flush=True)

    sub = pairs.iloc[sel]
    labels = y[sel].astype(np.float32)
    left = load_left(split)
    right = load_right(split, keep=sub["cand_id"].unique())
    a, b = _pair_texts(sub, left, right)
    del left, right, pairs, sub

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=1).cuda()
    model.gradient_checkpointing_disable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    lossf = torch.nn.BCEWithLogitsLoss()
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.06)

    n = len(a)
    model.train()
    step = 0
    run = 0.0
    while step < steps:
        idx = rng.integers(0, n, batch)
        enc = tok(list(a[idx]), list(b[idx]), padding=True, truncation=True,
                  max_length=MAXLEN, return_tensors="pt")
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        lab = torch.from_numpy(labels[idx]).cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**enc).logits.squeeze(-1)
            loss = lossf(out.float(), lab)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run += float(loss)
        step += 1
        if step % 200 == 0:
            print(f"[ce] step {step}/{steps} loss {run / 200:.4f} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            run = 0.0
    CKPT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(CKPT)
    tok.save_pretrained(CKPT)
    print(f"[ce] saved {CKPT} [{time.time() - t0:.0f}s]", flush=True)


def score(cands_path, split, topm=10, batch=512, prob_col=None):
    """Add `ce_score` for the top-M candidates per entity; NaN elsewhere."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    t0 = time.time()
    df = pd.read_parquet(cands_path)
    rank_col = prob_col if (prob_col and prob_col in df.columns) else "sim_block"
    codes, _ = pd.factorize(df["source1_entity_id"].to_numpy())
    v = df[rank_col].to_numpy(np.float32)
    o = np.lexsort((-v, codes))
    cs = codes[o]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    gs = np.repeat(starts, np.diff(np.r_[starts, len(cs)]))
    rank = np.empty(len(df), np.int64)
    rank[o] = np.arange(len(df)) - gs
    sel = np.flatnonzero(rank < topm)
    print(f"[ce] scoring {len(sel):,} of {len(df):,} pairs (top {topm})",
          flush=True)

    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16).cuda().eval()
    left = load_left(split)
    right = load_right(split, keep=df["cand_id"].to_numpy()[sel])
    a, b = _pair_texts(df.iloc[sel], left, right)
    del left, right

    out = np.full(len(df), np.nan, np.float32)
    buf = np.empty(len(sel), np.float32)
    for i in range(0, len(sel), batch):
        enc = tok(list(a[i:i + batch]), list(b[i:i + batch]), padding=True,
                  truncation=True, max_length=MAXLEN, return_tensors="pt")
        enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
        with torch.no_grad():
            lg = model(**enc).logits.squeeze(-1).float()
        buf[i:i + batch] = torch.sigmoid(lg).cpu().numpy()
        if (i // batch) % 400 == 0:
            el = time.time() - t0
            print(f"[ce] {i:,}/{len(sel):,} ({i / max(el, 1):.0f}/s) [{el:.0f}s]",
                  flush=True)
    out[sel] = buf
    df["ce_score"] = out
    df.to_parquet(cands_path, index=False)
    print(f"[ce] wrote ce_score to {cands_path} [{time.time() - t0:.0f}s]",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "score"])
    ap.add_argument("--cands", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--topm", type=int, default=10)
    a = ap.parse_args(argv)
    if a.mode == "fit":
        fit(a.cands, steps=a.steps, batch=a.batch, split=a.split)
    else:
        score(a.cands, a.split, topm=a.topm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
