"""Encode every record with a small multilingual model, on GPU.

Motivation: the model sits at 0.921 against a 0.966 oracle — it has the right
candidates and mis-ranks them. Every hand-crafted feature is a string-similarity
statistic; an embedding cosine is genuinely different information, covering
transliteration drift, abbreviation semantics and token reordering that edit
distance does not capture.

Timing is why this runs at all: blocking and feature building are pure CPU, so
the GPU is otherwise idle for hours. Encoding concurrently costs no wall-clock.

Model: `intfloat/multilingual-e5-small` — 118M parameters, MIT licence, well
inside the challenge's MIT/Apache and 8B limits. It is a *pretrained encoder*,
not an external data source: no business identity is ever looked up, and the
text encoded is only what the organisers provided.

Vectors are written float16 in parquet row order, so row i of the .npy matches
row i of the corresponding parquet; lookup is positional, no join needed.

  python -m src.embed test source2 --batch 1024
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pyarrow.parquet as pq

from . import config

MODEL = "intfloat/multilingual-e5-small"
DIM = 384
MAXLEN = 64


def _texts(split, source, lo, hi):
    """'name | address' for a row range, in parquet order."""
    t = pq.read_table(config.WORK_DIR / f"{split}_{source}.parquet",
                      columns=["n_name", "n_addr"]).to_pandas()
    nm = t["n_name"].fillna("").to_numpy()
    ad = t["n_addr"].fillna("").to_numpy()
    del t
    out = [f"query: {a} | {b}" for a, b in zip(nm[lo:hi], ad[lo:hi])]
    return out, len(nm)


def encode(split: str, source: str, batch: int = 1024, threads: int = 4):
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)          # leave CPU for the blocking jobs
    dst = config.WORK_DIR / f"emb_{split}_{source}.npy"
    if dst.exists():
        print(f"[embed] {dst.name} exists, skipping", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()

    pf = pq.ParquetFile(config.WORK_DIR / f"{split}_{source}.parquet")
    n = pf.metadata.num_rows
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float16,
                                    shape=(n, DIM))
    t0 = time.time()
    done = 0
    for chunk in pf.iter_batches(batch_size=200_000,
                                 columns=["n_name", "n_addr"]):
        d = chunk.to_pandas()
        texts = ["query: " + (a or "") + " | " + (b or "")
                 for a, b in zip(d["n_name"], d["n_addr"])]
        del d, chunk
        for i in range(0, len(texts), batch):
            part = texts[i:i + batch]
            enc = tok(part, padding=True, truncation=True, max_length=MAXLEN,
                      return_tensors="pt")
            enc = {k: v.cuda(non_blocking=True) for k, v in enc.items()}
            with torch.no_grad():
                h = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                v = torch.nn.functional.normalize(v, dim=-1)
            out[done:done + len(part)] = v.cpu().numpy().astype(np.float16)
            done += len(part)
        el = time.time() - t0
        print(f"[embed] {split}/{source} {done:,}/{n:,} "
              f"({done / max(el, 1):.0f}/s) [{el:.0f}s]", flush=True)
    out.flush()
    print(f"[embed] wrote {dst} ({n:,} x {DIM}) in {time.time() - t0:.0f}s",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("source", choices=["source1", "source2", "source3", "all"])
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    srcs = (["source1", "source2", "source3"] if a.source == "all"
            else [a.source])
    for s in srcs:
        encode(a.split, s, a.batch, a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
