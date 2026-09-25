"""Stream the raw TSVs into normalised parquet shards.

Normalisation is pure Python and runs over ~11.7M records, so it is worth doing
exactly once and caching. Reading is chunked and each chunk is written out
before the next is read, so peak memory is one chunk regardless of file size --
the whole pipeline has to fit in 8 GB.

Only the columns blocking actually needs are kept. The full derived set
(`blob`, `name_sorted`, `name_acr`, ...) costs about 1.9 GB at test scale and is
cheaper to recompute on the few million surviving candidate pairs than to carry
through retrieval.

  python -m src.prep train
  python -m src.prep test
"""
from __future__ import annotations

import sys
import time

import pandas as pd

from . import config
from .normalize import acronym, addr_postal, basic_clean, norm_addr, norm_name, skel

CHUNK = 250_000
KEEP = ["entity_id", "country_key", "n_name", "s_name", "n_addr", "postal", "name_acr"]


def prep_file(path, out_path, verbose=True):
    t0 = time.time()
    n = 0
    writer = None
    import pyarrow as pa
    import pyarrow.parquet as pq

    reader = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
        quoting=3, on_bad_lines="warn", engine="c", chunksize=CHUNK,
    )
    for chunk in reader:
        chunk.columns = [c.strip() for c in chunk.columns]
        for c in ("entity_id", "business_name", "business_address", "country"):
            chunk[c] = chunk[c].fillna("").astype(str).str.strip()

        out = pd.DataFrame({
            "entity_id": chunk["entity_id"],
            "country_key": chunk["country"].str.lower(),
        })
        out["n_name"] = [norm_name(s) for s in chunk["business_name"]]
        out["s_name"] = [skel(s) for s in out["n_name"]]
        out["n_addr"] = [norm_addr(s) for s in chunk["business_address"]]
        out["postal"] = [addr_postal(s) for s in out["n_addr"]]
        out["name_acr"] = [acronym(s) for s in out["n_name"]]

        table = pa.Table.from_pandas(out[KEEP], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        n += len(out)
        if verbose:
            print(f"    {n:,} rows  [{time.time() - t0:.0f}s]", flush=True)
    if writer is not None:
        writer.close()
    if verbose:
        print(f"  -> {out_path} ({n:,} rows, {time.time() - t0:.0f}s)", flush=True)
    return n


def main(split: str):
    names = ("source1", "source2", "source3")
    for nm in names:
        src = getattr(config.PATHS, f"{split}_{nm}")
        dst = config.WORK_DIR / f"{split}_{nm}.parquet"
        if dst.exists():
            print(f"[prep] {dst.name} exists, skipping")
            continue
        print(f"[prep] {src.name} -> {dst.name}")
        prep_file(src, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "train"))
