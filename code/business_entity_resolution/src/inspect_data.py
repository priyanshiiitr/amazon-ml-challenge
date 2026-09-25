"""Step 0: find the data, print scale and the statistics that drive design.

Run:  python -m src.inspect_data
"""
from __future__ import annotations

import sys
from collections import Counter

import pandas as pd

from . import config
from .data_io import read_source, read_ground_truth


def hr(t=""):
    print("\n" + "=" * 72)
    if t:
        print(t)
        print("=" * 72)


def main():
    hr("RESOLVED PATHS")
    print(f"project root : {config.PROJECT_ROOT}")
    for key in ("train_source1", "train_source2", "train_source3",
                "train_ground_truth", "test_source1", "test_source2",
                "test_source3", "validator"):
        try:
            p = getattr(config.PATHS, key)
            size = p.stat().st_size / 1e6
            print(f"  {key:20s} {size:9.1f} MB  {p}")
        except FileNotFoundError as e:
            print(f"  {key:20s} NOT FOUND  ({e})")

    frames = {}
    for key in ("train_source1", "train_source2", "train_source3",
                "test_source1", "test_source2", "test_source3"):
        try:
            frames[key] = read_source(getattr(config.PATHS, key))
        except Exception as e:  # noqa: BLE001
            print(f"could not read {key}: {e}")

    hr("SHAPES + COUNTRY BREAKDOWN")
    for key, df in frames.items():
        print(f"\n{key}: {len(df):,} rows, columns={list(df.columns)}")
        print(df["country"].value_counts().to_string())
        print("  sample:")
        print(df.head(3).to_string(max_colwidth=60))

    hr("MISSINGNESS")
    for key, df in frames.items():
        empt = {c: int((df[c].str.strip() == "").sum()) for c in
                ("business_name", "business_address", "country")}
        print(f"{key:16s} empty-> {empt}")

    hr("ENTITY ID PREFIXES")
    for key, df in frames.items():
        print(f"{key:16s} {Counter(df['entity_id'].str.split('-').str[0]).most_common()}")

    # ---- ground truth ----------------------------------------------------
    try:
        gt = read_ground_truth(config.PATHS.train_ground_truth)
    except Exception as e:  # noqa: BLE001
        print(f"\ncould not read ground truth: {e}")
        return 0

    hr("GROUND TRUTH")
    n = len(gt)
    sizes = gt["matches"].map(len)
    singles = int((sizes == 0).sum())
    print(f"source1 entities in GT : {n:,}")
    print(f"singletons (no match)  : {singles:,}  ({singles / n:.2%})  "
          f"<- each is worth a free 1.0 if predicted empty")
    print(f"total match links      : {int(sizes.sum()):,}")
    print("\nmatches-per-entity distribution:")
    print(sizes.value_counts().sort_index().head(15).to_string())
    print(f"\nmean={sizes.mean():.3f} median={sizes.median():.0f} max={sizes.max()}")

    # how many matches come from each source
    src = Counter()
    for ms in gt["matches"]:
        for m in ms:
            src[m.split("-")[0]] += 1
    print(f"\nmatch links by source : {dict(src)}")

    # does any source2/3 record appear under more than one source1 entity?
    seen = Counter()
    for ms in gt["matches"]:
        for m in ms:
            seen[m] += 1
    multi = sum(1 for v in seen.values() if v > 1)
    print(f"source2/3 ids reused across >1 source1 entity: {multi:,} "
          f"of {len(seen):,} matched ids  "
          f"({'one-to-one constraint is safe' if multi == 0 else 'do NOT force one-to-one'})")

    # coverage: are all source2/3 records matched to something?
    if "train_source2" in frames and "train_source3" in frames:
        all23 = set(frames["train_source2"]["entity_id"]) | set(frames["train_source3"]["entity_id"])
        matched = set(seen)
        print(f"source2+3 records total      : {len(all23):,}")
        print(f"        ...appearing in GT   : {len(matched & all23):,} "
              f"({len(matched & all23) / max(len(all23), 1):.2%})")
        stray = matched - all23
        if stray:
            print(f"        WARNING: {len(stray):,} GT ids not present in source files")

    # per-country singleton rate — tells us whether the gate should differ by country
    if "train_source1" in frames:
        s1 = frames["train_source1"][["entity_id", "country"]]
        j = gt.merge(s1, left_on="source1_entity_id", right_on="entity_id", how="left")
        j["n"] = j["matches"].map(len)
        hr("PER-COUNTRY GROUND TRUTH")
        print(j.groupby("country")["n"].agg(
            entities="size", singleton_rate=lambda x: (x == 0).mean(),
            mean_matches="mean", max_matches="max").to_string())

    hr("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
