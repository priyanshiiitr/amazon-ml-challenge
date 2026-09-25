"""Step 1: streaming statistics. Flat memory — never loads a frame.

The full-scale files are 2.2M x 10M records; the pandas-based inspect_data.py
needs more RAM than an 8 GB laptop has. This one walks each file line by line
and keeps only fixed-size numpy arrays, so it runs anywhere.

  python -m src.inspect_fast
"""
from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np

from . import config


def hr(t=""):
    print("\n" + "=" * 74)
    if t:
        print(t)
        print("=" * 74)


def enc(eid: str) -> int:
    """'S2-681193310' -> int64 key that keeps the source separable."""
    src = eid[1]
    return (int(src) << 32) | int(eid[3:])


def scan_source(path, want_ids=False):
    """Row count, country histogram, empty-field counts, optional id/country arrays."""
    countries = Counter()
    n = 0
    empty_name = empty_addr = empty_ctry = 0
    ids = [] if want_ids else None
    ctry_codes = [] if want_ids else None
    code_of: dict[str, int] = {}

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        ix = {c.strip(): i for i, c in enumerate(header)}
        i_id, i_nm = ix["entity_id"], ix["business_name"]
        i_ad, i_ct = ix["business_address"], ix["country"]
        for line in fh:
            f = line.rstrip("\r\n").split("\t")
            if len(f) <= i_ct:
                continue
            n += 1
            c = f[i_ct].strip()
            countries[c] += 1
            if not f[i_nm].strip():
                empty_name += 1
            if not f[i_ad].strip():
                empty_addr += 1
            if not c:
                empty_ctry += 1
            if want_ids:
                if c not in code_of:
                    code_of[c] = len(code_of)
                ids.append(enc(f[i_id]))
                ctry_codes.append(code_of[c])

    out = {
        "rows": n, "countries": countries, "path": path,
        "empty": {"name": empty_name, "address": empty_addr, "country": empty_ctry},
    }
    if want_ids:
        out["ids"] = np.asarray(ids, dtype=np.int64)
        out["ctry"] = np.asarray(ctry_codes, dtype=np.int8)
        out["ctry_names"] = {v: k for k, v in code_of.items()}
    return out


def scan_ground_truth(path):
    """Per-entity match counts plus every matched id, as flat arrays."""
    s1_ids: list[int] = []
    sizes: list[int] = []
    n_s2: list[int] = []
    links: list[int] = []

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        fh.readline()  # header
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            sid, _, rest = line.partition("\t")
            s1_ids.append(enc(sid))
            if not rest.strip():
                sizes.append(0)
                n_s2.append(0)
                continue
            parts = [p for p in rest.split(",") if p.strip()]
            k2 = 0
            for p in parts:
                p = p.strip()
                links.append(enc(p))
                if p[1] == "2":
                    k2 += 1
            sizes.append(len(parts))
            n_s2.append(k2)

    return (np.asarray(s1_ids, dtype=np.int64),
            np.asarray(sizes, dtype=np.int32),
            np.asarray(n_s2, dtype=np.int32),
            np.asarray(links, dtype=np.int64))


def main():
    t0 = time.time()
    hr("RESOLVED PATHS")
    print(f"project root : {config.PROJECT_ROOT}")
    keys = ("train_source1", "train_source2", "train_source3", "train_ground_truth",
            "test_source1", "test_source2", "test_source3", "validator")
    paths = {}
    for key in keys:
        try:
            p = getattr(config.PATHS, key)
            paths[key] = p
            print(f"  {key:20s} {p.stat().st_size / 1e6:9.1f} MB  {p}")
        except FileNotFoundError as e:
            print(f"  {key:20s} NOT FOUND ({e})")

    hr("SCALE + COUNTRY BREAKDOWN")
    stats = {}
    for key in ("train_source1", "train_source2", "train_source3",
                "test_source1", "test_source2", "test_source3"):
        if key not in paths:
            continue
        want = key.endswith("source1")
        st = scan_source(paths[key], want_ids=want)
        stats[key] = st
        tot = st["rows"]
        print(f"\n{key}: {tot:,} rows   [{time.time() - t0:.0f}s]")
        for c, k in st["countries"].most_common():
            print(f"    {c or '<empty>':10s} {k:>12,}  ({k / tot:6.2%})")
        print(f"    empty fields: {st['empty']}")

    if "train_ground_truth" not in paths:
        return 0

    hr("GROUND TRUTH")
    s1_ids, sizes, n_s2, links = scan_ground_truth(paths["train_ground_truth"])
    n = len(s1_ids)
    singles = int((sizes == 0).sum())
    print(f"source1 entities in GT : {n:,}")
    print(f"unique source1 ids     : {len(np.unique(s1_ids)):,}")
    print(f"SINGLETON RATE         : {singles:,} / {n:,} = {singles / n:.4%}")
    print(f"total match links      : {len(links):,}")
    nz = sizes[sizes > 0]
    print(f"mean matches (all)     : {sizes.mean():.3f}")
    print(f"mean matches (non-sgl) : {nz.mean():.3f}   median={np.median(nz):.0f}  max={sizes.max()}")

    print("\nmatches-per-entity distribution:")
    vals, cnts = np.unique(sizes, return_counts=True)
    for v, c in list(zip(vals, cnts))[:16]:
        print(f"    {v:>3} matches : {c:>10,}  ({c / n:6.2%})")
    if len(vals) > 16:
        tail = int(cnts[16:].sum())
        print(f"    >{vals[15]:>2} matches : {tail:>10,}  ({tail / n:6.2%})")

    k2 = int(n_s2.sum())
    k3 = len(links) - k2
    print(f"\nlinks by source        : S2={k2:,} ({k2 / len(links):.2%})  "
          f"S3={k3:,} ({k3 / len(links):.2%})")
    ent_with_s2 = int((n_s2 > 0).sum())
    ent_with_s3 = int(((sizes - n_s2) > 0).sum())
    print(f"entities with >=1 S2   : {ent_with_s2:,} ({ent_with_s2 / n:.2%})")
    print(f"entities with >=1 S3   : {ent_with_s3:,} ({ent_with_s3 / n:.2%})")

    # --- the one-to-one question -----------------------------------------
    hr("IS THE ONE-TO-ONE CONSTRAINT VALID?")
    uniq, counts = np.unique(links, return_counts=True)
    reused = int((counts > 1).sum())
    print(f"distinct S2/S3 ids in GT : {len(uniq):,}")
    print(f"...linked to >1 S1 entity: {reused:,} ({reused / len(uniq):.4%})")
    if reused:
        print(f"max reuse of a single id : {counts.max()}")
        rv, rc = np.unique(counts[counts > 1], return_counts=True)
        print("reuse histogram:", dict(zip(rv.tolist(), rc.tolist())))
    print("VERDICT:", "one-to-one assignment is SAFE" if reused == 0
          else "do NOT force a hard one-to-one constraint")

    # --- coverage of the right-hand sources ------------------------------
    if "train_source2" in stats and "train_source3" in stats:
        tot23 = stats["train_source2"]["rows"] + stats["train_source3"]["rows"]
        print(f"\nS2+S3 records total      : {tot23:,}")
        print(f"...appearing in GT       : {len(uniq):,} ({len(uniq) / tot23:.2%})")
        print(f"...never matched (noise) : {tot23 - len(uniq):,} "
              f"({1 - len(uniq) / tot23:.2%})")

    # --- per-country ground truth ----------------------------------------
    if "train_source1" in stats and "ids" in stats["train_source1"]:
        hr("PER-COUNTRY GROUND TRUTH (source1)")
        st = stats["train_source1"]
        order = np.argsort(st["ids"], kind="stable")
        sid_sorted = st["ids"][order]
        ctry_sorted = st["ctry"][order]
        pos = np.searchsorted(sid_sorted, s1_ids)
        pos = np.clip(pos, 0, len(sid_sorted) - 1)
        ok = sid_sorted[pos] == s1_ids
        cc = np.where(ok, ctry_sorted[pos], np.int8(-1))
        print(f"GT rows matched to a source1 row: {int(ok.sum()):,} / {n:,}")
        names = dict(st["ctry_names"])
        names[-1] = "<unmatched>"
        print(f"\n{'country':12s} {'entities':>12s} {'singleton%':>11s} "
              f"{'mean match':>11s} {'max':>5s}")
        for code in sorted(set(cc.tolist())):
            m = cc == code
            sub = sizes[m]
            print(f"{names.get(code, '?'):12s} {m.sum():>12,} "
                  f"{(sub == 0).mean():>10.2%} {sub.mean():>11.3f} {sub.max():>5}")

    hr(f"DONE in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
