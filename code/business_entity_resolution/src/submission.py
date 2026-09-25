"""Writing the two output TSVs, with every rule from the spec enforced here
rather than discovered by a failed upload."""
from __future__ import annotations

import pandas as pd

from . import config
from .data_io import format_id_list, write_tsv


def _rows(s1_ids, mapping: dict, valid_ids: set, col: str):
    out = []
    for sid in s1_ids:
        ids = mapping.get(sid, set())
        ids = {i for i in ids if i in valid_ids and (i.startswith("S2-") or i.startswith("S3-"))}
        out.append((sid, format_id_list(ids)))
    return pd.DataFrame(out, columns=["source1_entity_id", col])


def write_submission(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                     pred_map: dict, cand_map: dict, out_dir=None):
    """Every source1 entity gets exactly one row, singletons included."""
    out_dir = out_dir or config.OUTPUT_DIR
    valid = set(s2["entity_id"]) | set(s3["entity_id"])
    s1_ids = s1["entity_id"].tolist()

    # matches must be a subset of candidates, or the validator warns
    pred_map = {k: (v & cand_map.get(k, set())) for k, v in pred_map.items()}

    m = _rows(s1_ids, pred_map, valid, "matched_entity_ids")
    c = _rows(s1_ids, cand_map, valid, "candidate_entity_ids")

    assert len(m) == len(s1_ids) and m["source1_entity_id"].is_unique
    assert len(c) == len(s1_ids) and c["source1_entity_id"].is_unique

    mp = write_tsv(m, out_dir / "matching_results.tsv")
    cp = write_tsv(c, out_dir / "candidate_pairs.tsv")

    n_pred = int((m["matched_entity_ids"] != "").sum())
    print(f"[submission] {mp}")
    print(f"[submission] {cp}")
    print(f"[submission] {len(m):,} entities; {n_pred:,} with >=1 match "
          f"({n_pred / max(len(m), 1):.1%}); "
          f"{len(m) - n_pred:,} predicted singleton")
    return mp, cp


def run_official_validator(matching, candidate, test_dir=None):
    """Invoke the organisers' validate_submission.py so a rejection is caught here."""
    import subprocess
    import sys

    try:
        v = config.PATHS.validator
    except FileNotFoundError:
        print("[submission] official validator not found; skipping")
        return None
    test_dir = test_dir or config.PATHS.test_source1.parent
    cmd = [sys.executable, str(v), "--matching", str(matching),
           "--candidate", str(candidate), "--test-dir", str(test_dir)]
    print(f"[submission] running official validator")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        print("[submission] *** VALIDATOR FAILED — do not upload ***")
    return r.returncode
