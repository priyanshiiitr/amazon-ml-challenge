"""Loading and saving. Everything is tab-separated with an explicit sep='\\t'."""
from __future__ import annotations

import pandas as pd

from . import config
from .normalize import normalise_frame

EXPECTED_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        quoting=3,  # csv.QUOTE_NONE — addresses contain stray quote characters
        on_bad_lines="warn",
        engine="c",
    )
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; got {list(df.columns)}")
    for c in EXPECTED_COLS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["country_key"] = df["country"].str.lower()
    return df


def read_ground_truth(path) -> pd.DataFrame:
    gt = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=3
    )
    gt.columns = [c.strip() for c in gt.columns]
    if "source1_entity_id" not in gt.columns:
        raise ValueError(f"unexpected ground-truth columns: {list(gt.columns)}")
    mcol = [c for c in gt.columns if c != "source1_entity_id"][0]
    gt = gt.rename(columns={mcol: "matched_entity_ids"})
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("").astype(str)
    gt["matches"] = gt["matched_entity_ids"].map(parse_id_list)
    return gt[["source1_entity_id", "matched_entity_ids", "matches"]]


def parse_id_list(s) -> set[str]:
    if not isinstance(s, str) or not s.strip():
        return set()
    return {t.strip() for t in s.split(",") if t.strip()}


def format_id_list(ids) -> str:
    """Deterministic order so reruns diff cleanly. No spaces after commas."""
    return ",".join(sorted(ids))


def load_split(split: str, normalise: bool = True):
    """split in {'train','test'} -> (s1, s2, s3) with derived columns attached."""
    if split == "train":
        paths = (config.PATHS.train_source1, config.PATHS.train_source2,
                 config.PATHS.train_source3)
    elif split == "test":
        paths = (config.PATHS.test_source1, config.PATHS.test_source2,
                 config.PATHS.test_source3)
    else:
        raise ValueError(split)

    frames = []
    for p in paths:
        df = read_source(p)
        if normalise:
            normalise_frame(df)
        frames.append(df)
    return frames


def write_tsv(df: pd.DataFrame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=3, escapechar=None)
    return path
