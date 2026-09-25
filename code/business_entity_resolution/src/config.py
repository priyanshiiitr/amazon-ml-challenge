"""Central configuration. Edit PROJECT_ROOT if you move the folder."""
from pathlib import Path
import os

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
# src/ -> business_entity_resolution/ -> code/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(os.environ.get("AML_ROOT", PROJECT_ROOT))

# The dataset folder is auto-discovered (the zip may extract to different depths).
DATA_SEARCH_ROOTS = [PROJECT_ROOT]

WORK_DIR = PROJECT_ROOT / "work"        # intermediate artifacts (big, gitignored)
OUTPUT_DIR = PROJECT_ROOT / "output"    # the two submission TSVs
MODEL_DIR = WORK_DIR / "models"

for _d in (WORK_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _find_file(name: str):
    """Locate a dataset file anywhere under the project root."""
    for root in DATA_SEARCH_ROOTS:
        hits = sorted(root.rglob(name))
        # prefer shallower paths, and skip anything inside work/ or output/
        hits = [h for h in hits if WORK_DIR not in h.parents and OUTPUT_DIR not in h.parents]
        if hits:
            return min(hits, key=lambda p: len(p.parts))
    return None


class DataPaths:
    """Resolved lazily so import never fails before extraction."""

    def __init__(self):
        self._cache = {}

    def __getattr__(self, key):
        mapping = {
            "train_source1": "train_source1.tsv",
            "train_source2": "train_source2.tsv",
            "train_source3": "train_source3.tsv",
            "train_ground_truth": "train_ground_truth.tsv",
            "test_source1": "test_source1.tsv",
            "test_source2": "test_source2.tsv",
            "test_source3": "test_source3.tsv",
            "validator": "validate_submission.py",
        }
        if key not in mapping:
            raise AttributeError(key)
        if key not in self._cache:
            p = _find_file(mapping[key])
            if p is None:
                raise FileNotFoundError(
                    f"Could not find {mapping[key]} under {PROJECT_ROOT}. "
                    "Has student_resource.zip been extracted?"
                )
            self._cache[key] = p
        return self._cache[key]


PATHS = DataPaths()

# ----------------------------------------------------------------------------
# Pipeline hyper-parameters
# ----------------------------------------------------------------------------
SEED = 42

# --- blocking -----------------------------------------------------------
BLOCK_TOPK_FORWARD = 25      # top-K source2/3 candidates kept per source1 entity
BLOCK_CHUNK = 4_000          # left rows per retrieval chunk (raise it when RAM allows)
DF_CAP = 4_000               # global fallback cap on blocking-key frequency
# Per-channel caps, keyed by the tag each key string carries. A key more
# frequent than its cap is dropped: it is a stopword, carries almost no IDF,
# and dominates the cost. The shingle channel ("G") needs a far tighter cap
# than the rest -- it is the widest net and, left at 4000, it generated ~3000
# candidates per entity (~5e9 pair-touches at full scale).
DF_CAP_BY_TAG = {
    "F": 4_000,   # whole-name skeleton, selective by construction
    "T": 2_000,   # skeleton tokens
    "B": 2_000,   # skeleton bigrams
    "G": 250,     # 4-gram shingles -- rare ones only
    "P": 400,     # postal + street word
    "N": 400,     # street number + street word
    "A": 400,     # adjacent street words
}
BLOCK_WITHIN_COUNTRY = True  # only pair records sharing a country label

# --- embeddings (optional, set via env AML_USE_EMB=1) -------------------
USE_EMBEDDINGS = os.environ.get("AML_USE_EMB", "0") == "1"
EMB_MODEL = "intfloat/multilingual-e5-small"  # MIT licence, 118M params
EMB_TOPK = 20
EMB_BATCH = 512

# --- model --------------------------------------------------------------
VALID_FRACTION = 0.2         # share of source1 entities held out for tuning
LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=50,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbosity=-1,
    num_threads=0,
    seed=SEED,
)
LGB_ROUNDS = 3000
LGB_EARLY_STOP = 100

# --- decision layer -----------------------------------------------------
# grids searched by tune_decision(); widened automatically if the optimum sits
# on a boundary
TAU_EMPTY_GRID = [round(x, 3) for x in [i / 40 for i in range(4, 33)]]   # 0.10 .. 0.80
TAU_ABS_GRID = [round(x, 3) for x in [i / 40 for i in range(4, 33)]]
ALPHA_GRID = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ONE_TO_ONE_OPTIONS = [False, True]
