# Business Entity Resolution — Amazon ML Challenge 2026

Matches records from Source 2 and Source 3 to each Source 1 reference entity.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Expected layout (the code auto-discovers the dataset by filename, so only the
project root matters):

```
<project root>/
├── student_resource/dataset/train/train_source{1,2,3}.tsv
├── student_resource/dataset/train/train_ground_truth.tsv
├── student_resource/dataset/test/test_source{1,2,3}.tsv
├── code/business_entity_resolution/   <- this folder
├── work/                              <- intermediate artifacts (created)
└── output/                            <- matching_results.tsv, candidate_pairs.tsv
```

Override the root with `AML_ROOT=/path/to/project` if it is somewhere else.

## Reproduce end to end

```bash
cd code/business_entity_resolution

python -m src.run_pipeline inspect              # scale + ground-truth statistics
python -m src.run_pipeline blocking --sample 20000   # smoke test, prints recall ceiling
python -m src.run_pipeline train  --sample 20000     # features + model + tuned thresholds
python -m src.run_pipeline predict --sample 2000     # writes output/, runs the validator

python -m src.run_pipeline all                  # full scale
```

**Always run `--sample` first.** A full pass over the real data takes a long
time, and essentially every bug reproduces on twenty thousand rows.

## Pipeline

| Stage | Module | What it does |
| --- | --- | --- |
| Normalisation | `normalize.py` | Accent stripping, legal-suffix removal, address abbreviation expansion, postal/number extraction. Country-agnostic by construction. |
| Blocking | `blocking.py` | Union of char n-gram TF-IDF top-K over name, address and name+address (both directions), plus deterministic key joins. Per-country. |
| Pruning | `run_pipeline.prune_candidates` | Cuts to the top-N per entity. **This is what `candidate_pairs.tsv` reports** — the exact set the model scores, as the spec requires. |
| Features | `features.py` | ~55 features: rapidfuzz string similarities, IDF-weighted token overlap, address numerics, and rank-relative features within each entity. |
| Model | `model.py` | LightGBM binary classifier over candidate pairs. Split by entity, never by pair. |
| Decision | `decision.py` | Per-entity thresholds `(tau_empty, tau_abs, alpha)` plus optional one-to-one assignment, grid-searched against the real metric. |
| Output | `submission.py` | Writes both TSVs with every spec rule enforced, then shells out to the organisers' validator. |

## Why the decision layer is separate

The metric is macro-averaged per Source 1 entity, precision-heavy, and gives a
full 1.0 for correctly predicting *nothing* on a singleton. A single global
probability cut cannot express that. Three knobs are tuned jointly against the
real F_0.5 on a held-out split:

- `tau_empty` — if the best candidate for an entity scores below this, predict `[]`
- `tau_abs` — absolute floor for keeping a candidate
- `alpha` — keep candidates scoring at least `alpha × (best for this entity)`

`evaluate.detailed_report` breaks the score into singleton vs non-singleton
contributions so you can see which half is costing you.

## Notes on the unseen country

The test set contains France, absent from training. Nothing in the pipeline
branches on the country *value*: it is only ever used as a blocking key and an
equality feature. The address abbreviation table is a union of US, Indian and
French conventions, and accent stripping is applied everywhere. Per-country
threshold tuning falls back to the global setting for any country it never saw.

## Licensing

All dependencies are MIT / BSD / Apache-2.0. The optional embedding stage uses
`intfloat/multilingual-e5-small` (MIT, 118M parameters), well inside the 8B
limit. No external data, APIs or lookups are used at any point — only the
provided training files.
