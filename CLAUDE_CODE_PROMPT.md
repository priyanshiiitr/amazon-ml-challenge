# Prompt for Claude Code

Paste everything below the line into Claude Code, running on the machine that
has the data (or on the GPU box, once the data is there).

---

You are helping me win the **Amazon ML Challenge 2026**. Submissions close
**27 Sep 2026, 23:59 IST**, with a maximum of **5 leaderboard submissions per
day**. Treat submission budget as a scarce resource: never upload a file you
have not first scored on a local held-out split and passed through the official
validator.

## The task

**Business Entity Resolution.** Three independent sources of business records,
no shared identifiers. Source 1 is a deduplicated reference. For **every**
Source 1 entity, output the set of Source 2 / Source 3 records that refer to the
same real-world business. An entity may match **zero, one, or many** records.

Every file is **tab-separated**. Always read with `sep="\t"` — without it pandas
silently returns a single column.

Columns in each source file: `entity_id` (prefixed `S1-`/`S2-`/`S3-`),
`business_name`, `business_address`, `country`.
`train_ground_truth.tsv`: `source1_entity_id`, `matched_entity_ids`
(comma-separated, **empty when the entity has no matches**).

### Data location and scale

```
<root>/student_resource/
├── dataset/train/train_source1.tsv        210 MB
├── dataset/train/train_source2.tsv        489 MB
├── dataset/train/train_source3.tsv        504 MB
├── dataset/train/train_ground_truth.tsv   127 MB
├── dataset/test/test_source1.tsv          175 MB
├── dataset/test/test_source2.tsv          509 MB
├── dataset/test/test_source3.tsv          506 MB
├── utils/validate_submission.py           (official, stdlib only)
├── Documentation_template.md
└── README.md
```

Roughly **millions of Source 1 entities against ~10M Source 2/3 records**.
Brute-force pairwise comparison is ~10¹³ pairs and is not an option. Blocking
quality is the single biggest lever in this competition — it sets a hard ceiling
on recall that no downstream model can recover.

### The metric — read this twice

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

computed **per Source 1 entity**, then **macro-averaged over all Source 1
entities**, singletons included.

Three consequences that should shape every design decision:

1. **Precision is weighted 2× over recall.** A false merge hurts roughly twice
   as much as a missed link. When in doubt, predict fewer matches.
2. **A singleton scores a full 1.0 for an empty prediction, and a flat 0.0 if
   you predict anything at all.** If singletons are, say, 35% of entities,
   the "does this entity have any match at all" gate is worth more than all
   within-entity ranking combined. Measure the singleton rate first and let it
   drive your effort allocation.
3. **Macro, not micro.** An entity with 1 candidate and an entity with 50
   candidates count equally. Per-entity relative thresholds beat a single global
   probability cut. Verify this rather than assuming it.

### Output format

Two tab-separated files in `output/`:

- **`matching_results.tsv`** — columns `source1_entity_id`, `matched_entity_ids`.
  The only file scored on the leaderboard.
- **`candidate_pairs.tsv`** — columns `source1_entity_id`, `candidate_entity_ids`.
  Must be **exactly the set your model runs inference over** — the last blocking
  stage, not a raw early pass you later filter. Not scored, but audited.

Rules, all of which cause rejection: every Source 1 test entity gets exactly one
row; empty string for singletons; no duplicate IDs within a list; no duplicate
`source1_entity_id` rows; only `S2-`/`S3-` IDs that exist in the test set; final
matches must be a subset of candidates.

### Hard constraints

- **No external data, APIs, databases or geocoding services.** Any external
  lookup of business identities is immediate disqualification. Only the provided
  training data. This includes pretrained *knowledge* lookups — a general
  pretrained encoder is fine, querying a business registry is not.
- Any final model must be **MIT/Apache-2.0 licensed and ≤8B parameters**.
- The test set contains **France**, which never appears in training (train is US
  and India only). Do not hard-code, filter or one-hot on country. Every French
  entity must still appear in the submission.

## What already exists — start here, do not start from scratch

There is a working, validated pipeline at `code/business_entity_resolution/`.
It has been smoke-tested end to end on synthetic data with the same schema: it
produces both TSVs and **passes the official validator**, including on an unseen
third country. Read `README.md` there first, then `src/run_pipeline.py`.

```
src/config.py        paths (auto-discovers the dataset) + all hyper-parameters
src/normalize.py     accent stripping, legal-suffix removal, address abbreviation
                     expansion (US + India + France union), postal/number extraction
src/data_io.py       TSV loading with the right quoting, ID-list parsing
src/inspect_data.py  scale + ground-truth statistics
src/blocking.py      char n-gram TF-IDF top-K (name / address / combined, both
                     directions) + deterministic key joins, per country
src/features.py      ~55 pair features: rapidfuzz similarities, IDF-weighted token
                     overlap, address numerics, rank-relative-within-entity
src/model.py         LightGBM over candidate pairs, split by entity
src/decision.py      (tau_empty, tau_abs, alpha) + one-to-one, grid-searched
                     against the real metric with vectorised bincount scoring
src/evaluate.py      the exact competition metric + a diagnostic breakdown
src/submission.py    writes both TSVs, enforces every rule, runs the official validator
```

Run it:

```bash
pip install -r requirements.txt
python -m src.run_pipeline inspect
python -m src.run_pipeline blocking --sample 20000
python -m src.run_pipeline train   --sample 20000
python -m src.run_pipeline predict --sample 2000
```

## Your job, in order

**Step 1 — Measure before you model.** Run `inspect`. Report back: row counts per
file, the **singleton rate**, the distribution of matches per entity, the split
of matches between S2 and S3, whether any S2/S3 record is matched to more than
one S1 entity (this decides whether the one-to-one assignment constraint is
valid), and the per-country breakdown. These numbers determine everything that
follows — do not skip this and do not guess them.

**Step 2 — Make blocking work at full scale.** This is the hard engineering
problem. The current implementation uses `sparse_dot_topn` over char n-gram
TF-IDF per country, which may not survive millions × millions on your hardware.
Profile it on a sample, extrapolate, and if it will not finish, change the
approach — candidate directions: word-level TF-IDF with IDF-pruned inverted
indexes; a cheap exact-key pre-block before the expensive n-gram pass; ANN over
GPU-computed embeddings (FAISS / hnswlib); or sharding by a coarse key.
**Report `link_recall` (the recall ceiling) and the reduction ratio.** Do not
move on until recall ceiling is ≥0.97 at a candidate count you can actually
afford to featurise. If you must trade, remember F_0.5 forgives recall more than
precision — but blocking recall is a *ceiling*, which is different: you can
always throw candidates away later, never get them back.

**Step 3 — Train and validate honestly.** Split by **entity**, never by pair.
Score with `evaluate.macro_f_beta` on entities that produced no candidates too —
they are part of the average. Use `detailed_report` to see whether you are
losing points on singletons or on non-singletons, and attack whichever is worse.

**Step 4 — Tune the decision layer.** This is usually worth more than another
hundred LightGBM rounds. Check whether the tuned optimum sits on a grid boundary
and widen the grid if so.

**Step 5 — Predict, validate, submit.** Run the official validator before every
upload. Predict on the **full** test set; check France entities specifically —
confirm they get sensible predictions rather than all-empty or all-matched.

**Step 6 — Iterate with the budget in mind.** Keep a log of every submission:
what changed, local validation score, leaderboard score. If local and
leaderboard diverge sharply, your validation split is lying to you — fix that
before tuning anything else.

## Ideas worth trying once the baseline is scored

- **A cross-encoder reranker** on the top few candidates per entity: a small
  MIT/Apache multilingual encoder (e.g. `intfloat/multilingual-e5-small`, 118M,
  or a small MiniLM) fine-tuned on the training pairs, used as an extra feature
  feeding the LightGBM rather than replacing it. GPU makes this cheap. Watch the
  licence and parameter limits.
- **Embedding retrieval as an additional blocking channel**, specifically to
  help France, where the character n-gram statistics differ from training.
- **Transitivity / graph structure**: if S2-x and S3-y are both confidently
  matched to S1-a and are highly similar to each other, that is mutual evidence.
- **Calibration per country**, with France falling back to a global setting.
- **Abstention tuning**: explicitly optimise the singleton gate on its own,
  since each singleton decision is worth a full 1.0.

## How I want you to work

- Always smoke-test on `--sample` before a full run. Full runs are expensive.
- Report real numbers, not estimates. If something takes 40 minutes, say so and
  run it in the background rather than guessing at its output.
- Tell me the recall ceiling, the validation F_0.5, and the singleton/non-singleton
  breakdown at every iteration. Those three numbers are the state of the project.
- Do not silently change the output format. The validator must pass.
- Flag honestly when something is not working rather than tuning around it.

Start with Step 1 and report the numbers.
