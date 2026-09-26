# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve entities with a three-stage pipeline: script-normalising every record
into a comparable Latin form, retrieving candidates with an IDF-weighted
inverted index over discrete keys, and scoring pairs with LightGBM behind a
decision layer tuned directly against the competition metric. The two decisions
that mattered most were both discovered by measurement rather than assumed: a
**cross-script consonant-skeleton key** that makes Indic-script Source 2/3 names
comparable to Latin Source 1 names (mean similarity on affected pairs rises from
4.3 to 86.0), and **address-only blocking keys**, which recover the large class
of true matches where the business name has been replaced outright but the
address survives. Held-out macro F_0.5 is **0.8989**.

---

## 2. Methodology

### 2.1 Problem Analysis

We measured the training ground truth before designing anything. Several results
contradicted reasonable priors and redirected the work:

| Property | Measured value |
| --- | --- |
| Source 1 entities | 2,206,821 |
| Total match links | 7,638,365 |
| **Singleton rate** | **5.58%** |
| Matches per entity | mean 3.461, median 4, max 11 (85% have 2–6) |
| Links by source | S2 48.4%, S3 51.6% |
| **S2/S3 ids reused across entities** | **0 of 7,638,365** |
| Unmatched (noise) S2/S3 records | 26.0% |

Three consequences:

1. **The singleton gate is a minor term.** At a 5.58% singleton rate, correctly
   abstaining is worth ~5.6% of the macro score. Within-entity ranking dominates,
   so we deliberately did *not* over-invest in abstention.
2. **One-to-one assignment is provably safe.** Every matched Source 2/3 record is
   claimed by exactly one Source 1 entity — zero violations across 7.6M links. We
   exploit this as a hard constraint in the decision layer.
3. **Country is not a usable feature.** US and India have statistically identical
   ground-truth distributions (singleton rate 5.58% vs 5.59%; mean matches 3.459
   vs 3.465). Combined with France appearing only at test time, this told us to
   make every rule country-agnostic rather than country-conditioned.

**Noise patterns found in real matched pairs.** The most consequential was not in
the problem statement:

- **Writing system.** Source 1 is 100% Latin, but ~23% of Indian Source 2/3
  *names* are written in one of nine Indic scripts (Devanagari 6.2% of all S2
  rows, plus Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Gurmukhi,
  Oriya). A naive `[^0-9a-z]` strip maps these to the empty string, silently
  discarding ~9% of all Source 2/3 records.
- **Name destruction with address preservation.** In a large class of true pairs
  the right-hand name is replaced by an unrelated token while the address stays
  nearly identical:

  ```
  S1  Probst & Duran Newhold LLC | Boston, 31 Floyd Street, MA
  S2  NYLADREX                   | 31D FLOYD STREET, BOSTON, MA     name 31 / addr 98
  S1  SJ Ace Vendome Inc         | 60 Uneeda Street, Madison, WV
  S2  #sjace                     | UNEEDA STREET, MADISON, WV       name 20 / addr 100
  ```
  Measured on 6,000 entities, **68.6%** of true links that no name-derived key
  could reach were of this form.
- Domain-name substitution (`dprobst.com`), handle prefixes (`@`, `#`), junk
  prefixes (`--`, `<<`), spurious accents (`Léarning`), US state abbreviation vs
  full name (`TX` vs `Texas`), address component reordering, and case changes.
- ~3.3% of Source 2/3 addresses are empty; no names are empty.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier + metric-tuned decision layer (hybrid).

**Core Innovation:** A *voicing- and aspiration-folded consonant skeleton*,
applied after rule-based Indic→Latin transliteration, used simultaneously as a
blocking key and a similarity feature. It collapses the three systematic ways a
transliterated name drifts from its English spelling:

| drift | example | skeleton |
| --- | --- | --- |
| inherent-vowel insertion | `devalapars` / `developers` | `TVLPLS` |
| no voicing distinction (Tamil) | `kulopal` / `global` | `KLPL` |
| aspiration spelling | `fud` / `food` | `PT` |

Measured on 10,193 true (Latin S1, Indic S2/S3) pairs, mean `token_set_ratio`
rises **4.3 → 86.0**; on random non-matching pairs it rises only 30.6 → 46.0, so
discrimination is preserved. It also helps Latin-only pairs slightly (93.2 →
93.9), so it is applied universally rather than conditionally.

---

## 3. Candidate Generation (Blocking)

Character n-gram TF-IDF with sparse top-N matrix products was our first
approach and does not survive this scale — it materialises the transpose of a
~5M × 400k sparse matrix per country and merges >150M rows. We replaced it with
an **IDF-weighted inverted index over discrete keys**, where cost is
proportional to actual key collisions rather than to the product of the two
sides.

**Blocking keys used** (all computed on the consonant skeleton, all
country-agnostic):

| tag | key | purpose |
| --- | --- | --- |
| `F` | whole-name skeleton, token-sorted | high precision; mean bucket 1.33 |
| `T` | individual skeleton tokens | broad recall net |
| `B` | adjacent skeleton-token bigrams | selective |
| `G` | 4-gram skeleton shingles, deterministically subsampled to 25% | rescues near-miss skeletons |
| `PA` | postal code + distinctive street word | **address-only** |
| `NA` | street number (letter-suffix stripped) + street word | **address-only**, survives missing postal |
| `AB` | adjacent distinctive street words | **address-only**, for numberless addresses |

Each channel carries its own document-frequency cap (`G` at 250, address
channels at 400, name channels at 2,000–4,000). Keys above the cap are
stopword-like: they contribute almost no IDF and dominate cost. Without
per-channel caps the shingle channel alone generated ~3,000 candidates per
entity (~5×10⁹ pair-touches at full scale).

Pairs are scored by **cosine over IDF weights**, not by raw summed IDF — without
L2 normalisation, records with many keys (long names and addresses) outrank the
true match on every query.

**Engineering.** Keys are hashed to int64 on creation and the strings discarded
(a vocabulary dict over ~10M distinct keys costs >1.5 GB by itself); lookup is
`np.searchsorted`. The right side is indexed one ~1M-record shard at a time and
merged through a running top-K, so peak memory is set by shard size rather than
by data size. Shards are independent, so they are queried in parallel.

**Candidate pairs generated:** **207,880,980** for the test set (119.99 per
Source 1 entity, K=120), produced in 2,025 s on 16 cores.

**How we ensured true matches were not lost.** We measured the recall ceiling
directly against the **full 10.3M-record right side**, because a reduced
distractor pool overstates it badly — the same configuration scores 0.954
against a 250k pool and **0.847** against the real 10.3M:

| configuration | link recall |
| --- | --- |
| name-derived keys only, K=25 | 0.8052 |
| + address-only keys + shingles, K=25 | 0.8475 |
| + address-only keys + shingles, K=40 | 0.8698 |
| + address-only keys + shingles, **K=120** | **0.9073** |

Reachability at K=∞ (i.e. sharing at least one key) is 0.9934 for India and
0.9953 for US, so the residual loss is ranking, not blocking design. Because
β=0.5 discounts recall, a 0.907 ceiling with perfect precision still permits a
per-entity F_0.5 of 0.978, so we stopped widening K there and spent the
remaining effort on precision.

---

## 4. Matching Model

**Features used** (36 total, all country-agnostic by construction):

- **Name features:** `token_set_ratio`, `token_sort_ratio`, plain `ratio` and
  `partial_ratio` over the normalised name; token intersection and Jaccard.
- **Skeleton features:** `token_set_ratio` and `ratio` over the consonant
  skeleton — the cross-script channel.
- **Address features:** `token_set_ratio`, `token_sort_ratio`, `ratio`; token
  intersection and Jaccard; numeric-token intersection and Jaccard; postal-code
  equality and both-present flags; empty-address flag.
- **Other:** blocking cosine score; acronym equality; name lengths, token counts
  and length ratio; candidate source (S2 vs S3).
- **Rank-relative features** (within each Source 1 entity): for the blocking
  score and for name/skeleton/address similarity, the ratio to that entity's best
  candidate and the gap from it, plus the candidate's rank and the entity's
  candidate count. The metric is macro-averaged per entity, so an entity with 2
  candidates and one with 120 carry equal weight; these features let the model
  express a per-entity decision that an absolute similarity cannot.

**Deliberately excluded:** the `country` field. The test set contains France with
no training representation; a model given `country` would learn US/India-specific
behaviour and apply something arbitrary to 259,452 French entities.

**Model type:** LightGBM binary classifier (gradient-boosted trees), 2,876
rounds, early-stopped on a held-out split. Trained on 28.8M candidate pairs from
300,000 Source 1 entities; **split by entity, never by pair**.

Top features by gain: `addr_tsr_rel` (rank-relative address similarity),
`addr_tok_jacc`, `name_par`, `addr_tsr`, `skel_rank`. That the top feature is
*rank-relative* rather than absolute is the macro-metric structure surfacing in
the model.

**Threshold selection method:** direct F_0.5 optimisation on a held-out split of
60,000 entities. A single global probability cut cannot express this metric, so
we tune three knobs jointly (3,480 configurations, scored with vectorised
`bincount` arithmetic rather than Python set operations):

- `tau_empty` — predict `[]` if the entity's best candidate falls below it
- `tau_abs` — absolute floor for keeping a candidate
- `alpha` — keep candidates scoring ≥ `alpha ×` the entity's best

plus an optional one-to-one assignment pass, justified by the zero-reuse finding.
Selected: `tau_empty=0.60`, `tau_abs=0.425`, `alpha=0.70`, one-to-one enabled.
All three sit interior to their search grids, so widening them does not help.

Entities that produced *no* candidates are still scored — each is a correct
empty prediction worth a full 1.0, and omitting them would inflate the estimate.

---

## 5. Results & Error Analysis

**F_0.5 Score (macro):** **0.89886** on 60,000 held-out entities.

| metric | value |
| --- | --- |
| Macro F_0.5 | **0.89886** |
| Score on singletons | 0.85185 |
| Score on non-singletons | 0.90161 |
| Link precision | 0.9760 |
| Link recall | 0.8028 |
| False merges on singletons | 492 of 3,321 |
| Non-singletons left empty | 1,563 |

**Common false positives (wrong merges).** Concentrated on singletons: 492 of
3,321 singletons received at least one prediction, and each costs a full 1.0.
These are entities whose true answer is "nothing" but which have a
high-similarity distractor — typically a same-street, same-postal business with a
generic name. This is the single largest remaining block of recoverable score.

**Common false negatives (missed matches).** Two distinct causes:

1. *Blocking-limited* — 9.3% of true links never became candidates at K=120.
   Reachability analysis shows almost all of these are retrievable in principle
   (0.993/0.995 at K=∞); they are out-ranked by distractors rather than absent.
2. *Decision-limited* — link recall (0.803) sits below the blocking ceiling
   (0.907), so the precision-weighted thresholds discard retrieved true matches.
   This is a deliberate trade: β=0.5 penalises a false merge roughly twice as
   much as a miss.

---

## 6. Conclusion

A well-measured simple pipeline outperformed our initial instincts at every
stage: the decisive wins came from *measuring the data* (the 5.58% singleton
rate, zero id reuse, the Indic-script gap, and address-preserving name
destruction) rather than from model complexity. The two highest-value changes —
cross-script skeleton keys and address-only blocking keys — were each identified
by categorising actual failures rather than by tuning. The clearest remaining
headroom is precision on singletons, where 492 false merges on a 3,321-singleton
sample each forfeit a full point of macro score.

---

## Appendix

### A. Code Artefacts

All source ships under `code/business_entity_resolution/src/`, with `README.md`
and pinned `requirements.txt`.

| Module | Responsibility |
| --- | --- |
| `config.py` | Paths (auto-discovered) and all hyper-parameters |
| `translit.py` | Dependency-free Indic→Latin transliteration over nine scripts |
| `normalize.py` | Accent stripping, legal-suffix removal, address abbreviation expansion, consonant skeleton |
| `prep.py` | Streams raw TSVs into normalised parquet at bounded memory |
| `inspect_fast.py` | Streaming ground-truth and scale statistics |
| `retrieve.py` | Hashed-key inverted index, shard indexing, top-K query |
| `run_blocking.py` | Full-scale candidate generation, parallel across shards |
| `pairfeat.py` | The 36 pair features, vectorised via `rapidfuzz.process.cpdist` |
| `run_model.py` | LightGBM training, decision tuning, prediction, submission writing |
| `decision.py` | `(tau_empty, tau_abs, alpha)` + one-to-one, grid-searched on the real metric |
| `evaluate.py` | The competition metric and a singleton/non-singleton breakdown |

**Reproduction:**

```bash
pip install -r requirements.txt
python -m src.prep train && python -m src.prep test
python -m src.run_blocking train --sample 300000 --top-k 120 --workers 8 \
       --out work/cands_train_300k.parquet
python -m src.run_model  train   --cands work/cands_train_300k.parquet
python -m src.run_blocking test  --top-k 120 --workers 8 \
       --out work/cands_test_k120.parquet
python -m src.run_model  predict --cands work/cands_test_k120.parquet
```

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
`candidate_pairs.tsv` is the exact set the model runs inference over — the last
blocking stage, not an earlier pass later filtered. Random seeds are fixed
(`SEED=42`).

### B. Additional Results

**Transliteration validation**, 10,193 true (Latin S1, Indic S2/S3) pairs:

| representation | mean | median | ≥60 | name emptied |
| --- | --- | --- | --- | --- |
| prior normalisation | 4.3 | 0.0 | 4.3% | 95.7% |
| + transliteration | 62.2 | 62.1 | 56.7% | 0% |
| + consonant skeleton | **86.0** | 92.3 | **93.4%** | 0% |

**Discrimination check** (skeleton similarity, true vs random pairs):

| pair type | name ratio | skeleton |
| --- | --- | --- |
| true, Latin (n=19,144) | 93.2 | 93.9 |
| true, Indic (n=1,525) | 67.9 | **90.6** |
| random, Latin (n=8,714) | 30.6 | 46.0 |
| random, Indic (n=648) | 30.2 | 46.5 |

Random pairs reach ≥85 only 0.1% of the time, so the skeleton's lossiness does
not compromise separation.

**Unreachable-link categorisation** (6,000 entities; 8.73% of links unreachable
before address-only keys were added):

| cause | share |
| --- | --- |
| name differs, address matches | 68.6% |
| name similar, key too rigid | 30.2% |
| both differ | 1.0% |
| empty right-hand address | 0.3% |
