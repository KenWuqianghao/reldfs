# Profile at the start (2026-09-24)

Written before the first experiment. It records the feasibility checks, the baseline
reproduction and the first version's weak spots. Candidate names (lgbm_base, c0_dfs) are
explained in EXPERIMENTS.md.

## 1. Feasibility evidence

| question | answer | evidence |
|---|---|---|
| boards and datasets | 3 boards, 31 tasks, 7 datasets; every board needs rel-amazon | `docs/rules.md` |
| board without rel-amazon | none exists | `LEADERBOARD_TASKS` |
| can rel-amazon be read without the 7.2 GB file | **yes** | see below |
| slim size of all 7 datasets | db 0.85 GB + task tables 0.41 GB = **1.25 GB** | column sizes from parquet footers |
| CPU cost on the largest dataset | ~2 min for a 9-feature pass over amazon user-churn | measured, see below |
| how submissions are checked | test-split prediction CSVs only; no code is run | `leaderboard_validate.yml` |
| relbench without CUDA torch | yes; the core has no torch dependency | wheel METADATA |

**rel-amazon remote read (measured).** `review.parquet` holds 20,862,040 rows in 20 row groups.
Its compressed columns are review_text 6,643 MB, summary 387 MB, customer_id 94 MB,
product_id 67 MB, rating 7.7 MB, verified 2.7 MB and review_time 0.1 MB. Reading the footer over
HTTP range requests (`HfFileSystem`, pyarrow) takes 1.5 s. Reading only
`review_time, customer_id, product_id, rating, verified` for all rows takes **24 s**, gives a
597 MB Arrow table, and writes a **123 MB** zstd parquet. The run peaked at 1.2 GB RSS.
`harness/fetch.py` does this for every table. It drops string columns whose compressed size is
at least 5 MB, and it logs what it dropped.

**Slim mirror size per dataset** (db tables without the big text columns, then hosted task
tables): amazon 184 + 254 MB, avito 269 + 25, event 119 + 0.2, f1 1 + 0.1, hm 106 + 88,
stack 89 + 30, trial 79 + 9. The full set is 1.25 GB. With the 0.54 GB virtual environment that is
1.8 GB. Keeping the task tables of only the leaderboard tasks, or holding one dataset at
a time, cuts it further.

**CPU (measured, amazon user-churn, the largest classification task).** The task has 4,708,383
train rows over 31 cutoffs. Nine per-customer aggregates (counts, windows, rating mean, recency,
distinct products) were built with one duckdb GROUP BY per cutoff on 2 threads: 62 s for train
and 9 s for val. LightGBM with 300 rounds on 2 threads fit the full 4.7M x 9 matrix in 63 s.
Peak RSS was 3.0 GB. Val AUROC was 0.685, against the test rows GNN 0.605, GNN+LightGBM 0.683
and Kapso 0.716 (different splits, so indicative only). Extrapolation for c0-size feature sets
(about 100 columns, train capped at 1M rows): 10-20 min per amazon/hm/stack task, and about
**1.5-3 h per full 12-task classification pass** on 2 workers. The regression board costs about
the same.

**Warning, measured.** A single duckdb range join of every cutoff against every review
(`JOIN review r ON r.review_time < ts.timestamp`) blew up to about 600M rows and filled the disk
through duckdb's spill directory. Never write that join shape. Build per cutoff, or use the
entity-keyed join that `reldfs/features.py` uses. Always set `temp_directory` and
`max_temp_directory_size`.

## 2. Baseline reproduction (official LightGBM, torch-free)

`lgbm_base` (a torch-free re-implementation of the official baseline, not in this
repository): entity-table columns only, 50k train sample, a 10-trial search tuned on
val. It uses no torch, so it has no GloVe text embedding, and the long text columns are not in
the mirror.

| task | val (ours) | test (ours) | board LightGBM (test) | delta |
|---|---|---|---|---|
| rel-f1/driver-dnf | 0.6245 | 0.6751 | 0.7255 | -0.050 |
| rel-f1/driver-top3 | 0.6359 | 0.6315 | 0.6493 | -0.018 |
| rel-trial/study-outcome | 0.6579 | 0.7188 | 0.6885 | +0.030 |
| mean | | 0.6751 | 0.6878 | -0.013 |

The test sets are tiny (702 / 726 / 825 rows), so the AUROC SE is about 0.02-0.03. The mean sits
within 0.013 of the board. The dnf gap is the largest: the official run embeds driver names
with GloVe, and its search picks a different model. That is a setup difference, not an
evaluator mismatch. The evaluator is the official `evaluate_task`.

## 3. c0 (automated time-aware aggregations + LightGBM), test report-only

| task | c0 val | c0 test | LightGBM | GNN | GNN+LGBM | RT-PluRel ft | Kapso |
|---|---|---|---|---|---|---|---|
| rel-f1/driver-dnf | 0.7896 | 0.6994 | 0.7255 | 0.7261 | 0.7157 | 0.8031 | 0.8327 |
| rel-f1/driver-top3 | 0.8800 | **0.8447** | 0.6493 | 0.6801 | 0.7248 | 0.9069 | 0.9357 |
| rel-trial/study-outcome | 0.6489 | **0.7019** | 0.6885 | 0.6693 | 0.6627 | 0.7319 | 0.8208 |
| mean of 3 | 0.7728 | **0.7487** | 0.6878 | 0.6918 | 0.7011 | 0.8140 | 0.8631 |

The leakcheck passes on 17 sampled cutoffs across the three tasks, and it catches a planted
leak (108 of 114 columns flagged). Feature build takes 1-5 s and each run 2-11 s.

## 4. Where c0 lost

- **driver-dnf: big val-to-test drop (0.79 -> 0.70) and early stopping at 36 rounds.** The
  label rate drifts: DNF share in train by decade is 0.86 / 0.88 / 0.88 / 0.92 / 0.88 / 0.82
  (1950s-2000s), 0.78 in val (2005-08) and probably lower in test (2010-13). The top features
  are per-driver means of result position, points and grid. They rank drivers but carry no era
  context. Fixes: recency-weighted or recent-only training, refitting on train+val for test,
  era-relative features (a driver's position relative to the field at the same cutoff), and
  the driver's own past DNF rate from `results.statusId` (c0 has only the top-8 status
  counts).
- **driver-top3: strong (0.845 test, above GNN+LGBM by 0.12), but 0.06 short of RT-PluRel.** The
  label is "top-3 in qualifying". Qualifying history is aggregated, but only over all time and
  the 4 x 30-day window. Missing: last-k-races features (the rows are per race, not per day),
  the constructor's current form (a 2-hop path driver -> results -> constructor ->
  constructor_standings), and the driver's rank within the cutoff.
- **study-outcome: at the GNN level (0.70), 0.03 short of RT-PluRel and 0.12 short of Kapso.**
  The direct children of a study are dated at the study start (checked for designs,
  eligibilities and sponsors_studies: date = start_date for every row; facilities and
  conditions are not yet checked). Outcomes and outcome_analyses are never
  before the cutoff for the labelled studies (coverage 0 %), so 1-hop aggregation sees only
  static study attributes. The signal lives **two hops away**: the sponsor's, facilities' and
  conditions' track record on other, earlier trials, i.e. the share of their earlier studies
  with a primary p-value <= 0.05 and how many trials they have run. That needs
  link-table traversal (studies <- sponsors_studies -> sponsors <- sponsors_studies -> other
  studies <- outcome_analyses), time-filtered at every hop. Val is one cutoff with 960 rows
  (AUROC SE about 0.02), so it is noisy.
- **Not yet profiled:** rel-amazon, avito, event, hm and stack (not fetched). The 9-feature
  amazon test above already reaches val 0.685 on user-churn.

## 5. Noise floor

The val sets are small: 566 (dnf), 588 (top3), 960 (study-outcome). A paired bootstrap SE of
the AUROC difference is roughly 0.01-0.02 per task. The keep rule therefore uses
the mean over tasks, and a seed repeat when a gain is under 0.01 (EXPERIMENTS.md).
