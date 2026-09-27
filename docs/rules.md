# RelBench v3 leaderboard: rules, protocol and current rows

Sources, all fetched 2026-09-24:
- `relbench` 3.0.1 from PyPI (MIT): `relbench/submit.py` (task lists, validation, packaging),
  `relbench/load.py`, `relbench/metrics.py`.
- https://github.com/stanford-star/relbench @ `584a03d` (2026-09-17): README "Leaderboard",
  `.github/ISSUE_TEMPLATE/submit.yml`, `.github/workflows/leaderboard_validate.yml`,
  `examples/lightgbm_entity.py`.
- Leaderboard page https://star-project.stanford.edu/relbench/leaderboard/ (rules text) and its
  data file `https://raw.githubusercontent.com/stanford-star/relbench/main/leaderboard/leaderboard.json`
  (snapshot: `docs/leaderboard-2026-09-24.json`).
- Data: HF dataset `stanford-star/relbench-v1` (sizes from the Hub tree API).

## Boards (verbatim from `relbench.submit.LEADERBOARD_TASKS`)

"There are three independent boards: classification (12 tasks), regression (9), and
recommendation (10) ... You can submit to any of them; each requires predictions for *all* of
its tasks." (README). The board score is the plain mean over the board's tasks.

| board | metric | tasks |
|---|---|---|
| classification | AUROC (`roc_auc`), higher is better | rel-amazon/user-churn, rel-amazon/item-churn, rel-avito/user-visits, rel-avito/user-clicks, rel-event/user-repeat, rel-event/user-ignore, rel-f1/driver-dnf, rel-f1/driver-top3, rel-hm/user-churn, rel-stack/user-engagement, rel-stack/user-badge, rel-trial/study-outcome |
| regression | NMAE = MAE / std(train target, ddof=1), lower is better; stds hosted in `regression_stds.json` | rel-amazon/user-ltv, rel-amazon/item-ltv, rel-avito/ad-ctr, rel-event/user-attendance, rel-f1/driver-position, rel-hm/item-sales, rel-stack/post-votes, rel-trial/study-adverse, rel-trial/site-success |
| recommendation | MAP@k (task's `eval_k`), higher is better | rel-amazon/user-item-purchase, rel-amazon/user-item-rate, rel-amazon/user-item-review, rel-avito/user-ad-visit, rel-f1/driver-circuit-compete, rel-hm/user-item-purchase, rel-stack/user-post-comment, rel-stack/post-post-related, rel-trial/condition-sponsor-run, rel-trial/site-sponsor-run |

**Every board contains rel-amazon tasks** (2 / 2 / 3), so no board can be completed without
rel-amazon. There is no amazon-free board definition.

## Submission protocol

1. One prediction CSV per task, `<dataset>__<task>.csv`, written by
   `relbench.submit.write_prediction_table(task, test_pred, path)`. Classification: probability in
   [0, 1]; regression: original target scale; recommendation: JSON list of top-`eval_k` dst ids.
   Key columns `[entity_col, time_col]` must biject with the test table.
2. `python -m relbench.submit <dir>` validates and scores every CSV against the hosted test
   tables, prints a verdict per board and writes one zip per complete board.
3. Open the GitHub issue form `submit.yml` (fields: Name, Note, In-context? Yes/No, URL, files).
   CI (`leaderboard_validate.yml`) "downloads the attached prediction tables, scores them against
   the test tables, and posts the report as an issue comment. Only issue *data* is consumed — no
   submitter code ever runs." A maintainer then adds the `accept` label.

So submissions are **predictions on the test split, self-reported**; no code is run or checked.
The test labels are public (hosted `tasks/<t>/test.parquet`); the evaluator reads them.

Rules text on the leaderboard page, verbatim: "The RelBench leaderboard uses self-reported
prediction files. We place no restriction on submissions (e.g. runtime, hyperparameter-tuning,
etc.) except the following: 1. For all submissions, any info at or in the future of the
prediction time is off-limits. 2. For in-context submissions, gradient-based training on the
target DB is not allowed." reldfs trains on the target DB, so its In-context field is No.

## Current rows (leaderboard.json, 20 entries, fetched 2026-09-24)

Classification (mean AUROC over 12, test):

| rank | row | mean | date | kind |
|---|---|---|---|---|
| 1 | Kapso (Leeroo Team) | 0.8119 | 2026-08-26 | LLM-driven data-science system |
| 2 | RT-PluRel (fine-tuned) | 0.7896 | 2026-08-27 | relational transformer |
| 3 | RT | 0.7776 | 2026-08-27 | relational transformer |
| 4 | RT-PluRel (in-context) | 0.7411 | 2026-08-27 | in-context |
| 5 | GNN + LightGBM | 0.7389 | 2026-08-25 | hybrid (GNN embeddings) |
| 6 | Rel-MOSS | 0.7367 | 2026-09-09 | RDL / GNN |
| 7 | LEDGER | 0.7289 | 2026-09-10 | event-sequence transformer |
| 8 | GNN (reference) | 0.7283 | 2026-08-25 | GNN |
| 9 | Entity Mean | 0.6579 | 2026-08-25 | trivial, in-context |
| 10 | **LightGBM (official baseline)** | 0.6306 | 2026-08-25 | entity-table features only |
| 11-13 | Entity Median 0.5944, Majority 0.5000, Random 0.4980 | | | |

Regression (mean NMAE over 9, lower is better): Kapso 0.2476, RT-PluRel (ft) 0.2814, RT 0.2926,
GNN 0.3130, RT-PluRel (ic) 0.3299, GNN+LightGBM 0.3319, LEDGER 0.3391, **LightGBM 0.3403**,
Global Median 0.3610, Entity Median 0.4283, Global Mean 0.4530, Entity Mean 0.4553, Zero 0.4932.

Recommendation (mean MAP): Kapso 0.1837, ID-GNN 0.0933, LEDGER 0.0875, **LightGBM 0.0759**,
Global Popularity 0.0587, Past Visit 0.0501, GNN 0.0414.

On this snapshot every row above the reference GNN on classification is neural or LLM-driven.

Per-task test rows on three of the tasks (classification AUROC):

| task | Kapso | RT-PluRel ft | GNN+LGBM | GNN | LightGBM |
|---|---|---|---|---|---|
| rel-f1/driver-dnf | 0.8327 | 0.8031 | 0.7157 | 0.7261 | 0.7255 |
| rel-f1/driver-top3 | 0.9357 | 0.9069 | 0.7248 | 0.6801 | 0.6493 |
| rel-trial/study-outcome | 0.8208 | 0.7319 | 0.6627 | 0.6693 | 0.6885 |

Full per-task table for all rows: `docs/leaderboard-2026-09-24.json`.

## Notes on the snapshot

- leaderboard.json has 19 entries dated 2026-08-25 .. 09-10 (issues #380-#403); 12 are
  maintainer baselines and 7 are third-party (Kapso, RT-PluRel x2, RT, LEDGER x2 (one a
  re-upload), Rel-MOSS).
- 31 tasks, 7 of them on rel-amazon (12 + 9 + 10 = 31; amazon 2 + 2 + 3 = 7).
- rel-amazon's review.parquet is 7,201.7 MB on the Hub; rel-stack is 1,073.9 MB in total.
- Rel-MOSS 0.7367 and LEDGER 0.7289 also sit above the GNN on classification (both neural).
- The README's table and every comparison in this repository use this snapshot. The live
  board may have changed since.
