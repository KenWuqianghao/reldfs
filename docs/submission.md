# Submission to the RelBench classification board

Status: prepared, **not submitted**. The zip is not committed to this repository;
`make zip` (or `python harness/finaltest.py --final reldfs --package-only`) rebuilds it from
`preds/final-reldfs/`, which `make final` writes.

## What is submitted

- System: reldfs, the final system (development name c14_fam3, EXPERIMENTS.md step 7b). It
  was fixed by rule before any test number of the closing stage existed.
- Predictions: the final full-size pass (EXPERIMENTS.md step 7c). Train is capped at 1M
  rows, the model is refit on train + val, and each prediction is the mean of 2 seeds.
- Local test score, scored once with `relbench.submit.evaluate_task` (the function the board's
  CI runs; EXPERIMENTS.md step 7d): **12-task mean AUROC 0.7742**. On the 2026-09-24 snapshot,
  the reference GNN scores 0.7283, GNN + LightGBM 0.7389 and RT 0.7776.
- The zip: `submissions/reldfs-classification.zip`, 12 CSVs, 11,377,079 bytes (under GitHub's
  25 MB attachment limit), sha1 `4a6b69231a3edc2ada0729927767e19908f83fbb` as recorded. It was
  written by relbench.submit's own packager after the classification family validated 12/12
  (keys match each test table one-to-one; every probability is in [0, 1]). A rebuilt zip has
  identical members but different bytes, because the zip stores file times. It holds exactly:
  rel-amazon__user-churn.csv, rel-amazon__item-churn.csv, rel-avito__user-visits.csv,
  rel-avito__user-clicks.csv, rel-event__user-repeat.csv, rel-event__user-ignore.csv,
  rel-f1__driver-dnf.csv, rel-f1__driver-top3.csv, rel-hm__user-churn.csv,
  rel-stack__user-engagement.csv, rel-stack__user-badge.csv, rel-trial__study-outcome.csv.

## Issue-form fields

Form: https://github.com/stanford-star/relbench/issues/new?template=submit.yml

| field | value |
|---|---|
| **Name** | `reldfs (time-aware relational DFS + LightGBM)` |
| **Note** | `CPU-only, non-neural, non-LLM. Automated time-windowed aggregates over the schema's foreign-key paths (1-hop windows, 2-hop track records, grandchild and parent-context aggregates, day-level recency), cut strictly before each prediction time; LightGBM; per-task feature families chosen by a pre-registered rule on train-internal backtests; refit on train+val; 2-seed mean. Test labels are public, so this is self-reported; test was scored once, locally, for this system.` |
| **In-context?** | **No** (the model trains on the target database) |
| **URL** | `https://github.com/KenWuqianghao/reldfs` |
| **File(s)** | `submissions/reldfs-classification.zip` |

After the form is filed, CI (`leaderboard_validate.yml`) rescores the attached tables and posts
its report as an issue comment; a maintainer then adds the `accept` label.

## Method note

**reldfs** is CPU-only and non-neural. It uses no GNN, no transformer, no embeddings, no
pretrained model and no LLM, at any stage.

- **Features.** Automated deep-feature-synthesis-style aggregation over the schema graph. The
  engine is driven only by the manifest's primary keys, foreign keys and time columns, and
  by column dtypes; it has no per-dataset or per-task code path. For each label row
  (entity, prediction time t) it computes:
  - counts and numeric aggregates of the entity's timed child rows in windows of 1, 4 and
    16 x the task's timedelta, plus days since the first and last event;
  - 2-hop track records through static parents;
  - grandchild aggregates through timed intermediate tables;
  - context from the child rows' static parents;
  - day-level recency: active days in recent windows, and the gaps between active days.

  A scale-up guard bounds every path join from size estimates on the full database. When a
  path is too large, the guard buckets it by day, sheds columns or drops it.
- **Model.** LightGBM (binary, one thread, fixed default-style parameters), early-stopped on
  the validation split. Test predictions come from a refit on train + val for
  best_iter x (1 + n_val / n_train) rounds, averaged over 2 seeds. Train is capped at 1M
  sampled rows.
- **Per-task choices.** Which feature families a task uses (grandchild, parent-context,
  day-level recency) was chosen per task by a rule fixed in advance. The rule reads only
  train-internal time backtests: a family set is eligible only if it improves both backtests
  on both seeds, and the eligible set with the largest mean gain wins. It never reads
  validation or test.
- **Leak guards.**
  - Every feature uses only rows with time strictly before the label row's prediction time,
    at every hop.
  - A leak checker rebuilds features for sampled cutoffs from a database truncated at the
    cutoff and requires identical output. It is batched with later-cutoff rows to catch
    batch-level leaks, and it passes on all 12 tasks.
  - Each feature family has a planted-leak test: its time filter is removed, and the
    checker must fail.
  - Test labels are never read by the method. Test was scored once for this system (in one
    run with two supplementary systems) after it had been fixed and its CSVs hashed; the only
    other test scoring was at the start of the project, for a baseline reproduction and a
    first version on three tasks. No choice was made on a test number.
- **Cost.**
  - The final pass took about 3.5 h of single-process time on a shared 4-core CPU machine
    (LightGBM on 1 thread, duckdb on 2 threads). That time covers three systems: this one
    and two supplementary ones.
  - Peak memory was 4.2 GB per process.
  - The largest tasks took 14 to 61 minutes each; the f1, trial and event tasks 0.5 to 3.5
    minutes each.
  - No GPU was used.
- **Local test result.** Mean AUROC 0.7742 over the 12 classification tasks. It is above the
  GNN on 10 of the 12 tasks. The validation-to-test change on the mean is -0.0068.
