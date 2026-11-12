# reldfs

**reldfs** is a CPU-only predictor for relational databases. It builds features by
deep-feature-synthesis-style aggregation over the foreign-key paths of a database schema, cut
strictly before each prediction time, and fits LightGBM on them. It uses no neural network,
no embeddings, no pretrained model and no LLM.

It was developed on [RelBench](https://github.com/stanford-star/relbench) v3 (relbench 3.0.1).
On the 12 tasks of the RelBench classification board it scores a **test mean AUROC of 0.7742**,
scored once, locally, with RelBench's own evaluator. On the leaderboard snapshot of 2026-09-24,
that is above the reference GNN (0.7283) and GNN + LightGBM (0.7389), and below RT (0.7776),
RT-PluRel fine-tuned (0.7896) and Kapso (0.8119). These are self-reported numbers; see
"Caveats" below. The entry was later submitted as RelBench issue #406, where the board's CI
scored it at the same .7742.

## The four entries

| entry | what it is | 12-task test mean AUROC | neural? | where |
|---|---|---|---|---|
| **v1: reldfs** | time-aware feature synthesis + LightGBM, schema-generic | **0.7742** | **no**: no neural network, no pretrained model | the `reldfs` package (this README) |
| **v2: reldfs + TabPFN** | v1 on 7 tasks; hand-written features, a text score, recency weights and TabPFN v2 on 5 | **0.7785** | yes: a pretrained tabular foundation model on 3 tasks | [round3/](round3/) |
| **v3: v2 with the full rel-f1 database** | v2's models on all 12 tasks; the rel-f1 test features use `get_db(upto_test_timestamp=False)` | **0.7953** | yes: as v2 | [round3/README.md](round3/README.md#v3-rel-f1-with-the-full-database) |
| **v4: v3 with seven tasks changed** | v3's files on 5 tasks; replayed label anchors and window features on rel-avito, badge-state and cadence features on rel-stack, rel-hm and rel-amazon, and static-column summaries (incl. the length of `users.AboutMe`) | **0.8023** | yes: as v2 | [round3/README.md](round3/README.md#v4-seven-tasks-changed) |

- v1 is the submitted entry (RelBench issue #406, CI-scored at .7742).
- v2 was developed after v1's test scores were seen. Its choices were made on validation and
  train-internal folds only, and its test number comes from one report-only run.
- v3 changes only the two rel-f1 prediction files. It was developed after v2's test scores
  were seen, and its test number also comes from one report-only run.
- v4 changes seven prediction files. It was developed after v3's test scores were seen, in
  three steps on one day. Each step was chosen on validation by a rule written down first,
  and test was scored once per step on files hashed before the run. It uses static columns
  that are state as of the dump date (the length of `users.AboutMe` on rel-stack); see the
  disclosure in round3/README.md.
- v2, v3 and v4 are **Built with PriorLabs-TabPFN**. They use the TabPFN v2 classifier weights
  (`Prior-Labs/TabPFN-v2-clf`). These are under the Prior Labs License 1.1: Apache-2.0
  plus an attribution clause that requires the line "Built with PriorLabs-TabPFN". The
  `tabpfn` package is Apache-2.0. Neither the weights nor the package is redistributed here,
  and no model is trained on TabPFN outputs. Details:
  [round3/docs/licensing.md](round3/docs/licensing.md).

## v2 (round 3, "r3"): reldfs + TabPFN on three tasks

[round3/](round3/) holds the v2 system. It was developed on 2026-10-01, after the first
submission's test scores were seen. It is the round-1 system on 7 tasks, plus changes on 5:
- hand-written point-in-time features for rel-f1 and rel-event;
- a point-in-time text score for rel-trial;
- recency weights for rel-amazon item-churn;
- **TabPFN v2, a pretrained tabular foundation model (Built with PriorLabs-TabPFN)**, as an
  extra learner on the two f1 tasks and on trial.

Every choice was made on validation and train-internal folds: 61 task x variant combinations
were tried in round 3, and none was selected on test. It passed a pre-registered keep gate,
thinly. Its one report-only test run gave a **12-task mean AUROC of .7785**, against .7742
for the package below; RT is at .7776. It is **not** non-neural, and it is not part of the
`reldfs` package. See [round3/README.md](round3/README.md) for the protocol,
the honesty statement, the logs and the reproduction steps.

## Install and use

```sh
pip install git+https://github.com/KenWuqianghao/reldfs      # or, from a checkout: pip install .
```

Python 3.10 or later, CPU only. The dependencies are relbench 3.0.1 (its core has no torch
dependency), LightGBM 4.7.0, duckdb 1.5.5, pandas, numpy and pyarrow. The exact environment
of the recorded numbers is in `harness/requirements.txt` (Python 3.11).

Data comes from a local mirror of the Hugging Face dataset `stanford-star/relbench-v1`, one
folder per dataset in RelBench's layout. `harness/fetch.py` builds it by remote column reads
(below). Then:

```sh
reldfs predict --data data --task rel-f1/driver-top3 --out preds/
# [reldfs] rel-f1/driver-top3 system=reldfs features=364 val per seed 0.884198 0.884808 seed-mean 0.885453 (15 s) -> preds/rel-f1__driver-top3.csv
```

```python
import reldfs

ctx = reldfs.load_task("data", "rel-f1/driver-top3")      # relbench db, task, train/val/test rows
res = reldfs.run_task(ctx.db, ctx.task, ctx.name, ctx.train, ctx.val, ctx.test)["reldfs"]
res["test_pred"]          # mean over seeds 0 and 1 of the refit models' test probabilities
res["val_preds"]          # per-seed validation predictions of the early-stopped models

# the feature engine alone, on any relbench Database and a task with entity_table,
# entity_col, time_col and timedelta:
plan = reldfs.make_plan(ctx.db, ctx.task)
X = reldfs.build(ctx.db, ctx.task, plan, ctx.val[[ctx.task.entity_col, ctx.task.time_col]])
```

`run_task` samples train to at most 1M rows, builds train, val and test features, fits
LightGBM with early stopping on val, refits on train + val and averages two seeds. The test
CSV is written with `relbench.submit.write_prediction_table`, the format the board expects.
Only binary classification is implemented and evaluated.

## Results: RelBench classification board, test

Mean AUROC over the 12 classification tasks, test split. The reldfs row was scored locally
with `relbench.submit.evaluate_task`, the function the board's CI runs. Every other row is
from the board's `leaderboard.json`, snapshot of 2026-09-24
([docs/leaderboard-2026-09-24.json](docs/leaderboard-2026-09-24.json)). Rows are sorted by
mean.

| row | kind | **mean** | f1 dnf | f1 top3 | trial outcome | event repeat | event ignore | avito visits | avito clicks | hm churn | stack engage | stack badge | amazon user | amazon item |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kapso (Leeroo Team) | LLM-driven data-science system | .8119 | .8327 | .9357 | .8208 | .8116 | .8889 | .6778 | .7021 | .7162 | .9151 | .8949 | .7155 | .8313 |
| RT-PluRel (fine-tuned) | relational transformer | .7896 | .8031 | .9069 | .7319 | .8113 | .8124 | .6684 | .6901 | .7036 | .9094 | .8928 | .7130 | .8318 |
| RT | relational transformer | .7776 | .7993 | .8384 | .6955 | .7897 | .8141 | .6621 | .6859 | .7024 | .9074 | .8933 | .7130 | .8296 |
| **reldfs** | time-aware feature synthesis + LightGBM | **.7742** | **.7334** | **.8266** | **.7468** | **.8048** | **.8434** | **.6696** | **.6506** | **.7096** | **.9058** | **.8749** | **.7075** | **.8168** |
| RT-PluRel (in-context) | relational transformer, in-context | .7411 | .8112 | .9030 | .6733 | .7777 | .8375 | .6203 | .5001 | .6599 | .8800 | .8315 | .6509 | .7477 |
| GNN + LightGBM | GNN embeddings + LightGBM | .7389 | .7157 | .7248 | .6627 | .7661 | .7696 | .6616 | .6545 | .6811 | .8894 | .8503 | .6834 | .8077 |
| Rel-MOSS | GNN | .7367 | .7016 | .7963 | .6705 | .7081 | .8844 | .6262 | .6558 | .6751 | .8854 | .8587 | .6227 | .7557 |
| LEDGER | event-sequence transformer | .7289 | .6787 | .7386 | .6744 | .7617 | .7119 | .6491 | .6562 | .6699 | .8902 | .8523 | .6662 | .7971 |
| GNN | GNN (reference) | .7283 | .7261 | .6801 | .6693 | .6857 | .8038 | .6029 | .6656 | .6989 | .9056 | .8883 | .6052 | .8077 |
| Entity Mean | trivial, in-context | .6579 | .4467 | .7037 | .5000 | .7204 | .8655 | .6057 | .5557 | .5890 | .7879 | .7697 | .6066 | .7437 |
| LightGBM | entity-table columns only (official baseline) | .6306 | .7255 | .6493 | .6885 | .6892 | .7732 | .5315 | .5476 | .5531 | .6274 | .6309 | .5250 | .6259 |
| Entity Median | trivial, in-context | .5944 | .4277 | .6203 | .5000 | .6857 | .7267 | .6029 | .5253 | .5249 | .6486 | .5805 | .6052 | .6853 |
| Majority | trivial | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 | .5000 |
| Random | trivial | .4980 | .4602 | .5024 | .4643 | .5162 | .5190 | .5052 | .5043 | .4980 | .5066 | .4995 | .5014 | .4992 |

Two supplementary systems were scored in the same test run. They are reported, not chosen
between: the final system was fixed before any test number existed.

| system | mean | f1 dnf | f1 top3 | trial outcome | event repeat | event ignore | avito visits | avito clicks | hm churn | stack engage | stack badge | amazon user | amazon item |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| h13 (switches over grandchild and parent-context families only) | .7727 | .7334 | .8351 | .7468 | .8048 | .8434 | .6709 | .6485 | .6949 | .9058 | .8746 | .6976 | .8160 |
| base (no per-task families) | .7678 | .7334 | .8351 | .7468 | .8048 | .8438 | .6709 | .6058 | .6949 | .9058 | .8643 | .6916 | .8160 |

- reldfs is above the GNN on 10 of the 12 tasks. It is below it on avito user-clicks (-.015)
  and stack user-badge (-.013), and level on stack user-engagement (+.0002).
- Validation mean at the final settings (1M-row train cap, 2-seed mean) was .7810; the
  validation-to-test change is -.0068. The large per-task moves are on the small test sets:
  f1 dnf -.053 and top3 -.059 (702 and 726 test rows), trial +.051 (825), event user-repeat
  +.069 (246). At these sizes the standard error of an AUROC is about 0.02-0.03.
- Per-task numbers are rounded to 4 decimals, as recorded; the means are over those values.
  Every run is a row of [results.tsv](results.tsv).

### Caveats

- **Test labels are public, so the results are self-reported.** RelBench hosts the test
  tables with their labels, and the board scores prediction files, not code. Nothing outside
  this repository can check how the predictions were made. What this repository offers
  instead: the code that produced them, the recorded runs, and a byte-identical reproduction
  check (below). In this code, test labels are read only by relbench's evaluator: the local
  mirror's test table has no target column, and `reldfs.data.load_task` asserts it.
- **Test was scored once for the final system.** One test run scored the final system and
  the two supplementary systems together, after the final system had been fixed by rule.
  Nothing was chosen, reverted or tuned on its numbers. The only other test scoring in the
  project was at the start, for a baseline reproduction and the first version on three tasks
  (EXPERIMENTS.md, step 0).
- **The 36 prediction CSVs (3 systems x 12 tasks) were hashed before scoring.** The recorded
  hashes are the sha1 of each system's 12 CSVs concatenated in file-name order: reldfs
  `b6eca1002f12`, h13 `1ab6ca530a3e`, base `cc49c66eb08f` (first 12 hex digits).
  [results/final-preds.sha256.tsv](results/final-preds.sha256.tsv) lists the per-file sha256
  of the same files; it was computed later, and the files still match the recorded prefixes.
- **The per-task families were chosen by a rule fixed in advance, on backtests only.** Which
  of three feature families each task uses (step H13b in EXPERIMENTS.md) was chosen by
  `reldfs.families.select`, written down before any number of that experiment existed. It
  reads only two time backtests carved from the train split, on two seeds, never validation
  or test. The validation gate that then accepted the change was less clean than a fresh one:
  one of the three families had been measured on validation in an earlier, rejected
  experiment. `harness/famselect.py` reproduces the recorded selection from the recorded
  backtest lines ([results/grid/](results/grid/)).
- **In-context: No.** reldfs trains on the target database.
- **Snapshot.** The comparison rows are the 2026-09-24 snapshot. The live board may differ.

## How it works

For a label row (entity e, prediction time t), every feature is an aggregate over rows with
time strictly before t, at every hop. The engine reads only the schema (primary keys, foreign
keys, time columns) and column dtypes; there is no per-dataset or per-task code path.

1. **1-hop windows.** For every table with a foreign key to the entity table: row counts over
   all time and over windows of 1, 4 and 16 x the task's timedelta, days since the last and
   first row, per numeric column mean / min / max / std / 4-timedelta mean / last value, and
   counts of the 8 most frequent values of each low-cardinality column. Plus the entity's
   own columns and its static parents' columns.
2. **2-hop track records** through static parents (tables without a time column, such as
   sponsors, constructors or articles): the parent's own timed rows, and the parent's other
   entities' timed rows, entering only when both hops are before t; reduced over the
   entity's links by mean, max and most recent.
3. **Switchable families**, computed on every schema and kept per task:
   - `gc.*` grandchild aggregates through a timed intermediate (user <- search <- ad
     impression, user <- post <- vote);
   - `pe.*` the static parent's columns joined to the entity's timed child rows (a review with
     its product's price and category);
   - `ft.*` day-level recency: active days in the last 7 / 30 days and in 1 and 4 timedeltas,
     the mean gap between active days, the last gap.
4. **Scale-up guard.** Before building, size estimates on the full database fix per-path caps
   (link rows per label row, parents per label row, day bucketing of event times, earliest-K
   links per time bucket, dropping a path still over budget). Each cap is invariant to
   truncating the database at t, so it cannot leak. On rel-amazon it drops the
   customer-product-customer paths, which would otherwise build 479M- and 2.6B-row tables.
5. **Model.** LightGBM, binary, fixed parameters (learning rate 0.03, 31 leaves, feature
   fraction 0.7, bagging 0.8, L2 1), one thread, early stopping on validation (150 rounds of
   patience). The test prediction comes from a refit on train + validation for
   round(best_iter x (1 + n_val / n_train)) rounds; the submitted value is the mean of seeds
   0 and 1. Train is sampled to at most 1M rows (RandomState(0)).
6. **Per-task families** (`reldfs/families.py`), from the backtest-only rule:

   | task | families kept | task | families kept |
   |---|---|---|---|
   | f1 driver-dnf | none | avito user-clicks | gc, pe, ft |
   | f1 driver-top3 | ft | hm user-churn | ft |
   | trial study-outcome | none | stack user-engagement | none |
   | event user-repeat | none | stack user-badge | gc, ft |
   | event user-ignore | gc | amazon user-churn | pe, ft |
   | avito user-visits | pe, ft | amazon item-churn | ft |

   A task outside these 12 gets none of the three; `harness/backtest.py` and
   `harness/famselect.py` run the same selection for a new task.

The feature matrix is float32: duckdb's parallel sums can differ in the last bit between runs,
and float32 rounding makes two builds bit-identical (a 1e-14 difference once moved a LightGBM
bin edge and a validation AUROC by .005).

## Leak guards

- **Strict cutoffs at every hop.** Every join carries `time < t`; multi-hop rows enter at the
  later of their hop times; ASOF joins read cumulative sums strictly before t; day-level
  features admit a day d only when d + 1 day <= t (all RelBench cutoffs are at midnight).
- **Leak checker** (`reldfs/leak.py`, `harness/leakcheck.py`). For sampled cutoffs, features
  are built twice: from the full database, in one batch with rows of the latest cutoff (so
  batch-level state cannot hide a leak), and from a database truncated to rows before the
  cutoff. Any difference fails the check. The final engine passed on all 12 tasks (86
  cutoffs).
- **Planted leaks.** Each family has a switch that breaks its own time filter
  (`--plant ft|pe|gc-lookahead|gc-drop_g`); the checker must then fail, and it did wherever
  the broken filter admits rows at or after the cutoff (per dataset: EXPERIMENTS.md). `tests/test_leak.py` runs the check and all four
  plants on a synthetic schema.
- **No test labels in the working data.** `harness/fetch.py` strips the target column from
  the local test table; only `relbench.submit`'s evaluator downloads the hosted labels.
- **Selection without validation or test.** Per-task families were chosen on train-internal
  backtests only; validation served as the keep gate and test only as a report.

## Cost

- **Final pass** (all 12 tasks, three systems' fits, two seeds): about 3.5 h of single-process
  time on a shared 4-core CPU machine (load 3-8), LightGBM on 1 thread and duckdb on 2; about
  3.3 h of wall time on two parallel processes. The largest tasks took 14-61 min each
  (amazon user-churn 61 min); f1, trial and event 0.5-3.5 min each. Per-task times are in
  [docs/board-plan.md](docs/board-plan.md).
- **Memory:** peak 4.23 GB per process (avito user-clicks); the amazon, hm and stack-badge
  tasks were run alone.
- **Disk:** the 7 slim mirrors take about 0.6 GB (rel-amazon 136 MB instead of 7.4 GB on the
  Hub, by dropping large text columns).
- **No GPU** at any stage. rel-f1/driver-top3 runs end to end in about 15 s.

## Limitations

- **Classification only.** The 9 regression and 10 recommendation tasks were never run;
  `reldfs.model` implements binary classification only.
- **Below the GNN on two tasks** (avito user-clicks, stack user-badge), and 0.0034 below RT on
  the mean.
- **Validation-to-test drift on f1.** dnf and top3 lose about .05 from validation to test;
  the label rate drifts over time there, and neither recency weights nor refitting fixed it.
- **Small test sets.** Five tasks have fewer than 2,000 test rows; single-task differences of
  a few hundredths are within noise.
- **The per-task families are specific to the 12 board tasks.** They were chosen per task; a
  new task needs its own backtest grid, or it runs without the three switchable families.
- **Information the guard drops.** On rel-amazon the 2-hop customer-product-customer paths are
  dropped as too large, and the product path keeps counts only. No text is used: large text
  columns are not even downloaded.
- **Reproducibility.** Bit-identical reproduction needs the pinned versions and one LightGBM
  thread. rel-hm features were not bit-reproducible across processes (differences near 1e-4
  AUROC), and the final pass needed two value-preserving memory fixes (EXPERIMENTS.md 7c).
- **Validation reuse.** Two of the three feature families were first proposed and measured on
  validation before the backtest-only switch turned them on; the switch reduces but does not
  remove that reuse.

## Reproducing the numbers

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r harness/requirements.txt ".[test,harness]"
make test                     # 21 tests on a synthetic schema and the recorded selection logs
make fetch                    # 7 slim mirrors under data/ (about 0.6 GB; RELDFS_DATA moves them)
make leakcheck                # the leak check on all 12 tasks
make select                   # the per-task selection from the recorded backtest lines
make final                    # the full-size pass: preds/final-{reldfs,h13,base}/ (about 3.5 h)
make check-hashes             # compare preds/ with results/final-preds.sha256.tsv
make zip                      # submissions/reldfs-classification.zip from preds/final-reldfs, no scoring
```

- `make final` runs `harness/finalpass.py` once per task (`RELDFS_NOCACHE_WRITE=1`, so no
  1M-row matrix is written to disk). It writes the three systems' CSVs to
  `preds/final-<system>/`.
- **Reproduction check.** On rel-f1/driver-top3, the installed package (`pip install .` in a
  fresh environment with the pinned versions) writes a CSV byte-identical to the recorded one
  (sha256 `500048f58cda16eeb450c8afd8e2703b88db46277b18a1cb8deca93809b759aa`), for the final system and both supplementary systems. The
  backtest lines of that task (`harness/backtest.py`, 4 variants, 2 seeds) match the recorded
  ones to all 6 printed decimals.
- **The submission zip is not committed.** `make zip` (`harness/finaltest.py --package-only`)
  rebuilds it from `preds/final-reldfs/` with relbench.submit's own packager and scores
  nothing. The zip stores file times, so its bytes differ from the recorded zip (sha1
  `4a6b69231a3edc2ada0729927767e19908f83fbb`, 11,377,079 bytes); its 12 members are identical.
- `harness/finaltest.py --final reldfs --supp h13,base` is the test run: it downloads the
  hosted test labels, scores all 36 CSVs with relbench's evaluator and writes the zip.
- `harness/backtest.py` produces the `[bt]` lines that `harness/famselect.py` reads; the
  recorded ones are in `results/grid/`. The gating runs used `--cap 250000` (stratified) on
  hm, stack and amazon.

## Repository layout

- `reldfs/`: the package. `features.py` (the engine), `families.py` (per-task switches and
  the selection rule), `model.py` (LightGBM fit and refit), `pipeline.py` (one task end to
  end), `data.py` (local mirror loader), `leak.py` (leak checker), `cli.py`.
- `harness/`: `fetch.py`, `leakcheck.py`, `backtest.py`, `famselect.py`, `finalpass.py`,
  `finaltest.py`, `check_hashes.py`, `common.py`, and the pinned `requirements.txt`.
- `tests/`: leak checker and planted leaks on a synthetic schema, builder determinism, the
  selection rule.
- `results.tsv`: every scored run, one row per task and split (2,144 rows; column `system`
  holds the development names, mapped in EXPERIMENTS.md). `results/grid/`: the backtest lines
  and selection outputs of H13 and H13b. `results/final-preds.sha256.tsv`: hashes of the 36
  final CSVs.
- `docs/`: [rules.md](docs/rules.md) (board rules and rows), [profile.md](docs/profile.md)
  (feasibility and the first version), [board-plan.md](docs/board-plan.md) (the submission
  procedure and its cost), [submission.md](docs/submission.md) (issue-form fields and method
  note), the leaderboard snapshot. [EXPERIMENTS.md](EXPERIMENTS.md): every experiment, kept or
  rejected.

## License

MIT ([LICENSE](LICENSE)). reldfs contains no code copied from relbench or torch_frame; it
imports relbench (MIT, Copyright (c) 2023 RelBench Team) as a dependency and calls its
loader, evaluator and submission packager. The RelBench datasets are not included; see
https://github.com/stanford-star/relbench for their terms and citation.
