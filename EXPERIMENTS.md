# Experiments

The log behind the numbers in the README: every experiment in the order it was run, kept or
rejected, with its numbers. The work ran from 2026-09-24 21:55 to 2026-09-25 17:53 UTC in
eight rounds (0-7), split into steps (1a, 1b, ...). The `step` column of
[results.tsv](results.tsv) holds the round number. H1, H2, ... are the hypotheses, numbered
in the order they were proposed, not the order they were run.

Scores are AUROC from the task's own `task.evaluate` on the validation split (val) or on two
time backtests carved from the train split (bt1, bt2; defined under "Protocol"), unless
marked test. Board rows are test numbers from the 2026-09-24 snapshot, so comparing a val number
with them is indicative only. "2 seeds" means LightGBM seeds 0 and 1.

## Development names

Each experiment was a copy of the current system with one change, called a candidate. The
candidates are not in this repository: the final one is the `reldfs/` package (its feature
engine is the final candidate's, unchanged except for the duckdb spill path), and
`results.tsv` keeps the candidate names in its `system` column.

| candidate | what it is | status |
|---|---|---|
| lgbm_base | torch-free re-implementation of the official LightGBM baseline (entity-table columns only) | reference |
| c0_dfs | 1-hop time-windowed aggregates over every child table + entity and parent columns + LightGBM | first version |
| c1_recent | c0 + refit on train + val for test (H3b); recency weights tried (H3a) | kept (H3b) |
| c2_track | + 2-hop track records through static parents (H1) | kept |
| c3_labhist, c5_labhist | + the entity's own earlier labels, rebuilt from the task SQL on cut-off data (H2) | rejected twice |
| c4_det | c2 with the feature matrix cast to float32 (determinism) | base refactor |
| c6_labtrack | + other entities' earlier labels through 2 hops (H8) | rejected |
| c7_guard | c4 + the scale-up guard for path joins | base refactor |
| c8_gchild | + grandchild aggregates `gc.*` (H9) | rejected |
| c9_tie | c7 + a deterministic tie-break in the track records; the gating cap | base refactor; `base` system |
| c10_hp | c9 + per-task LightGBM search on val (H6) | rejected |
| c11_penr | c9 + parent context `pe.*` (H7) | rejected |
| c12_fine | c9 + day-level recency `ft.*` (H10) | rejected |
| c13_fam | c9 + per-task switches over {gc, pe} chosen on backtests (H13) | rejected; `h13` system |
| c14_fam3 | c9 + per-task switches over {gc, pe, ft} chosen on backtests (H13b) | **kept; final system `reldfs`** |

`c13_fam:b0pe` and similar names are variants of one candidate scored in one process
(`reldfs.families.VARIANTS`: b0 = base, b1 = base + ft, suffix gc / pe / gp = + gc / + pe /
+ both). `final:<candidate>` rows are the final full-size pass (7c) and the test run (7d).

## Protocol

Fixed before the first experiment unless a step below says otherwise.

- **Val for choices, test for reports.** Every choice (features, windows, parameters, rounds,
  per-task switches) was made on val or on train-internal backtests. Test was report-only:
  scored at the start (step 0) and once at the end (7d). No choice was made on a test number.
- **No test labels.** The local mirror's test table has no target column; only relbench's
  evaluator reads the hosted test labels.
- **Leak check before any val number.** `harness/leakcheck.py` had to PASS on every task
  before a candidate's val number was recorded, and every new feature family got a
  planted-leak test (its time filter removed; the checker must FAIL). A single-step gain above
  .05 on a task (or .03 on the mean) was treated as a suspected leak until the diff had been
  re-read and the planted-leak test had passed. That never triggered.
- **Backtests.** bt1 holds out the last train period: the train cutoffs are split in time
  order into train' | val' | test' (k = min(#val cutoffs, #train cutoffs / 4) cutoffs each for
  val' and test'); fit on train', early stop on val', score test'. bt2 fits on train minus its
  last k cutoffs, early stops on those k and scores val. `_refit` variants score a model refit
  on train' + val'. Fitting folds are purged at the label window.
- **Keep rule.** Keep a change when the mean val AUROC over the active task set improves by at
  least .005 and no task drops more than .01. If the gain is under .01, a second seed is run
  and both seeds must clear .005.
- **Schema-applicability clause** (added in step 4a). A feature family that is a structural
  no-op on some schemas (it adds no columns, or only columns constant on every split) may also
  be kept if (a) on the tasks where it is not a no-op the mean val gain is >= .005 and both
  seeds agree, (b) both refit backtests agree in sign on those tasks, (c) no task anywhere
  drops by more than .004, (d) the leak check passes on all tasks. "Both seeds agree" was read
  strictly: each seed's applicable-task mean must be >= .005. When H9 failed by .00036 on one
  seed (4a), the strict reading was confirmed (5a) rather than loosened after seeing the
  margin.
- **Gating cap** (added in step 2a). For comparisons on the large datasets (hm, stack,
  amazon), train was sampled to 250k rows, stratified by label, with a fixed sample. Final
  runs use up to 1M random rows.
- **Machine limits.** A shared 4-core CPU machine: at most 2 processes at once, LightGBM on 1
  thread (two OpenMP threads ran 18x slower under the machine's load), duckdb on 2, under
  6 GB of RAM per process.

## Summary

Val means are 2-seed means (the 5- to 12-task active set at the time); d = change against the
base of that step.

| step | hypothesis | candidate | result | verdict |
|---|---|---|---|---|
| 0 | first version | c0_dfs | 3-task val .7728 (dnf .7896, top3 .8800, study-outcome .6489; seed 0); test .7487 vs GNN .6918 on the same 3 | starting point |
| 1a | H3a recency weights | c1_recent | val at best +.002 (half-lives 4y, 16y); dnf -.010 | rejected |
| 1a | H3b refit on train + val | c1_recent | bt1_refit +.0085, bt2_refit +.0042 (both seeds positive) | **kept** (keep 1) |
| 1b | H1 2-hop track records | c2_track | 3-task val .7726 -> .7881 (+.0155; seeds +.013 / +.018); study-outcome .650 -> .696 | **kept** (keep 2) |
| 1c | H2 own label history | c3_labhist | 3-task val +.0029 | rejected |
| 1d | float32 matrix; add rel-event | c4_det | 5-task val .7896 | base refactor |
| 1e | H2 again, 5 tasks | c5_labhist | +.0022 | rejected |
| 2a | H8 other entities' labels through 2 hops | c6_labtrack | 5-task val -.0031 (user-repeat -.011) | rejected |
| 2b-2c | scale-up guard; add rel-avito | c7_guard | identical on the 5 earlier tasks; 7-task val .7585 | base refactor |
| 3a | H9 grandchild aggregates | c8_gchild | 10-task val +.0030 (seeds +.0028 / +.0032); user-clicks +.025, user-badge +.0066 | rejected (keep rule) |
| 3b | tie-break fix; add rel-hm, rel-stack | c9_tie | 10-task val .7777 | base refactor |
| 4a | H9 under the new clause | c8_gchild | applicable 6-task gain +.00464 / +.00542: (a) fails on seed 0 | rejected |
| 4b | add rel-amazon (all 12 tasks) | c9_tie | 12-task val .7730 | task set complete |
| 4c | H6 per-task LightGBM search on val | c10_hp | 12-task val -.00002 | rejected |
| 5a | H7 parent context | c11_penr | 12-task val +.00258 / +.00148; applicable 7-task +.00443 / +.00254 | rejected |
| 6a | H10 day-level recency | c12_fine | 12-task val +.00217 / +.00321; hm +.0158, amazon user-churn +.0133 | rejected |
| 6c | H13 backtest-only per-task switches over {gc, pe} | c13_fam | 12-task val +.00324 / +.00356 | rejected |
| 6d | gating-cap diagnostic (250k vs 1M) | c9_tie | hm +.0027, stack engagement +.0029 at 1M | diagnostic |
| 7b | H13b switches over {gc, pe, ft} | c14_fam3 | 12-task val .7730 -> .7796 (+.00654 / +.00648) | **kept** (keep 3) |
| 7c | final full-size pass (1M rows, 2-seed mean) | c14_fam3 | 12-task val .7810 | final |
| 7d | the test run | c14_fam3 | **12-task test .7742** (supplementary: c13_fam .7727, c9_tie .7678) | report |

## Round 0: feasibility, harness, first version

- Every board includes rel-amazon (2 / 2 / 3 tasks), and its review table is 7.2 GB on the
  Hub. Reading only the needed columns over HTTP range requests takes 24 s and gives a 123 MB
  parquet; `harness/fetch.py` does this for every table, dropping string columns of 5 MB or
  more. relbench 3.0.1 needs no torch. Details: [docs/profile.md](docs/profile.md).
- Harness: fetch, val scoring with `task.evaluate`, test scoring only through
  `relbench.submit.evaluate_task`, and the leak checker (truncated-database equality; a
  planted leak flagged 108 of 114 columns).
- lgbm_base (the official LightGBM baseline without torch and without GloVe text): test
  dnf .6751, top3 .6315, study-outcome .7188; mean .675 against the board's .688.
- c0_dfs: val dnf .7896, top3 .8800, study-outcome .6489; **test** .6994 / .8447 / .7019, mean
  .7487 (GNN .6918 on the same three). This was the first of the project's two test runs.
- Incident: a duckdb range join of every cutoff against every review spilled to disk and
  filled the shared disk; it was removed at once. Since then every join is keyed by entity
  and duckdb's spill directory is capped at 500 MB.

## Round 1: refit, track records, label history

- **1a, H3 (drift).** dnf lost .09 from val to test at step 0, with a falling label rate.
  (a) Recency weights (half-lives 2-16 years, or an 8-year window): at best +.002 on the val
  mean, and they hurt dnf itself (-.010 val at a 4-year half-life). Rejected. (b) Refit on
  train + val for round(best_iter x (1 + n_val / n_train)) rounds: bt1_refit +.0085 (seeds
  +.0107 / +.0062), bt2_refit +.0042 (+.0015 / +.0069); no task below -.004. Val cannot
  measure a model trained on val, so the backtest was the gate. **Kept.**
- **1b, H1 (2-hop track records).** study-outcome's signal lives two hops away: the sponsors',
  facilities' and conditions' record on earlier trials. Schema-generic: for every static
  parent P linked through a link table C, path A aggregates P's own timed children and path
  B P's other entities' timed children, each row entering only when both hops are before t;
  per-parent cumulative sums read with ASOF joins. f1 gets 229 columns (constructors), trial
  212. Planted leaks (ASOF hop filter removed; link-hop time removed) FAIL 6/6. 3-task val
  .7726 -> .7881 (+.0155; seeds +.013 / +.018), study-outcome .6503 -> .6964, bt1_refit
  +.0071, bt2_refit +.0153. **Kept.**
- **1c, H2 (own label history).** The task's own SQL re-run on views cut at each cutoff gives
  the entity's earlier labels (never read from the label tables). Planted leak FAIL 6/6.
  3-task val +.0029. Rejected.
- **1d (infrastructure).** rel-event added. relbench's `get_db` peaked at 8.7 GB on rel-event
  (a second probe was OOM-killed at 10 GB); a lean loader with the same values
  (`reldfs/data.py`, checked column by column on f1 and trial) peaks at 2.3 GB. duckdb's
  parallel float sums differed in the last bit between runs (94 columns on trial, about
  1e-14) and moved a LightGBM bin edge: trial val .6939 cached vs .6889 cold. Casting the
  matrix to float32 (c4_det) makes builds bit-identical. 5-task val .7896. An ordering slip:
  the c2 event val rows were recorded before that task's leak check ran; the check passed.
- **1e, H2 again** on 5 tasks (the board's trivial Entity Mean row is strong on event):
  +.0022. Rejected; the windowed child counts already carry the recent status mix.

## Round 2: other entities' labels, the scale-up guard, rel-avito

- Rules added before 2a: LightGBM on 1 thread, at most 2 processes, the 250k gating cap.
- **2a, H8.** Other entities' earlier labels, aggregated through static parents and
  self-links (a sponsor's past success rate, friends' past repeat rate). The leak checker
  gained batch mode here: features are built together with rows of the latest cutoff, which
  catches a planted leak that the per-cutoff check passed (FAIL 5/7). 5-task val -.0031
  (user-repeat -.011, on 268 val rows). Rejected.
- **2b, guard.** Without bounds, avito's AdsInfo path B would build event tables of 1.75B +
  595M + 70M rows (about 106 GB). The guard fixes per-path caps from full-database estimates
  in make_plan: link rows per label row, parents per label row, day bucketing (times rounded
  UP, so "< t" stays strict), earliest-K links per (parent, timedelta bucket), then dropping.
  Each cap is invariant to truncation at t. Planted guard leak (latest-K instead of
  earliest-K) FAIL 3/4. The features are bit-identical to c4_det's on f1, trial and event.
- **2c, rel-avito.** user-visits val .7073 and user-clicks .6541 (2 seeds); 7-task val .7585.
  user-clicks never saw the click history, which sits behind a timed intermediate
  (user <- SearchInfo <- SearchStream): that became H9.

## Round 3: grandchild aggregates, rel-hm and rel-stack

- **3a, H9 (`gc.*`).** Every child C with a primary key and every table G with a foreign key
  to C; a row counts at the later of the two hop times. Planted leaks: "lookahead" FAIL 7/9;
  "drop G time" FAIL 8/8 on stack and PASS on avito, event and trial, where G's time equals
  C's on every joined row, so it is not a leak there. Leak check PASS on 10 tasks.
  10-task val +.0030 (seeds +.0028 / +.0032), bt1_refit +.0030, bt2_refit +.0028; user-clicks
  +.025 and user-badge +.0066 on both seeds and both backtests; a structural no-op on 6 of 10
  tasks. **Rejected** on the keep rule (+.0030 < .005).
- **3b, tie-break.** The track-record "last" value picked an arbitrary parent when link times
  tied (several articles bought on one day), so two builds disagreed and the leak check
  FAILED on rel-hm: a tie, not a leak. c9_tie breaks ties by parent id and became the base.
  rel-hm (user-churn .6943 at 250k) and rel-stack (engagement .9006, badge .8760) were added;
  10-task val .7777.
- **3c.** [docs/board-plan.md](docs/board-plan.md): the path to a 12-task submission.

## Round 4: the applicability clause, rel-amazon, a LightGBM search

- **4a.** The schema-applicability clause was added (Protocol). H9 re-judged: applicable tasks
  event x2, avito x2, stack x2 (trial's 129 gc columns are constant on every split).
  Applicable-task gain +.00464 (seed 0) / +.00542 (seed 1); the 2-seed mean +.00503 clears
  .005 by .00003 only with the two event tasks, where the path is nearly empty. (b), (c) and
  (d) pass; (a) fails on seed 0. **Not kept.** Under a looser reading (2-seed mean and same
  sign) it would have passed; that reading was not adopted (5a).
- **4b, rel-amazon.** Fetch 47 s, mirror 136 MB. The guard drops path B on both tasks
  (479M and 2.6B rows unbounded; still 23M and 71M at the tightest edge cap) and keeps path A
  as counts only. Leak check PASS, planted leak FAIL 3/3. Val user-churn .6882 / .6883,
  item-churn .8112 / .8111. **12-task val .7730**, against board test means GNN .7283,
  GNN + LightGBM .7389, RT .7776.
- **4c, H6.** A fixed 288-point LightGBM grid, 12 trials per task (the default plus 11 draws),
  on the four tasks below the GNN on val. The default won on hm and engagement; the
  user-clicks winner gained +.0046 on seed 0 and lost -.0064 on seed 1 and -.013 on
  bt1_refit; badge +.0005. 12-task val -.00002. **Rejected**: the gap to the GNN is not a
  tuning gap.

## Round 5: parent context, and a diagnosis

- **5a.** The strict reading of the clause was confirmed: H9 stays rejected, since loosening a
  rule after seeing a .00003 margin would tune the rule to the result.
- **5a, H7 (`pe.*`).** A child row joined to its static parent's columns (a review with the
  product's price and category, a purchase with the article's attributes). No-op on event,
  stack and amazon item-churn. Planted leak FAIL on f1, avito, hm and amazon; PASS on trial,
  where the sponsor link date equals the study start on every row. Leak check PASS 12/12.
  12-task val +.00258 / +.00148; applicable 7-task +.00443 / +.00254: (a) fails on both seeds.
  Consistent gains on user-clicks (+.0165), amazon user-churn (+.0047) and hm (+.0013); noise
  elsewhere. **Rejected.**
- **5b, diagnosis** of the four tasks below the GNN on val. The GNN reads one more hop through
  timed intermediates (H9: clicks, badge) and fine-grained time; static-parent context helps
  clicks and amazon. Fine-grained time is the only one that applies to every task: H10.

## Round 6: day-level recency, backtest-only switches, the cap

- **6a, H10 (`ft.*`).** For every timed child: rows in the last 7 / 30 days, distinct active
  days in the last 7, 30, 1 x td and 4 x td days, all-time active days, the mean and last gap
  between active days; a day d enters only when d + 1 day <= t. Planted leak FAIL on all 7
  datasets. 12-task val +.00217 / +.00321. hm +.0158 (.6943 -> .7102, all 4 backtests
  +.012 to +.016), amazon user-churn +.0133, top3 +.0053; stack +.0001 / +.0002; event
  user-repeat -.006, user-clicks -.004. **Rejected** (+.0022 / +.0032 < .005). The gain sat on
  the customer-churn tasks, not on stack, where 5b had expected it.
- **6b, H13 pre-registration**, written before any H13 number: every family is computed on
  every schema, and the model keeps per task a subset of {gc, pe}. For each non-base
  configuration take the 4 deltas against the base on bt1_refit and bt2_refit, seeds 0 and 1;
  eligible only if all 4 are > 0; the largest mean wins; otherwise the base. The rule reads no
  val (`reldfs.families.select`; the script was committed before the grid ran). Gate: the
  combined system against c9_tie on val under the keep rule. Stated then: the families had
  already been measured on val (3a, 5a); the selection itself cannot read val.
- **6c, H13.** Chosen: gc + pe on user-clicks, gc on user-ignore and badge, pe on amazon
  user-churn, base on the other 8 (amazon item-churn had no configuration to offer). Combined leak
  check PASS 12/12. 12-task val +.00324 / +.00356; clicks .6526 -> .6832, badge +.0066, amazon
  user-churn +.0047, user-ignore -.001. **Rejected** (< .005). H10 was not in the
  pre-registered grid and was left out, as written.
- **6d, cap diagnostic.** The base at 1M train rows instead of 250k: hm +.0027, stack
  engagement +.0029 on val, about 60 % of each task's gap to the GNN's test number. It changed
  no decision; gating stayed at 250k.

## Round 7: H13b, the final pass, the test run

- **7a, decisions for the closing stage**, made before any H13b number: returns per hypothesis
  had fallen to +.002 to +.004, so H13b would be the last hypothesis. Afterwards: the final
  system chosen by rule (c14_fam3 if H13b is kept, otherwise c9_tie), one full-size pass
  (1M train cap, refit, 2 seeds), one report-only test run through relbench's evaluator for
  the final system and two supplementary systems (c13_fam and c9_tie, never chosen between
  on test), then a submission package that is prepared but not submitted.
- **7a, H13b pre-registration.** H13 with ft.* as a third switch; same rule, same script
  (unchanged, checked with git diff). Grid per task: b0 plus every non-empty subset of its
  non-no-op families. The leak check of the combined system runs after the choice and before
  any H13b val is recorded. Stated then: ft.*'s val numbers were seen in 6a, so this gate is
  slightly less clean than a fresh one.
- **7b, result.** Determinism check: the re-run b0 reproduced 6c exactly on every task except
  hm (bt1_refit .677845 vs .677977); hm's whole grid was re-run in one process
  (`results/grid/h13b-hm-rerun.bt`) and the partial run superseded. Selection
  ([results/grid/h13b.select.txt](results/grid/h13b.select.txt)):

  | task | chosen | its 4 backtest deltas | runner-up |
  |---|---|---|---|
  | f1 driver-dnf | base | none eligible | - |
  | f1 driver-top3 | ft | +.0065 +.0201 +.0055 +.0028 | ft + pe not eligible |
  | trial study-outcome | base | none eligible | - |
  | event user-repeat | base | none eligible | - |
  | event user-ignore | gc | +.0036 +.0005 +.0024 +.0048 | ft + gc not eligible |
  | avito user-visits | pe + ft | +.0028 +.0015 +.0017 +.0024 | the only eligible one |
  | avito user-clicks | gc + pe + ft | +.0238 +.0263 +.0381 +.0287 | gc + pe (+.0276 mean) |
  | hm user-churn | ft | +.0142 +.0138 +.0152 +.0166 | ft + pe (+.0146) |
  | stack user-engagement | base | none eligible | - |
  | stack user-badge | gc + ft | +.0121 +.0100 +.0067 +.0064 | gc (+.0074) |
  | amazon user-churn | pe + ft | +.0141 +.0146 +.0160 +.0157 | ft (+.0129) |
  | amazon item-churn | ft | +.0007 +.0005 +.0006 +.0006 | the only configuration |

  Leak check of c14_fam3 PASS on all 12 tasks (86 cutoffs). Gate: 12-task val .77304 ->
  .77955, **+.00654 (seed 0) / +.00648 (seed 1)**; backtests +.0065 to +.0073; worst task
  user-ignore seed 0 -.0038. **Kept** (keep 3). Of the gain, about +.0034 comes from H13's
  families and most of the rest from ft.* on hm and amazon user-churn, which 6a had already
  shown on val and on all 4 backtests. This was the second per-task-switch gate; H13 alone
  had not passed.
- **7b, checks for the final pass.** Dropping gc, pe and ft from c13_fam's matrix gives
  c9_tie's matrix exactly (values, dtypes, column order) on four tasks. A leaner fit function
  (no redundant full-matrix copies; now `reldfs.model.fit_refit`) gives bit-identical
  predictions to the candidate's own fit on top3 and on stack engagement at 300k rows. One
  lapse: a 60k-row amazon val build for that check ran next to another process for a short
  time, against the amazon-alone rule.
- **7c, final pass** (`harness/finalpass.py`): features built once per task (train sampled to
  1M rows at random), three systems x 2 seeds, CSV = 2-seed mean of the refit predictions.
  Two value-preserving memory fixes: the refit matrix became one row-major float32 array after
  the first badge run peaked at 5.02 GB (that run was stopped; the rerun reproduced its val to
  6 decimals and the fit was re-checked bit-identical), and amazon item-churn was rebuilt with
  125k-row chunks and heap trimming after its first build reached 4.57 GB (stopped before any
  row was written). Val at full size (2-seed mean prediction):

  | task | reldfs (c14_fam3) | h13 (c13_fam) | base (c9_tie) | reldfs at 250k (7b) |
  |---|---|---|---|---|
  | f1 driver-dnf | .7860 | .7860 | .7860 | .7843 |
  | f1 driver-top3 | .8855 | .8799 | .8799 | .8845 |
  | trial study-outcome | .6954 | .6954 | .6954 | .6935 |
  | event user-repeat | .7360 | .7360 | .7360 | .7360 |
  | event user-ignore | .8526 | .8526 | .8536 | .8520 |
  | avito user-visits | .7095 | .7084 | .7084 | .7092 |
  | avito user-clicks | .6872 | .6841 | .6541 | .6858 |
  | hm user-churn | .7126 | .6982 | .6982 | .7102 |
  | stack user-engagement | .9033 | .9033 | .9033 | .9006 |
  | stack user-badge | .8855 | .8857 | .8777 | .8829 |
  | amazon user-churn | .7049 | .6950 | .6901 | .7039 |
  | amazon item-churn | .8133 | .8126 | .8126 | .8117 |
  | **mean** | **.7810** | .7781 | .7746 | .7796 |

  Before the test run the CSV sets were hashed (sha1 of the 12 CSVs concatenated in file-name
  order): c14_fam3 `b6eca1002f12`, c13_fam `1ab6ca530a3e`, c9_tie `cc49c66eb08f`.
- **7d, the test run** (`harness/finaltest.py`, run once): relbench's `evaluate_submission`,
  which calls `evaluate_task` per CSV. **reldfs 12-task test .7742**; supplementary h13 .7727
  and base .7678. The per-task table is in the README. Val to test: -.0068 on the mean
  (dnf -.053, top3 -.059, visits -.040, clicks -.037, badge -.011, user-ignore -.009, hm -.003;
  user-repeat +.069, trial +.051, item-churn +.004, engagement +.003, amazon user-churn
  +.003). The supplementary systems dropped -.0055 and -.0068. The val order
  (reldfs > h13 > base) held on test; that was not used for anything. The submission zip was
  written by relbench's packager (12 CSVs, 11.4 MB) and not submitted.

## What was rejected, and what was never tried

- Rejected on the keep rule: H3a recency weights (1a), H2 own label history (1c, 1e), H8
  other entities' labels (2a), H9 grandchild aggregates (3a, 4a), H6 LightGBM search (4c), H7
  parent context (5a), H10 day-level recency (6a), H13 switches over {gc, pe} (6c). H9, H7 and
  H10 are in the final system only as H13b's per-task switches.
- Never tried: H11, a cheap substitute for the guard-dropped 2-hop paths on rel-amazon (per
  parent running statistics of the other entities' activity); H4, field-relative features
  (rank or z-score among the entities at the same cutoff), aimed at the f1 val-to-test drift.
  The regression and recommendation boards were never started.
- Left open: avito user-clicks (test .6506) and stack user-badge (.8749) are below the GNN
  (.6656, .8883); f1 dnf and top3 lose about .05 from val to test.

## Caveats

- Two per-task-switch gates (H13, H13b) reused families first measured on val; the H13b gate
  was slightly less clean because H10's val numbers had been seen (stated before the numbers,
  7a).
- rel-hm features are not bit-reproducible across processes (differences near 1e-4 AUROC).
- The final pass needed two value-preserving memory fixes (7c).
- Test was scored twice in the project: at the start, for lgbm_base and c0_dfs on three tasks
  (step 0), and once for the final systems (7d).

## results.tsv

One row per (run, task, split): `iso_time`, `step` (round 0-7), `system` (candidate or
`candidate:variant`, or `final:<candidate>`), `task`, `split` (val, test, bt1, bt1_refit,
bt2, bt2_refit), `metric`, `value` (4 decimals), the board's LightGBM, GNN and Kapso test
numbers for that task, `secs`, `peak_rss_gb` and free-text `notes` (settings, feature counts,
top features, gating cap). Rows whose notes say SUPERSEDED were replaced by a later run and are
kept for the record. The backtest lines behind the two selections, at 6 decimals, are in
`results/grid/`.
