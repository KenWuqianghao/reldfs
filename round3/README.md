# reldfs round 3 ("r3"): the third-round system

This folder reproduces the system of the third development round (2026-10-01). It is **not**
the `reldfs` package of the repository root. The package is the round-1 system: CPU-only,
schema-generic, no pretrained model. Round 3 changes five of the twelve classification tasks
and adds two things the package does not have:

- **Hand-written, per-dataset features** (`candidates/c16_r3/domain.py`), for rel-f1
  (`dom.*`, `era.*`), rel-event (`dom.*`) and rel-avito (`dom.*`, `rate.*`, not selected).
- **A pretrained tabular foundation model, TabPFN v2** (Prior Labs), as an extra learner on
  rel-f1 and rel-trial. **Built with PriorLabs-TabPFN.** Licences: [docs/licensing.md](docs/licensing.md).
  TabPFN is never trained or fine-tuned on RelBench data. It conditions in context on the
  same point-in-time feature rows LightGBM uses.

So the round-3 system is **no longer non-neural**. In-context on the board's form is still
**No**, because LightGBM trains on each target database.

## Honesty statement

- **Rounds 2 and 3 were opened after the first submission's test scores were seen**: the
  local report-only run, and the maintainers' CI on RelBench issue #406, 12-task test .7742.
  The tasks worked on were picked from the public board gaps.
- **Every choice was made on validation and train-internal rolling folds only.** The rule:
  per task, the variant with the largest mean gain over the rolling refit folds, among
  variants that gain at least .005 on those folds, win at least 2/3 of them and do not lose
  on validation. A combined keep gate then requires a 12-task validation gain of at least
  .005 on each of seeds 0 and 1.
- **Every hypothesis was written down before its numbers**: steps 8b-8i and 9b-9g of the
  [development log](#development-log-steps-8a-9j) below.
- **Test was scored once for the round-3 system, after the gate passed and the CSVs were
  hashed.**
- **The gate pass is thin.** The 12-task validation gain is +.00622 (seed 0) and +.00509
  (seed 1); seed 1 clears +.005 by .00009. It follows a long sequential search: 14
  hypothesis blocks over rounds 2 and 3, and 61 task x variant combinations in round 3
  alone. Two of the final choices were decided by fold-score margins of 2e-5 and 6e-5. The
  selected validation gains are therefore optimistic.

## Per-task system

| task | round-3 choice | what it is |
|---|---|---|
| rel-f1/driver-top3 | `base+domonly+era+noid+robust+pfn` | TabPFN v2 alone, on the top 32 (by LightGBM gain) of the compact f1 matrix: dom.* + era.* + the driver's own columns without name identifiers |
| rel-f1/driver-dnf | `D~D+pfnk64`, D = `base+domonly+era+noid+robust` | rank-average of LightGBM on the compact f1 matrix (ROBUST parameters) and TabPFN v2 on its top 64 columns |
| rel-trial/study-outcome | `base+txt\|base+txt+robust~base+txt+pfnk64` | rank-average of (the mean of two LightGBM settings on the engine + a point-in-time TF-IDF text score) and TabPFN v2 on the top 64 of those columns |
| rel-event/user-ignore | `base+dom+noid+robust` | LightGBM on the engine + the hand-written event family, ROBUST parameters (round 2) |
| rel-amazon/item-churn | `base+hlq16` | the round-1 system with recency weights, halflife = the fitting set's cutoff span / 16 |
| the other 7 tasks | `base` | the round-1 system's CSVs, unchanged |

Variant grammar: `harness/rollbt.py`, `candidates/c16_r3/model.py` (`variant_cfg`). `|` is
the mean of probabilities; `~` is a rank average per split.

TabPFN settings:
- weights `Prior-Labs/TabPFN-v2-clf`, `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, the base
  v2 checkpoint;
- 4 ensemble members, CPU, 1 thread;
- context: at most 3000 fitting rows, sampled with the seed;
- columns: the top K by the gain importance of a fixed 300-round LightGBM on the fitting rows
  only.

Final predictions:
- 5 seeds, refit on train + validation;
- groups pooled over seeds, then rank-averaged;
- amazon item-churn at the 1M-row train cap.

## Results

Validation is the gating split: LightGBM early-stopped on val, and TabPFN never sees val
labels. It is a seed mean: 3 seeds on f1 / trial / event, 2 on amazon at the 250k gating cap.
Test is the **one** report-only run, done with `relbench.submit.evaluate_task`, the function
CI runs. Both columns are AUROC.

| task | val, round 1 | val, r3 | test, round 1 (submitted, #406) | **test, r3** |
|---|---|---|---|---|
| rel-f1/driver-dnf | .7865 | .7961 | .7334 | **.7031** |
| rel-f1/driver-top3 | .8847 | .9098 | .8266 | **.8570** |
| rel-trial/study-outcome | .6939 | .7135 | .7468 | **.7657** |
| rel-event/user-ignore | .8518 | .8546 | .8434 | **.8670** |
| rel-amazon/item-churn | .8117 | .8196 | .8168 | **.8260** |
| 7 other tasks | unchanged | unchanged | (same CSVs) | (same CSVs) |
| **12-task mean** | gate: +.00622 / +.00509 (seeds 0 / 1) | | **.7742** | **.7785** |

- **Board position.** .7785 is above RT (.7776) and InRTL (.7772), and below RT-PluRel
  fine-tuned (.7896) and Kapso (.8119).
- **The margin over RT (+.0009) is far inside test noise.** The f1 test sets have 700-730
  rows, so the standard error of an AUROC there is about .02.
- **f1 driver-dnf lost .030 on test** although its validation rose. Every dnf variant of
  rounds 2 and 3 lost on the most recent train-internal fold.
- **The CSVs were fixed before the test run.** The 12 CSVs were hashed before the run (sha1
  of their concatenation in file-name order: `45447fdd631b`). The zip holds 12 CSVs,
  11,363,337 bytes, sha1 `0c4efda7d54af5e1ffd20588d11a4727bafb1849`. It has not been
  submitted.

## Reproduce

```sh
cd round3
pip install --no-cache-dir -r ../harness/requirements.txt
pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
ln -s ../data data              # the slim mirrors from `make fetch` (or: export RELDFS_DATA=<mirrors>)
make text                       # rel-trial registration text (~85 MB)
make select gate                # the recorded per-task choices and the combined gate, from the logs
make leakcheck                  # leak check of the 5 changed tasks
make final copy-unchanged       # preds/final-r3/ (needs ../preds/final-reldfs from `make final` at the root)
make hashes zip
```

- **Recorded logs:** [results/logs/](results/logs/).
  - `r2/`: round 2 (steps 8b-8j).
  - `r3/`: round 3 (steps 9b-9i). This includes every rolling-fold grid, the leak checks, and the
    planted-leak failures of the `era.*` and `rate.*` families.
  - Each `[rb]` line holds one variant and seed: validation, then every rolling fold.
- **Selection output:** [results/select_all.txt](results/select_all.txt).
- **Gate output:** [results/gate.txt](results/gate.txt).
- **Final-pass logs:** [results/final-logs/](results/final-logs/).
- **CSV hashes:** [results/final-r3.sha256.tsv](results/final-r3.sha256.tsv).

TabPFN and duckdb are deterministic for a fixed seed on the machine used. Bit-identical
reproduction across machines is not guaranteed: hm features already differed around 1e-4
AUROC across processes in round 1.

## Development log: steps 8a-9j

Step numbers continue those of [../EXPERIMENTS.md](../EXPERIMENTS.md), whose rounds 0-7 (steps
1a-7e) all belong to the first submission. Steps 8a-8j are round 2 and 9a-9j are round 3.
Code comments and log file names cite these steps (e.g. `rb8d_*.log` holds the step-8d grid,
`rb96_*` / `rb97_*` the H9.6 / H9.7 grids). Each pre-registration below was written down
before the first number of its hypothesis existed. In the tables, R is the mean refit-fold
gain over `base`.

### Round 2 (steps 8a-8j): hand-written features, a text score; gate failed

- **8a, scope.** Round 2 was opened after the first submission's test scores were seen (local
  run, and CI on issue #406: .7742). Tasks were picked from the public board gaps. Baseline:
  the round-1 system `c14_fam3` (12-task val .7796 at the 250k gating cap). Hand-written
  per-dataset features were allowed for this round, in a separate, labelled module
  (`domain.py`). Tasks whose system does not change keep the round-1 CSVs.
- **8b, protocol (pre-registered before any round-2 number).**
  - Candidate `c15_dom` = `c14_fam3` + the f1 `dom.*` family (appended after every round-1
    column). Model options: `noid` (drop identifier-like entity string columns: on f1,
    driverRef, forename and surname), `robust` (learning rate .02, min_data_in_leaf 50,
    feature_fraction .5, lambda_l2 5). `harness/rollbt.py` scores val plus rolling refit
    folds inside train (bt1, r2-r4, bt2).
  - **Selection rule** (per task, never test). Fold scores are every `*_refit` fold, each
    averaged over seeds; R(v) is their mean. A variant is eligible iff (i) R(v) - R(base) >=
    +.005, (ii) it beats base on at least 2/3 of the folds, and (iii) its seed-mean val is not
    below base's. The eligible variant with the largest R gain is chosen; if none, base.
    Seeds 0-2 on f1, trial and event; 0-1 elsewhere (hm, stack and amazon: folds bt1 and
    bt2, 250k gating cap).
  - **Combined gate** (the round-1 keep rule, unchanged): the combined system against
    `c14_fam3` on the 12-task val mean, gain >= .005 on each of seeds 0 and 1, and no task
    below -.01. A leak check of every changed task and a planted-leak FAIL for each new
    family come before any val number of that family.
  - Final CSVs of changed tasks: 5 seeds (0-4), refit on train + val. Test: one report-only
    run at the end, after the CSVs are hashed, and only if the gate passes.
  - f1 `dom.*`: leak check PASS; planted leak FAIL on all 16 cutoffs.
- **8c, f1 grid.** top3: `base+dom+noid+robust` chosen (R +.0230); dnf: no variant eligible.
- **8d, pre-registered: compact f1 model** (`domonly`: only `dom.*` and the driver's own
  columns). Last f1 grid of the round.
- **8e, result.** top3: `base+domonly+noid+robust` (R +.0341; 3-seed val .8847 -> .9083).
  dnf: still base (best R +.0045, below .005; the most recent fold loses .016).
- **8f, rel-event and rel-avito `dom.*`.** Leak checks PASS, planted leaks FAIL. user-ignore:
  `base+dom+noid+robust` (R +.0068); user-repeat: base. avito: no gain over `gc.*`.
- **8g, pre-registered: `robust` alone** on the tasks without a `dom` family.
- **8h, pre-registered: settings blends** `X|X+robust` (mean of the two LightGBM settings).
- **8i, pre-registered: rel-trial text score** `txt.score`. TF-IDF (word 1-2 grams, min_df
  5, sublinear tf, at most 200k features) + logistic regression (C = 1) on title, official
  title and brief summary. It is point-in-time: a row at cutoff s gets a model fit only on
  rows whose label was known by s. No val label is used for a val score, and no test label
  ever. Also the status: `robust` and blends were worth +.002 to +.005 on small tasks and
  nothing on large ones, and none was eligible.
- **8j, result: combined gate FAIL.** Choices: top3 `base+domonly+noid+robust`, user-ignore
  `base+dom+noid+robust`, trial `base+txt|base+txt+robust` (R +.0173). 12-task val gain
  +.00428 (seed 0) / +.00335 (seed 1), below +.005. Round 2 was not kept: no final pass, and
  test was not looked at.

### Round 3 (steps 9a-9j): TabPFN, an f1 era family, recency weights; gate passed

- **9a, scope.** Round 3 was opened after the first submission's test scores were seen.
  Tasks were again picked from the public board gaps. The 8b selection rule and the combined
  gate are unchanged, and round 2's logs stay in the selection union. Every hypothesis is
  listed, including the failed ones.
- **9b, protocol (pre-registered before any round-3 number).**
  - Candidate `c16_r3` = `c15_dom` + the f1 `era.*` family (own prefix, so every round-2
    variant keeps its matrix).
  - Learner `pfn` = TabPFN v2: the top K = 32 columns by the gain of a fixed 300-round
    LightGBM on the fitting rows only, a context of at most 3000 fitting rows, 4 ensemble
    members, CPU. On folds and in the final pass it is fit on train + val, like the LightGBM
    refit.
  - Recency tokens `hl<N>y` / `win<N>y` and `hlq<k>` / `winq<k>`. `a~b` is a rank-average
    blend.
  - `era.*`: leak check PASS on f1; planted leak FAIL on all 16 cutoffs. TabPFN reads the
    same point-in-time matrix and adds no feature.
  - Hypotheses H9.1-H9.5 (X = the task's round-2 choice, base where none):
    - H9.1 `X+pfn`;
    - H9.2 `X~X+pfn`;
    - H9.3 dnf `era`;
    - H9.4 dnf recency;
    - H9.5 `X+hlq4` and `X+winq2` on every task.
  - Later hypotheses only in their own pre-registered step.
  - Train cap of the final pass: lifted on a large task only if full-size val is not lower
    and peak memory stays under 6 GB.
- **9c, pre-registered: H9.6.** The f1 combination D = `base+domonly+era+noid+robust`, with
  D, `D+pfn` and `D~D+pfn`.
- **9e, pre-registered: H9.7.** TabPFN on the top 64 columns (`pfnk64`) for f1 and trial.
- **9f, pre-registered: H9.8.** Stronger recency on rel-amazon (`hlq8`, `hlq16`, `winq4`).
  This was chosen after seeing H9.5's amazon numbers.
- **9g, pre-registered: H9.9.** The rel-avito `rate.*` family (smoothed group click-through
  rates before t). Leak check and planted-leak FAIL came before any number.
- **9h, result: combined gate PASS, thin.** 61 task x variant combinations were tried in
  round 3; [results/select_all.txt](results/select_all.txt) holds them all.

  | hypothesis | outcome |
  |---|---|
  | H9.1 TabPFN alone | eligible on dnf, top3, trial; repeat and ignore lose val |
  | H9.2 rank blend | eligible on dnf, top3, trial; not on repeat, ignore |
  | H9.3 era | dnf `base+era` no; compact D eligible (R +.0073) |
  | H9.4 dnf recency | all R < 0 |
  | H9.5 recency on every task | no gain beyond amazon's +.0035 (below .005) |
  | H9.6 f1 combination | top3 `D+pfn` chosen (R +.0370); dnf `D~D+pfn` R +.0092 |
  | H9.7 64 columns | dnf and trial chosen by R margins of 2e-5 and 6e-5; top3 unchanged |
  | H9.8 stronger recency | amazon item-churn `base+hlq16` chosen (R +.0070); user-churn no |
  | H9.9 avito `rate.*` | all R < 0 |
  | train cap | amazon item-churn keeps 1M (2.5M rows would exceed the memory budget) |

  Gate: +.00622 (seed 0) / +.00509 (seed 1); worst task user-ignore seed 1 -.0034
  ([results/gate.txt](results/gate.txt)).
- **9i, final pass.** The changed tasks' leak check passed with the final code. The 5 changed
  tasks were run with 5 seeds, refit on train + val. The 7 others are byte copies of the
  round-1 CSVs. The 12 CSVs were hashed before the test run.
- **9j, the one report-only test run.** Its results are in the table under "Results" above.
  Nothing was chosen, reverted or tuned on it. The zip was rebuilt and not submitted.
