# Third-party licences used by reldfs

Recorded 2026-10-01 (round 3, step 9b of [../README.md](../README.md)). Research use only; nothing here is redistributed
except as stated.

| component | version | licence | how it is used |
|---|---|---|---|
| relbench | 3.0.1 | MIT | data loading, the official evaluator (`task.evaluate`, `relbench.submit.evaluate_task`) |
| LightGBM | 4.7.0 | MIT | the main learner |
| duckdb, pandas, numpy, pyarrow, scikit-learn | see requirements.txt | MIT / BSD-3 / Apache-2.0 | feature builds, the round-2 trial text score |
| torch (CPU wheel) | 2.14.1+cpu | BSD-3-Clause | TabPFN runtime (round 3, optional learner) |
| tabpfn (Python package) | 9.0.0 | Apache-2.0 (wheel `LICENSE`; `NOTICE`: "TabPFN, Copyright (c) 2026 Prior Labs GmbH") | TabPFN inference code (round 3, optional learner) |
| TabPFN v2 classifier weights | HF `Prior-Labs/TabPFN-v2-clf`, file `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` (29,009,539 bytes, sha256 `cf8c519c01eaf161...5015210e49`) | **Prior Labs License 1.1** (`license_name: priorlabs-1-1`): Apache-2.0 plus one added paragraph 10, "Additional attribution" | the pretrained TabPFN model (round 3, optional learner) |

## TabPFN details

- Package licence: the tabpfn 9.0.0 wheel ships the plain Apache-2.0 text, a NOTICE and
  THIRD-PARTY-NOTICES.md. PyPI metadata: `License-Expression: Apache-2.0`.
- Weights: the v2 classifier checkpoint is hosted on Hugging Face under the Prior Labs License
  1.1 (Apache-2.0 + paragraph 10). Paragraph 10 says: if you distribute or make available the
  work, a derivative, or a product or service containing the source or weights, you must ship
  the licence and prominently display "Built with PriorLabs-TabPFN"; if the weights or their
  outputs are used to create, train, fine-tune or distil an AI model that is distributed, its
  name must begin with "TabPFN". Internal benchmarking and testing without external
  communication is not distribution.
  - Consequence for us: our prediction files are model *outputs*, not the weights. We do not
    redistribute the weights or the package; the public reldfs code only `pip install`s
    tabpfn and downloads the weights from Hugging Face. If a TabPFN-using system is
    published (leaderboard row, README), it says "Built with PriorLabs-TabPFN" in its
    documentation. No AI model is trained or distilled from TabPFN outputs (TabPFN
    predictions are only blended with LightGBM predictions), so the naming clause does not
    apply.
  - The model card states that `...finetuned-zk73skhh.ckpt` is a content-identical alias of
    `tabpfn-v2-classifier-v2_default.ckpt`, the base v2 checkpoint of the Nature (Jan 2025)
    paper, which was pretrained on synthetic data only. We cannot inspect the pretraining
    corpus ourselves; the checkpoint was published 2025-01-02.
- Newer TabPFN weights (v2.5, v2.6, v3, v3.5; the package's default is v3.5) are under
  Prior Labs' non-commercial licences (e.g. "TABPFN-3.5 License v1.0": non-commercial,
  non-production use; research and benchmarking allowed) and the package requires an
  interactive browser-based licence acceptance with a Prior Labs account before downloading
  them. That sign-up was not done. **Only the
  v2 weights are used** (`ModelVersion.V2`, explicitly).
- Download: the v2 checkpoint came from huggingface.co
  (ungated repository, no token), cached in `data/tabpfn/` (gitignored). Install:
  `pip --python .venv/bin/python install --no-cache-dir --extra-index-url
  https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu" tabpfn==9.0.0`, with every
  already-installed package pinned by a constraints file (no existing package changed).
- No telemetry code is present in tabpfn 9.0.0 (grep for telemetry / posthog: none).

## Leaderboard flags

- Board rules ([../../docs/rules.md](../../docs/rules.md)): "1. ... any info at or in the future of the prediction time is
  off-limits. 2. For in-context submissions, gradient-based training on the target DB is not
  allowed." There is no rule against pretrained models.
- "In-context?" on the issue form describes whether the submission trains on the target DB.
  reldfs trains LightGBM (gradient boosting) on each target DB, so the answer stays **No**,
  whether or not TabPFN is part of the system. TabPFN itself is used in context (it is never
  trained or fine-tuned on RelBench data; it conditions on our point-in-time feature rows),
  and its inputs obey rule 1 like every other feature.
- Disclosure: if a TabPFN-using system is submitted, its note says that it uses a
  pretrained tabular foundation model (TabPFN v2, Prior Labs) as an additional learner on
  the named tasks, i.e. the row is no longer "non-neural". See [../README.md](../README.md) and the top-level README.
