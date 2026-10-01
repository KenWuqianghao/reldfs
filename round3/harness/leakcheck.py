"""Time-leakage check for a candidate's feature builder.

    python harness/leakcheck.py --cand c0_dfs --tasks rel-f1/driver-dnf,... [--k 8]

For K sampled cutoff timestamps (from train and val), the label rows at that cutoff are
featurised twice: (a) with the task db as the candidate sees it, in one batch together with
up to 50 rows of the latest cutoff (EXPERIMENTS.md round 2: catches batch-level leaks), (b) with every table
that has a time column truncated to rows with time < cutoff (the entity table is left
whole: its rows carry the vocabulary for category codes, and a row's own columns are
the dataset's static attributes, as in the official baselines). Any difference means a
feature read a row at or after the cutoff: FAIL, exit 1.

The candidate must expose features.make_plan(db, task) and features.build(db, task, plan,
lab, threads). The plan is built once from the full db (vocabulary only) and reused.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, load_ctx  # noqa: E402


def truncated(db, cutoff, keep: str):
    from relbench.base import Database

    tables = {}
    for name, tb in db.table_dict.items():
        t2 = copy.copy(tb)
        if tb.time_col is not None and name != keep:
            t2.df = tb.df[tb.df[tb.time_col] < cutoff]
        tables[name] = t2
    return Database(tables)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    path = ROOT / "candidates" / args.cand / "features.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("feat", path)
    F = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(F)

    bad = 0
    for name in args.tasks.split(","):
        ctx = load_ctx(name)
        task = ctx.task
        plan = F.make_plan(ctx.db, task)
        lab_all = pd.concat([ctx.train, ctx.val], ignore_index=True)
        ts = np.sort(lab_all[task.time_col].unique())
        rng = np.random.RandomState(0)
        pick = sorted(set(rng.choice(ts, size=min(args.k - 1, len(ts)), replace=False)) | {ts[-1]})
        for t in pick:
            lab = lab_all[lab_all[task.time_col] == t]
            if len(lab) > 300:
                lab = lab.sample(300, random_state=0)
            # (a) is built in a BATCH with rows of the latest cutoff appended (as in a real
            # multi-cutoff build), so batch-level state (e.g. a seed range set by the batch's
            # max cutoff) cannot hide a leak; only the rows at t are compared.
            comp = lab_all[lab_all[task.time_col] == ts[-1]]
            comp = comp.sample(min(len(comp), 50), random_state=1) if t != ts[-1] else comp.iloc[:0]
            a = F.build(ctx.db, task, plan, pd.concat([lab, comp], ignore_index=True), 2).iloc[:len(lab)]
            b = F.build(truncated(ctx.db, pd.Timestamp(t), task.entity_table), task, plan, lab, 2)
            same = np.isclose(a.to_numpy(), b.to_numpy(), equal_nan=True, rtol=1e-9, atol=1e-9)
            diff = [c for c, ok in zip(a.columns, same.all(axis=0)) if not ok]
            fam = {}   # step 9b: leaking columns per family prefix (e.g. dom., era.)
            for c in diff:
                k = c.split(".")[0] + "." if c.split(".")[0] in ("gc", "pe", "ft", "dom", "era", "txt", "rate") else "other"
                fam[k] = fam.get(k, 0) + 1
            status = "ok" if not diff else f"LEAK in {len(diff)} cols {fam}: {diff[:6]}"
            bad += bool(diff)
            print(f"[leakcheck] {args.cand} {name} cutoff={pd.Timestamp(t).date()} rows={len(lab)} "
                  f"feats={a.shape[1]} {status}")
    print("[leakcheck] PASS" if not bad else f"[leakcheck] FAIL ({bad} cutoffs)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
