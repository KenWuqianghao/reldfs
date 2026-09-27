"""reldfs command line.

    reldfs predict --data DATA_DIR --task rel-f1/driver-top3 --out preds/ [--system reldfs]
        [--seeds 0,1] [--max-train 1000000] [--chunk 250000]

Loads the task from a local mirror (harness/fetch.py), builds the features, fits the system
and writes the test prediction CSV with relbench.submit.write_prediction_table to
OUT/<dataset>__<task>.csv. It prints the validation AUROC of each seed and of the seed mean.
Test labels are never read.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def val_score(task, va, pred) -> float:
    from relbench.base import Table

    tb = Table(df=va.reset_index(drop=True), fkey_col_to_pkey_table={}, pkey_col=None,
               time_col=task.time_col)
    return float(next(iter(task.evaluate(np.asarray(pred), tb).values())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="reldfs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="fit a system on one task and write its test CSV")
    p.add_argument("--data", required=True, help="directory holding the local mirrors")
    p.add_argument("--task", required=True, help="e.g. rel-f1/driver-top3")
    p.add_argument("--out", required=True, help="output directory for the CSV")
    p.add_argument("--system", default="reldfs", help="reldfs (final), h13 or base")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--chunk", type=int, default=None)
    args = ap.parse_args(argv)

    from relbench.submit import write_prediction_table

    from .data import load_task
    from .model import MAX_TRAIN
    from .pipeline import CHUNK, run_task

    t0 = time.time()
    ctx = load_task(args.data, args.task)
    res = run_task(ctx.db, ctx.task, args.task, ctx.train, ctx.val, ctx.test, systems=[args.system],
                   seeds=[int(s) for s in args.seeds.split(",")],
                   max_train=args.max_train or MAX_TRAIN, chunk=args.chunk or CHUNK)[args.system]
    vals = " ".join(f"{val_score(ctx.task, ctx.val, v):.6f}" for v in res["val_preds"])
    out = Path(args.out) / f"{args.task.replace('/', '__')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_prediction_table(ctx.task, res["test_pred"], out)
    print(f"[reldfs] {args.task} system={args.system} features={len(res['cols'])} "
          f"val per seed {vals} seed-mean {val_score(ctx.task, ctx.val, res['val_mean_pred']):.6f} "
          f"({time.time() - t0:.0f} s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
