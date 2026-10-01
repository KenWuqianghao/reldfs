"""Val grid + time backtest for a candidate's training variants (never touches test).

    python harness/backtest.py --cand c1_recent --tasks a,b,c [--variants hl4y,uniform]
        [--seeds 0,1] [--iter N] [--record]

The candidate's model.py must expose features(ctx, splits), fit_predict(...) and VARIANTS.
For each variant and seed it reports AUROC (the task's own metric, via task.evaluate on a
Table holding the fold's labels) on:

  val    train -> early stop on val -> score val (what run.py records)
  bt1    backtest that HOLDS OUT THE LAST TRAIN PERIOD: the train cutoffs are split in time
         order into train' | val' | test', where val' and test' each hold k = min(#val
         cutoffs, #train cutoffs // 4) cutoffs. Fit on train', early stop on val', score test'.
  bt2    train' = train minus its last k cutoffs, val' = those k cutoffs, test' = val.
  *_refit the same folds, but the scoring model is refit on train'+val' for
         round(best_iter * (1 + n_val'/n_train')) rounds (the run.py refit rule).

Purging mirrors the real split: a row with cutoff s is dropped from a fitting fold when its
label window (s, s + timedelta] reaches past the first cutoff of the next fold.
Test labels are never read; the test split is not loaded here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_ctx  # noqa: E402
from run import append, load_cand  # noqa: E402


def score(task, df, pred) -> float:
    from relbench.base import Table

    tb = Table(df=df.reset_index(drop=True), fkey_col_to_pkey_table={}, pkey_col=None,
               time_col=task.time_col)
    return float(next(iter(task.evaluate(np.asarray(pred), tb).values())))


def purge(df, tcol, nxt, delta):
    return df[df[tcol] + delta <= nxt]


def folds(task, tr, va):
    tc = task.time_col
    cuts = np.sort(tr[tc].unique())
    k = max(1, min(va[tc].nunique(), len(cuts) // 4))
    d = task.timedelta
    out = {}
    te_c, va_c = cuts[-k:], cuts[-2 * k:-k]
    te1 = tr[tr[tc].isin(te_c)]
    va1 = purge(tr[tr[tc].isin(va_c)], tc, te_c.min(), d)
    tr1 = purge(tr[tr[tc] < va_c.min()], tc, va1[tc].min(), d)
    out["bt1"] = (tr1.index.to_numpy(), va1.index.to_numpy(), te1.index.to_numpy(), None)
    va2 = purge(tr[tr[tc].isin(te_c)], tc, va[tc].min(), d)
    tr2 = purge(tr[tr[tc] < te_c.min()], tc, va2[tc].min(), d)
    out["bt2"] = (tr2.index.to_numpy(), va2.index.to_numpy(), None, "val")
    return out, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--variants", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--iter", default="-")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--no-bt", action="store_true")
    args = ap.parse_args()
    mod = load_cand(args.cand)
    names = [v for v in args.variants.split(",") if v] or list(mod.VARIANTS)
    seeds = [int(s) for s in args.seeds.split(",")]
    table = []
    for name in args.tasks.split(","):
        t0 = time.time()
        ctx = load_ctx(name)
        task = ctx.task
        X, fr = mod.features(ctx, splits=("train", "val"))
        Xtr, Xva, tr, va = X["train"], X["val"], fr["train"], fr["val"]
        fds, k = folds(task, tr, va)
        for v in names:
            cfg = dict(mod.VARIANTS[v], refit=True)
            for seed in seeds:
                res = {}
                o, _, it, _ = mod.fit_predict(task, Xtr, tr, Xva, va, [Xva], dict(cfg, refit=False),
                                              seed)
                res["val"] = (score(task, va, o[0]), it)
                if not args.no_bt:
                    for fn, (i_tr, i_va, i_te, te_kind) in fds.items():
                        if te_kind == "val":
                            Xte, te = Xva, va
                        else:
                            Xte, te = Xtr.loc[i_te].reset_index(drop=True), tr.loc[i_te]
                        o, r, it, _ = mod.fit_predict(
                            task, Xtr.loc[i_tr].reset_index(drop=True), tr.loc[i_tr].reset_index(drop=True),
                            Xtr.loc[i_va].reset_index(drop=True), tr.loc[i_va].reset_index(drop=True),
                            [Xte], cfg, seed)
                        res[fn] = (score(task, te, o[0]), it)
                        res[fn + "_refit"] = (score(task, te, r[0]), it)
                secs = time.time() - t0
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                line = " ".join(f"{s}={a:.6f}" for s, (a, _) in res.items())
                print(f"[bt] {args.cand}:{v} {name} seed={seed} {line}", flush=True)
                for s, (a, it) in res.items():
                    table.append(dict(task=name, variant=v, seed=seed, split=s, auc=a))
                    if args.record:
                        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                        append([now, args.iter, f"{args.cand}:{v}", name, s, "roc_auc", f"{a:.4f}",
                                "", "", "", f"{secs:.0f}", f"{rss:.2f}",
                                f"backtest seed={seed} iters={it} k={k} n_tr={len(tr)}"])
    df = pd.DataFrame(table)
    piv = df.pivot_table(index=["variant", "split"], columns="task", values="auc", aggfunc="mean")
    piv["mean"] = piv.mean(axis=1)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_columns", 20):
        print(piv.round(4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
