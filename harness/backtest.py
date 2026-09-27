"""Validation score and train-internal time backtests of family variants (never reads test).

    python harness/backtest.py --tasks rel-f1/driver-top3 --variants b0,b0pe,b1,b1pe \
        --seeds 0,1 [--cap 250000] [--record]

Variants are the family grid of reldfs.families.VARIANTS: b0 = base (no switchable family),
b1 = base + ft.*, and the suffixes gc / pe / gp add gc.* / pe.* / both. For each variant and
seed it prints one line

    [bt] reldfs:<variant> <task> seed=<s> val=... bt1=... bt1_refit=... bt2=... bt2_refit=...

(6 decimals; harness/famselect.py reads these lines), where each value is the task's own
metric (task.evaluate) on:

  val    fit on train, early stop on val, score val
  bt1    the train cutoffs, in time order, are split into train' | val' | test', where val'
         and test' each hold k = min(#val cutoffs, #train cutoffs // 4) cutoffs; fit on
         train', early stop on val', score test'
  bt2    train' = train minus its last k cutoffs, val' = those k cutoffs, test' = val
  *_refit the same folds, scored by a model refit on train' + val' for
         round(best_iter * (1 + n_val' / n_train')) rounds

Fitting folds are purged: a row with cutoff s is dropped when its label window
(s, s + timedelta] reaches past the first cutoff of the next fold. --cap N samples train to N
rows, stratified by label (the gating cap used for hm, stack and amazon; results.tsv notes it).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import append, cached_build, load_ctx, now, rss_gb, score  # noqa: E402


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
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--variants", default="b0")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--cap", type=int, default=0, help="stratified gating cap on train rows")
    ap.add_argument("--step", default="-")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--no-bt", action="store_true")
    args = ap.parse_args()

    import reldfs.features as F
    from reldfs.families import VARIANTS
    from reldfs.model import fit_refit, sample_train

    names = [v for v in args.variants.split(",") if v]
    seeds = [int(s) for s in args.seeds.split(",")]
    for name in args.tasks.split(","):
        t0 = time.time()
        ctx = load_ctx(name)
        task = ctx.task
        plan = F.make_plan(ctx.db, task)
        tr = sample_train(ctx.train, task.target_col, gating_cap=args.cap)
        va = ctx.val
        Xtr_all = cached_build(ctx.db, task, plan, tr, f"{name}|train")
        Xva_all = cached_build(ctx.db, task, plan, va, f"{name}|val")
        fds, k = folds(task, tr, va)
        for v in names:
            cols = [c for c in Xtr_all.columns if not c.startswith(VARIANTS[v])]
            Xtr, Xva = Xtr_all[cols], Xva_all[cols]
            for seed in seeds:
                res = {}
                r = fit_refit(task, Xtr, tr, Xva, va, None, seed, refit=False)
                res["val"] = (score(task, va, r["val"]), r["best_iter"])
                if not args.no_bt:
                    for fn, (i_tr, i_va, i_te, te_kind) in fds.items():
                        if te_kind == "val":
                            Xte, te = Xva, va
                        else:
                            Xte, te = Xtr.loc[i_te].reset_index(drop=True), tr.loc[i_te]
                        r = fit_refit(task, Xtr.loc[i_tr].reset_index(drop=True), tr.loc[i_tr].reset_index(drop=True),
                                      Xtr.loc[i_va].reset_index(drop=True), tr.loc[i_va].reset_index(drop=True),
                                      Xte, seed)
                        res[fn] = (score(task, te, r["test_norefit"]), r["best_iter"])
                        res[fn + "_refit"] = (score(task, te, r["test"]), r["best_iter"])
                line = " ".join(f"{s}={a:.6f}" for s, (a, _) in res.items())
                print(f"[bt] reldfs:{v} {name} seed={seed} {line}", flush=True)
                if args.record:
                    for s, (a, it) in res.items():
                        append([now(), args.step, f"reldfs:{v}", name, s, "roc_auc", f"{a:.4f}", "", "", "",
                                f"{time.time() - t0:.0f}", f"{rss_gb():.2f}",
                                f"backtest seed={seed} iters={it} k={k} n_tr={len(tr)} cap={args.cap}"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
