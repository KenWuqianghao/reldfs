"""Round-2 grid: val + train-internal rolling backtests for a candidate's variant_cfg grid.

    python harness/rollbt.py --cand c15_dom --task rel-f1/driver-dnf \
        --variants base,base+dom,base+noid,base+dom+noid --seeds 0,1,2 [--blocks 4]

Never touches test. Per variant and seed it prints one line
  [rb] <cand>:<variant> <task> seed=<s> val=... bt1_refit=... bt2_refit=... r2=... r3=... r4=...
  val        train -> early stop on val -> score val (what run.py records)
  bt2_refit  harness/backtest.py's bt2 fold: fit on train minus its last k cutoffs, early stop
             on those k cutoffs, refit on both, score val (val is never used for stopping)
  bt1_refit  = r1: the last k train cutoffs held out (backtest.py's bt1 fold)
  r<j>_refit rolling block j: test' = train cutoffs [-j k, -(j-1) k), val' = the k cutoffs
             before it, train' = everything before val' (purged as in backtest.py); scored
             with the refit on train'+val'. A block is used only if train' holds >= k cutoffs.
  k = min(#val cutoffs, #train cutoffs // 4), as in backtest.py.
Variants: "a|b" (8h) = mean of the sub-variants' probabilities; "a~b" (9b) = rank-average of
the groups (each group may itself be an "|" blend).
Purging: a fitting row with cutoff s is dropped when s + timedelta passes the next fold's
first cutoff. Pre-registered in round3/README.md, step 8b.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import purge, score  # noqa: E402
from common import load_ctx  # noqa: E402
from run import load_cand  # noqa: E402


MEMO: dict = {}   # (sub-variant, seed, fold) -> predictions, within one process


def combine(preds):
    """One group: its predictions unchanged. Several ("~"): the mean of per-split ranks scaled to [0, 1]."""
    if len(preds) == 1:
        return preds[0]
    from scipy.stats import rankdata
    return np.mean([(rankdata(p) - 1) / max(1, len(p) - 1) for p in preds], axis=0)


def roll_folds(task, tr, va, blocks: int):
    tc = task.time_col
    cuts = np.sort(tr[tc].unique())
    k = max(1, min(va[tc].nunique(), len(cuts) // 4))
    d = task.timedelta
    out = {}
    for j in range(1, blocks + 1):
        hi = len(cuts) - (j - 1) * k
        te_c, va_c = cuts[hi - k:hi], cuts[hi - 2 * k:hi - k]
        if hi - 2 * k < k:
            break
        te = tr[tr[tc].isin(te_c)]
        v = purge(tr[tr[tc].isin(va_c)], tc, te_c.min(), d)
        t = purge(tr[tr[tc] < va_c.min()], tc, v[tc].min(), d)
        out[f"r{j}" if j > 1 else "bt1"] = (t.index.to_numpy(), v.index.to_numpy(), te.index.to_numpy(), None)
    te_c = cuts[-k:]
    v2 = purge(tr[tr[tc].isin(te_c)], tc, va[tc].min(), d)
    t2 = purge(tr[tr[tc] < te_c.min()], tc, v2[tc].min(), d)
    out["bt2"] = (t2.index.to_numpy(), v2.index.to_numpy(), None, "val")
    return out, k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--no-bt", action="store_true")
    args = ap.parse_args()
    mod = load_cand(args.cand)
    name = args.task
    t0 = time.time()
    ctx = load_ctx(name)
    task = ctx.task
    X, fr = mod.features(ctx, splits=("train", "val"))
    Xtr, Xva, tr, va = X["train"], X["val"], fr["train"], fr["val"]
    fds, k = roll_folds(task, tr, va, args.blocks)
    print(f"[rb] {name} k={k} folds={list(fds)} n_train={len(tr)} build {time.time() - t0:.0f}s", flush=True)
    for v in args.variants.split(","):
        # step 9b: "a~b" = rank-average blend of groups; inside a group, step 8h's
        # "a|b" = the mean of the sub-variants' predictions. Without "~" this is 8h's code path.
        groups = [[(sv, mod.variant_cfg(name, sv)) for sv in g.split("|")] for g in v.split("~")]
        for seed in [int(s) for s in args.seeds.split(",")]:
            res = {}
            pv = []
            for cfgs in groups:
                pr = []
                for sv, c in cfgs:
                    key = (sv, seed, "val")   # 9b: deterministic fits are shared between variants
                    if key not in MEMO:
                        MEMO[key] = mod.fit_predict(task, Xtr, tr, Xva, va, [Xva], dict(c, refit=False), seed)[0][0]
                    pr.append(MEMO[key])
                pv.append(np.mean(pr, axis=0))
            res["val"] = score(task, va, combine(pv))
            if not args.no_bt:
                for fn, (i_tr, i_va, i_te, kind) in fds.items():
                    if kind == "val":
                        Xte, te = Xva, va
                    else:
                        Xte, te = Xtr.loc[i_te].reset_index(drop=True), tr.loc[i_te]
                    pg = []
                    for cfgs in groups:
                        pr = []
                        for sv, c in cfgs:
                            key = (sv, seed, fn)
                            if key not in MEMO:
                                _, r, _, _ = mod.fit_predict(
                                    task, Xtr.loc[i_tr].reset_index(drop=True), tr.loc[i_tr].reset_index(drop=True),
                                    Xtr.loc[i_va].reset_index(drop=True), tr.loc[i_va].reset_index(drop=True),
                                    [Xte], dict(c, refit=True), seed)
                                MEMO[key] = r[0]
                            pr.append(MEMO[key])
                        pg.append(np.mean(pr, axis=0))
                    res[fn + "_refit"] = score(task, te, combine(pg))
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(f"[rb] {args.cand}:{v} {name} seed={seed} " + " ".join(f"{a}={b:.6f}" for a, b in res.items())
                  + f" ({time.time() - t0:.0f}s rss {rss:.2f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
