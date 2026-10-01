"""Round-2 final full-size pass (round3/README.md, step 8b): one task, the round-2 system.

    RELDFS_NOCACHE_WRITE=1 python harness/finalpass2.py --task rel-f1/driver-top3 \
        --cand c15_dom --system r2 --seeds 0,1,2,3,4 --iter 8

Same protocol as harness/finalpass.py (round 1): train sampled to MAX_TRAIN = 1M rows with
RandomState(0), val and test featurised in chunks, LightGBM early-stopped on val, refit on
train+val for round(best_iter x (1 + n_val / n_train)) rounds (finalpass.lean_fit's steps,
with the task's parameter override). The task's configuration is the candidate's
round-2 choice: dict(CFG, drop_prefix=task_drop(name), **TASK_CFG[name]). The test CSV is the
mean of the refit probabilities over ALL seeds given (5 in round 2, bagging). Val is recorded
per seed, for the seeds-0-1 mean prediction and for the all-seed mean prediction.
Test is NOT scored here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from backtest import score  # noqa: E402
from common import board_rows, load_ctx  # noqa: E402
from finalpass import rss, trim  # noqa: E402
from run import append, load_cand  # noqa: E402


def lean_fit2(mod, task, cfg, Xtr, tr, Xva, va, Xte, seed):
    y = task.target_col
    p = dict(mod.PARAMS, **(cfg.get("params") or {}))
    p.update(objective="binary", metric="auc", num_threads=mod.LGB_THREADS, seed=seed)
    assert cfg.get("refit") and cfg.get("learner") in (None, "lgb")
    # round 3: recency weights as in the candidate's fit_predict (mod.weights); rows of weight 0
    # (windows) are not supported here (asserted), uniform weights reproduce round 2 exactly
    w = mod.weights(tr[task.time_col], cfg)
    assert (w > 0).all()
    dtr = lgb.Dataset(Xtr, tr[y].to_numpy(), weight=w)
    dva = lgb.Dataset(Xva, va[y].to_numpy(), reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    it = max(1, m.best_iteration)
    pv = m.predict(Xva, num_iteration=it)
    del m, dtr, dva
    gc.collect()
    scale = 1.0 + len(va) / max(1, len(tr))
    Xall = np.empty((len(Xtr) + len(Xva), Xtr.shape[1]), dtype=np.float32, order="C")
    for j, c in enumerate(Xtr.columns):
        Xall[:len(Xtr), j] = Xtr[c].to_numpy()
        Xall[len(Xtr):, j] = Xva[c].to_numpy()
    yall = np.concatenate([tr[y].to_numpy(), va[y].to_numpy()])
    w2 = mod.weights(pd.concat([tr[task.time_col], va[task.time_col]], ignore_index=True), cfg)
    assert (w2 > 0).all()
    d2 = lgb.Dataset(Xall, yall, weight=w2, params=p)
    d2.construct()
    del Xall
    gc.collect()
    m2 = lgb.train(p, d2, num_boost_round=max(1, int(round(it * scale))))
    pt = m2.predict(Xte)
    del m2, d2
    gc.collect()
    return pv, pt, it


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cand", default="c15_dom")
    ap.add_argument("--system", default="r2")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--iter", default="8")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--variant", default="", help="round 3: cfg = mod.variant_cfg(task, variant)")
    args = ap.parse_args()
    from relbench.submit import write_prediction_table

    mod = load_cand(args.cand)
    F = mod.F
    name = args.task
    cfg = (mod.variant_cfg(name, args.variant) if args.variant
           else dict(mod.CFG, drop_prefix=mod.task_drop(name), **mod.TASK_CFG.get(name, {})))
    t0 = time.time()
    ctx = load_ctx(name)
    task = ctx.task
    mod.ID_COLS[:] = mod.id_cols(ctx.db, task)
    plan = F.make_plan(ctx.db, task)
    tr = ctx.train
    if len(tr) > mod.MAX_TRAIN:
        tr = tr.iloc[np.sort(np.random.RandomState(0).permutation(len(tr))[:mod.MAX_TRAIN])].reset_index(drop=True)
    va, te = ctx.val, ctx.test
    keep = None

    def build(lab):
        nonlocal keep
        parts = []
        for i in range(0, len(lab), args.chunk):
            X = F.build(ctx.db, task, plan, lab.iloc[i:i + args.chunk].reset_index(drop=True), 2)
            if keep is None:
                keep = mod.select_cols(X.columns, cfg)
            parts.append(X[keep])
            del X
            trim()
        out = pd.concat(parts, ignore_index=True)
        del parts
        trim()
        return out

    Xtr, Xva, Xte = build(tr), build(va), build(te)
    print(f"[final2] {name} cfg={ {k: v for k, v in cfg.items() if k != 'refit'} } train {Xtr.shape} val {Xva.shape} "
          f"test {Xte.shape} {time.time() - t0:.0f}s rss {rss():.2f}", flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    vp, tp, its = [], [], []
    for seed in seeds:
        t1 = time.time()
        pv, pt, it = lean_fit2(mod, task, cfg, Xtr, tr, Xva, va, Xte, seed)
        vp.append(pv)
        tp.append(pt)
        its.append(it)
        print(f"[final2] {name} seed={seed} val={score(task, va, pv):.6f} iters={it} fit {time.time() - t1:.0f}s "
              f"rss {rss():.2f}", flush=True)
    v2 = score(task, va, np.mean(vp[:2], axis=0))
    vall = score(task, va, np.mean(vp, axis=0))
    d = ROOT / "preds" / f"final-{args.system}"
    d.mkdir(parents=True, exist_ok=True)
    write_prediction_table(task, np.mean(tp, axis=0), d / f"{name.replace('/', '__')}.csv")
    ref = board_rows(name)
    brow = [f"{ref.get('LightGBM', float('nan')):.4f}", f"{ref.get('GNN', float('nan')):.4f}",
            f"{ref.get('Kapso (Leeroo Team)', float('nan')):.4f}"]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = (f"round-2 final full-size pass (1M random train cap, refit on train+val) system={args.system} "
            f"cand={args.cand} feats={len(keep)} n_train={len(tr)} n_val={len(va)} n_test={len(te)}")
    secs = f"{time.time() - t0:.0f}"
    for seed, p_, it in zip(seeds, vp, its):
        append([now, args.iter, f"final:{args.system}", name, "val", "roc_auc", f"{score(task, va, p_):.4f}", *brow,
                secs, f"{rss():.2f}", f"{base} seed={seed} iters={it}"])
    append([now, args.iter, f"final:{args.system}", name, "val", "roc_auc", f"{v2:.4f}", *brow, secs, f"{rss():.2f}",
            f"{base} seeds-0-1 mean prediction (for comparison with round 1)"])
    append([now, args.iter, f"final:{args.system}", name, "val", "roc_auc", f"{vall:.4f}", *brow, secs, f"{rss():.2f}",
            f"{base} {len(seeds)}-seed mean prediction; test CSV = {len(seeds)}-seed mean of refits"])
    print(f"[final2] {name} val seeds " + " ".join(f"{score(task, va, p_):.6f}" for p_ in vp)
          + f" 2-seed-mean {v2:.6f} {len(seeds)}-seed-mean {vall:.6f} csv {d.name} done {time.time() - t0:.0f}s "
          f"peak rss {rss():.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
