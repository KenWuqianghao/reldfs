"""Final full-size pass (EXPERIMENTS.md 7c; protocol fixed in step 7a before any final number).

    RELDFS_NOCACHE_WRITE=1 python harness/finalpass.py --task rel-hm/user-churn \
        --engine c13_fam --systems c9_tie,c13_fam,c14_fam3 --seeds 0,1 --iter 7

One process per task. The feature matrices are built ONCE with the engine candidate's
features.py (c13_fam: every family; c14_fam3's features.py is byte-identical): train sampled to
MAX_TRAIN = 1M rows at random with RandomState(0) (model.features' rule without a gating cap),
val, and test, in CHUNK-row chunks (features are row-wise functions of (entity, cutoff) and
the db, so chunking changes nothing; it bounds the duckdb join and RSS). Each chunk keeps only
the union of the columns the systems use.

A system is a candidate whose model.py defines task_drop(name) (c13_fam, c14_fam3), or c9_tie,
which is the engine's matrix with gc.*, pe.* and ft.* dropped (the same columns in the same
order as c9_tie's own features.py; EXPERIMENTS.md 6b). Per distinct column set and per seed, the
engine's fit_predict (identical to c9_tie's) early-stops on val and refits on train+val for
round(best_iter * (1 + n_val / n_train)) rounds. Val (no refit) is scored with task.evaluate
and recorded per seed and for the 2-seed mean prediction. The test CSV of each system is the
mean of the seeds' refit test probabilities, written with relbench.submit.write_prediction_table
to preds/final-<system>/. Test is NOT scored here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import resource
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
from run import append, load_cand  # noqa: E402

CHUNK = 250_000   # --chunk overrides
C9_DROP = ("gc.", "pe.", "ft.")


def lean_fit(eng, task, Xtr, tr, Xva, va, Xte, seed):
    """eng.fit_predict's exact steps for CFG (uniform weights, refit=True, no drop_prefix), minus
    its two redundant full-matrix copies (Xtr[keep] and Xall[k2] with every weight 1), so that a
    1M-row matrix fits the 4.5 GB per-process budget (the refit matrix is built as one row-major
    float32 array). Returns (val pred of the early-stopped
    model, test pred of the refit, best_iter). Checked bit-identical to fit_predict (--check)."""
    cfg = eng.CFG
    assert cfg.get("halflife_days") is None and not cfg.get("window_days") and cfg.get("refit")
    y, tc = task.target_col, task.time_col
    p = dict(eng.PARAMS, objective="binary", metric="auc", num_threads=eng.LGB_THREADS, seed=seed)
    w = eng.weights(tr[tc], cfg)
    assert (w == 1).all()
    dtr = lgb.Dataset(Xtr, tr[y].to_numpy(), weight=w)
    dva = lgb.Dataset(Xva, va[y].to_numpy(), reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    it = max(1, m.best_iteration)
    pv = m.predict(Xva, num_iteration=it)
    del m, dtr, dva
    gc.collect()
    lab = pd.concat([tr[[tc, y]], va[[tc, y]]], ignore_index=True)
    w2 = eng.weights(lab[tc], cfg)
    scale = 1.0 + len(va) / max(1, int((w > 0).sum()))
    # train+val as ONE C-contiguous float32 array, filled column by column: LightGBM would
    # otherwise copy pd.concat's column-major block into row-major order (one more full copy;
    # stack badge peaked at 5.0 GB). Same values and dtype as fit_predict's DataFrame (--check).
    assert all(d_ == np.float32 for d_ in Xtr.dtypes) and all(d_ == np.float32 for d_ in Xva.dtypes)
    Xall = np.empty((len(Xtr) + len(Xva), Xtr.shape[1]), dtype=np.float32, order="C")
    for j, c in enumerate(Xtr.columns):
        Xall[:len(Xtr), j] = Xtr[c].to_numpy()
        Xall[len(Xtr):, j] = Xva[c].to_numpy()
    d2 = lgb.Dataset(Xall, lab[y].to_numpy(), weight=w2, params=p)
    d2.construct()
    del Xall
    gc.collect()
    m2 = lgb.train(p, d2, num_boost_round=max(1, int(round(it * scale))))
    pt = m2.predict(Xte)
    del m2, d2
    gc.collect()
    return pv, pt, it


def trim() -> None:
    """Return freed heap pages to the OS between chunks (glibc malloc_trim); RSS on amazon grew
    2.9 -> 4.6 GB over 4 train chunks from fragmentation alone. Changes no value."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--engine", default="c13_fam")
    ap.add_argument("--systems", default="c9_tie,c13_fam,c14_fam3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--iter", default="7")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--check", type=int, default=0,
                    help="check mode: train capped at N random rows, lean_fit vs fit_predict, no record")
    args = ap.parse_args()
    from relbench.submit import write_prediction_table

    eng = load_cand(args.engine)
    F = eng.F
    name = args.task
    systems = args.systems.split(",")
    drops = {}
    for s in systems:
        drops[s] = C9_DROP if s == "c9_tie" else tuple(load_cand(s).task_drop(name))
    t0 = time.time()
    ctx = load_ctx(name)
    task = ctx.task
    plan = F.make_plan(ctx.db, task)
    tr = ctx.train
    cap = args.check or eng.MAX_TRAIN
    if len(tr) > cap:   # model.features' rule without RELDFS_TRAIN_CAP
        tr = tr.iloc[np.sort(np.random.RandomState(0).permutation(len(tr))[:cap])].reset_index(drop=True)
    va, te = ctx.val, ctx.test
    keep_u = None
    cols_all = None

    def build(lab):
        nonlocal keep_u, cols_all
        parts = []
        for i in range(0, len(lab), args.chunk):
            X = F.build(ctx.db, task, plan, lab.iloc[i:i + args.chunk].reset_index(drop=True), 2)
            if keep_u is None:
                cols_all = list(X.columns)
                keep_u = [c for c in cols_all if any(not c.startswith(d) for d in drops.values())]
            parts.append(X[keep_u])
            del X
            trim()
        out = pd.concat(parts, ignore_index=True)
        del parts
        trim()
        return out

    Xtr = build(tr)
    Xva = build(va)
    Xte = build(te) if not args.check else Xva
    tb = time.time() - t0
    print(f"[final] {name} built train {Xtr.shape} val {Xva.shape} test {Xte.shape} "
          f"({len(cols_all)} engine cols) {tb:.0f}s rss {rss():.2f}", flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.check:   # lean_fit vs the candidate's own fit_predict, same inputs; nothing recorded
        s = systems[0]
        cols = [c for c in keep_u if not c.startswith(drops[s])]
        A, B = Xtr[cols], Xva[cols]
        for seed in seeds:
            out, refp, it, _ = eng.fit_predict(task, A, tr, B, va, [B, B], dict(eng.CFG, refit=True), seed)
            pv, pt, it2 = lean_fit(eng, task, A, tr, B, va, B, seed)
            print(f"[check] {name} system={s} cols={len(cols)} n_train={len(tr)} seed={seed} iters {it}/{it2} "
                  f"val identical={np.array_equal(out[0], pv)} refit identical={np.array_equal(refp[1], pt)} "
                  f"max|d|={max(np.abs(out[0] - pv).max(), np.abs(refp[1] - pt).max()):.3g}", flush=True)
        return 0
    ref = board_rows(name)
    brow = [f"{ref.get('LightGBM', float('nan')):.4f}", f"{ref.get('GNN', float('nan')):.4f}",
            f"{ref.get('Kapso (Leeroo Team)', float('nan')):.4f}"]
    # systems in decreasing column count; the column sets are nested (c9_tie in c13_fam in c14_fam3
    # on every task), so the matrices are shrunk in place between systems instead of copied.
    colsets = {s: [c for c in keep_u if not c.startswith(drops[s])] for s in systems}
    order = sorted(systems, key=lambda s: -len(colsets[s]))
    done = {}   # column tuple -> (val preds per seed, refit test preds per seed, iters)
    for s in order:
        cols = colsets[s]
        key = tuple(cols)
        if key not in done:
            assert set(cols) <= set(Xtr.columns), "column sets are not nested"
            if len(cols) < Xtr.shape[1]:
                Xtr, Xva, Xte = Xtr[cols], Xva[cols], Xte[cols]
                gc.collect()
            vp, tp, its = [], [], []
            for seed in seeds:
                t1 = time.time()
                pv, pt, it = lean_fit(eng, task, Xtr, tr, Xva, va, Xte, seed)
                vp.append(pv)
                tp.append(pt)
                its.append(it)
                print(f"[final] {name} cols={len(cols)} seed={seed} val={score(task, va, pv):.6f} "
                      f"iters={it} fit {time.time() - t1:.0f}s rss {rss():.2f}", flush=True)
            done[key] = (vp, tp, its)
        vp, tp, its = done[key]
        vals = [score(task, va, p) for p in vp]
        vmean = score(task, va, np.mean(vp, axis=0))
        d = ROOT / "preds" / f"final-{s}"
        d.mkdir(parents=True, exist_ok=True)
        write_prediction_table(task, np.mean(tp, axis=0), d / f"{name.replace('/', '__')}.csv")
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        secs = f"{time.time() - t0:.0f}"
        base = (f"final full-size pass (1M random train cap, refit on train+val) system={s} "
                f"drop={drops[s]} feats={len(cols)} n_train={len(tr)} n_val={len(va)} n_test={len(te)}")
        for seed, v, it in zip(seeds, vals, its):
            append([now, args.iter, f"final:{s}", name, "val", "roc_auc", f"{v:.4f}", *brow, secs,
                    f"{rss():.2f}", f"{base} seed={seed} iters={it}"])
        append([now, args.iter, f"final:{s}", name, "val", "roc_auc", f"{vmean:.4f}", *brow, secs,
                f"{rss():.2f}", f"{base} 2-seed mean prediction (seeds {args.seeds}); test CSV = 2-seed mean of refits"])
        print(f"[final] {name} system={s} feats={len(cols)} val seeds " + " ".join(f"{v:.6f}" for v in vals)
              + f" 2-seed-mean {vmean:.6f}  csv {d.name}", flush=True)
    print(f"[final] {name} done {time.time() - t0:.0f}s peak rss {rss():.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
