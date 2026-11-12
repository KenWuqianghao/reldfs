"""Round-3 final pass (round3/README.md, step 9b): one small task, one variant string.

    python harness/finalpass3.py --task rel-f1/driver-dnf --variant "base~base+pfn" \
        --cand c16_r3 --system r3 --seeds 0,1,2,3,4 --iter 9

For the tasks whose round-3 choice changed (all of them small: no train cap applies). The
variant grammar is harness/rollbt.py's: groups joined by "~" (rank-average blend), each group
a "|" list of sub-variants (probability mean). Per sub-variant and seed:
  * val prediction: fit on train (LightGBM early-stopped on val; TabPFN fit on train only),
    exactly the number the gate read;
  * test prediction: the refit on train+val (LightGBM: round(best_iter x (1 + n_val / n_train))
    rounds; TabPFN: context sampled from train+val).
Per group: the mean over its sub-variants and ALL seeds; groups are then rank-averaged per
split (ranks scaled to [0, 1]). The CSV is written with relbench.submit.write_prediction_table.
Val is recorded per seed (seed-level blend), for seeds 0-1 and for all seeds. Test is NOT
scored here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from backtest import score  # noqa: E402
from common import board_rows, load_ctx  # noqa: E402
from rollbt import combine  # noqa: E402
from run import append, load_cand  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--cand", default="c16_r3")
    ap.add_argument("--system", default="r3")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--iter", default="9")
    args = ap.parse_args()
    from relbench.submit import write_prediction_table

    mod = load_cand(args.cand)
    name = args.task
    t0 = time.time()
    ctx = load_ctx(name)
    task = ctx.task
    # v4: a larger train table is cut to mod.MAX_TRAIN random rows by mod.features (n_train is logged)
    X, fr = mod.features(ctx)
    groups = [[mod.variant_cfg(name, sv) for sv in g.split("|")] for g in args.variant.split("~")]
    seeds = [int(s) for s in args.seeds.split(",")]
    PV = {s: [] for s in seeds}   # seed -> per group: mean over sub-variants (val)
    PT = {s: [] for s in seeds}   # seed -> per group (test)
    for seed in seeds:
        for cfgs in groups:
            pv, pt = [], []
            for c in cfgs:
                if c.get("learner") == "pfn":
                    v = mod.fit_predict(task, X["train"], fr["train"], X["val"], fr["val"], [X["val"]],
                                        dict(c, refit=False), seed)[0][0]
                    t = mod.fit_predict(task, X["train"], fr["train"], X["val"], fr["val"], [X["test"]],
                                        dict(c, refit=True), seed)[1][0]
                else:
                    out, ref, _, _ = mod.fit_predict(task, X["train"], fr["train"], X["val"], fr["val"],
                                                     [X["val"], X["test"]], dict(c, refit=True), seed)
                    v, t = out[0], ref[1]
                pv.append(v)
                pt.append(t)
            PV[seed].append(np.mean(pv, axis=0))
            PT[seed].append(np.mean(pt, axis=0))
        print(f"[final3] {name} seed={seed} val={score(task, fr['val'], combine(PV[seed])):.6f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    ng = len(groups)

    def pooled(P, ss):
        return combine([np.mean([P[s][g] for s in ss], axis=0) for g in range(ng)])

    v2 = score(task, fr["val"], pooled(PV, seeds[:2]))
    vall = score(task, fr["val"], pooled(PV, seeds))
    test_pred = pooled(PT, seeds)
    assert np.all((test_pred >= 0) & (test_pred <= 1)) and len(test_pred) == len(ctx.test)
    d = ROOT / "preds" / f"final-{args.system}"
    d.mkdir(parents=True, exist_ok=True)
    write_prediction_table(task, test_pred, d / f"{name.replace('/', '__')}.csv")
    ref = board_rows(name)
    brow = [f"{ref.get('LightGBM', float('nan')):.4f}", f"{ref.get('GNN', float('nan')):.4f}",
            f"{ref.get('Kapso (Leeroo Team)', float('nan')):.4f}"]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = (f"round-3 final pass (harness/finalpass3.py; refit on train+val) system={args.system} "
            f"cand={args.cand} variant={args.variant} n_train={len(fr['train'])} n_val={len(fr['val'])} "
            f"n_test={len(ctx.test)}")
    secs = f"{time.time() - t0:.0f}"
    for s in seeds:
        append([now, args.iter, f"final:{args.system}", name, "val", "roc_auc",
                f"{score(task, fr['val'], combine(PV[s])):.4f}", *brow, secs, "", f"{base} seed={s}"])
    append([now, args.iter, f"final:{args.system}", name, "val", "roc_auc", f"{v2:.4f}", *brow, secs, "",
            f"{base} seeds-0-1 pooled prediction"])
    append([now, args.iter, f"final:{args.system}", name, "val", "roc_auc", f"{vall:.4f}", *brow, secs, "",
            f"{base} {len(seeds)}-seed pooled prediction; test CSV = the same pooling of the refits"])
    print(f"[final3] {name} val 2-seed {v2:.6f} {len(seeds)}-seed {vall:.6f} csv {d.name} "
          f"done {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
