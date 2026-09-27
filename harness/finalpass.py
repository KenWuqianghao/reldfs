"""Final full-size pass for one task: features once, every system, 2 seeds, test CSVs.

    RELDFS_NOCACHE_WRITE=1 python harness/finalpass.py --task rel-f1/driver-top3 \
        [--systems reldfs,h13,base] [--seeds 0,1] [--record]

Train is sampled to 1M rows at random with RandomState(0) (tasks with fewer rows use all of
them); train, val and test are featurised once, in 250k-row chunks, keeping the union of the
systems' columns. Per system and seed, LightGBM is early-stopped on val and refit on
train + val for round(best_iter * (1 + n_val / n_train)) rounds. The test CSV of a system is
the mean of the seeds' refit probabilities, written with relbench.submit.write_prediction_table
to <RELDFS_OUT>/preds/final-<system>/<dataset>__<task>.csv. Val is scored with task.evaluate
per seed and for the seed-mean prediction; --record appends those rows to results.tsv. Test is
not scored here (harness/finaltest.py).

Systems: reldfs (the final system), h13 and base (supplementary; reldfs/families.py).
Peak memory on the largest tasks was 4.2 GB; run hm, stack badge and the amazon tasks alone.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, append, board_cols, load_ctx, now, rss_gb, score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--systems", default="reldfs,h13,base")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--step", default="7c")
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()

    from relbench.submit import write_prediction_table

    from reldfs.families import task_drop
    from reldfs.pipeline import run_task

    name = args.task
    systems = args.systems.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    t0 = time.time()
    ctx = load_ctx(name)
    task = ctx.task
    res = run_task(ctx.db, task, name, ctx.train, ctx.val, ctx.test, systems=systems, seeds=seeds,
                   chunk=args.chunk, log=lambda m: print(m, flush=True))
    brow = board_cols(name)
    for s in systems:
        r = res[s]
        vals = [score(task, ctx.val, p) for p in r["val_preds"]]
        vmean = score(task, ctx.val, r["val_mean_pred"])
        d = OUT / "preds" / f"final-{s}"
        d.mkdir(parents=True, exist_ok=True)
        csv = d / f"{name.replace('/', '__')}.csv"
        write_prediction_table(task, r["test_pred"], csv)
        print(f"[final] {name} system={s} feats={len(r['cols'])} val seeds "
              + " ".join(f"{v:.6f}" for v in vals) + f" seed-mean {vmean:.6f} -> {csv}", flush=True)
        if args.record:
            base = (f"final full-size pass (1M random train cap, refit on train+val) system={s} "
                    f"drop={task_drop(name, s)} feats={len(r['cols'])} n_train={r['n_train']} "
                    f"n_val={len(ctx.val)} n_test={len(ctx.test)}")
            secs = f"{time.time() - t0:.0f}"
            for seed, v, it in zip(seeds, vals, r["iters"]):
                append([now(), args.step, f"final:{s}", name, "val", "roc_auc", f"{v:.4f}", *brow, secs,
                        f"{rss_gb():.2f}", f"{base} seed={seed} iters={it}"])
            append([now(), args.step, f"final:{s}", name, "val", "roc_auc", f"{vmean:.4f}", *brow, secs,
                    f"{rss_gb():.2f}", f"{base} 2-seed mean prediction; test CSV = 2-seed mean of refits"])
    print(f"[final] {name} done {time.time() - t0:.0f}s peak rss {rss_gb():.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
