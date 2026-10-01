"""Run a candidate on RelBench tasks and score it with the official evaluator.

    python harness/run.py --cand c0_dfs --tasks rel-f1/driver-dnf,rel-f1/driver-top3 \
        [--test] [--iter N] [--note "..."] [--workers 2]

* val (dev split): scored in-process with the task's own `task.evaluate` (RelBench's
  metric, AUROC / NMAE / MAP) against the hosted val labels.
* test (report only): the candidate's test predictions are ALWAYS written to
  preds/<cand>/<dataset>__<task>.csv via relbench.submit.write_prediction_table (the
  exact submission format). They are only SCORED with --test, through
  relbench.submit.evaluate_task (the function CI runs), which fetches the hosted
  test.parquet into data/hf. The candidate never sees test labels: the local mirror's
  test.parquet has no target column.
* one results.tsv row per (task, split).

Candidate contract: candidates/<cand>/model.py defines run(ctx) -> (val_pred, test_pred, info)
with predictions aligned to ctx.val / ctx.test rows (probabilities for classification, raw
target scale for regression).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, board_rows, load_ctx  # noqa: E402

RESULTS = ROOT / "results.tsv"
HEADER = ("iso_time\titer\tcand\ttask\tsplit\tmetric\tvalue\tlgbm_board\tgnn_board"
          "\tkapso_board\tsecs\tpeak_rss_gb\tnotes\n")


def load_cand(name: str):
    path = ROOT / "candidates" / name / "model.py"
    spec = importlib.util.spec_from_file_location(f"cand_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def append(row: list) -> None:
    if not RESULTS.exists():
        RESULTS.write_text(HEADER)
    with RESULTS.open("a") as f:
        f.write("\t".join(str(x) for x in row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--test", action="store_true", help="also score test (report only)")
    ap.add_argument("--iter", default="-")
    ap.add_argument("--note", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()
    assert args.workers <= 2, "shared box: at most 2 workers"

    from relbench.submit import evaluate_task, write_prediction_table

    mod = load_cand(args.cand)
    pred_dir = ROOT / "preds" / args.cand
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name in args.tasks.split(","):
        t0 = time.time()
        ctx = load_ctx(name, workers=args.workers, seed=args.seed)
        val_pred, test_pred, info = mod.run(ctx)
        secs = time.time() - t0
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        task = ctx.task
        val_table = task.get_table("val")
        vm = task.evaluate(val_pred, val_table)
        metric, vval = next(iter(vm.items()))
        csv = pred_dir / f"{name.replace('/', '__')}.csv"
        write_prediction_table(task, test_pred, csv)
        ref = board_rows(name)
        note = (args.note + " " + (info.get("note", "") if isinstance(info, dict) else "")).strip()
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[run] {args.cand} {name} val {metric}={vval:.4f}  ({secs:.0f}s, rss {rss:.2f} GB)"
              f"  board test: LightGBM {ref.get('LightGBM', float('nan')):.4f}"
              f" GNN {ref.get('GNN', float('nan')):.4f}"
              f" Kapso {ref.get('Kapso (Leeroo Team)', float('nan')):.4f}")
        row = [now, args.iter, args.cand, name, "val", metric, f"{vval:.4f}",
               f"{ref.get('LightGBM', float('nan')):.4f}", f"{ref.get('GNN', float('nan')):.4f}",
               f"{ref.get('Kapso (Leeroo Team)', float('nan')):.4f}", f"{secs:.0f}", f"{rss:.2f}", note]
        if not args.no_record:
            append(row)
        if args.test:
            tm = evaluate_task(name, csv)  # official path: hosted test.parquet, same as CI
            tval = next(iter(tm.values()))
            print(f"[run] {args.cand} {name} TEST {metric}={tval:.4f} (report only)")
            if not args.no_record:
                append(row[:4] + ["test", metric, f"{tval:.4f}"] + row[7:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
