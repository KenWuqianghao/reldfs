"""Time-leakage check of the feature engine on real tasks.

    python harness/leakcheck.py --tasks rel-f1/driver-dnf,rel-f1/driver-top3 [--k 8]
        [--plant ft|pe|gc-lookahead|gc-drop_g]

For k sampled cutoffs per task (k - 1 random train/val cutoffs plus the latest one), the label
rows at the cutoff are featurised from the full database, in one batch with up to 50 rows of
the latest cutoff, and from a database truncated to rows with time < cutoff; any difference
is a leak (reldfs/leak.py). The engine computes every family, so this checks every system.
Exit 1 on any leak.

--plant switches on one planted leak (the checker must then FAIL where the planted family has
rows at or after the cutoff):
  ft            a day enters ft.* when d < t + timedelta
  pe            the pe.* child time filter is dropped
  gc-lookahead  gc.* reads rows < t + timedelta
  gc-drop_g     gc.* ignores the grandchild's own time
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_ctx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--plant", default="", choices=["", "ft", "pe", "gc-lookahead", "gc-drop_g"])
    args = ap.parse_args()
    if args.plant == "ft":
        os.environ["RELDFS_PLANT_FT"] = "1"
    elif args.plant == "pe":
        os.environ["RELDFS_PLANT_PE"] = "1"
    elif args.plant.startswith("gc-"):
        os.environ["RELDFS_PLANT_GC"] = args.plant[3:]

    import pandas as pd

    import reldfs.features as F
    from reldfs.leak import check_cutoff, pick_cutoffs

    tag = f" (planted: {args.plant})" if args.plant else ""
    bad = 0
    for name in args.tasks.split(","):
        ctx = load_ctx(name)
        task = ctx.task
        plan = F.make_plan(ctx.db, task)
        labels = pd.concat([ctx.train, ctx.val], ignore_index=True)
        for t in pick_cutoffs(labels[task.time_col], args.k):
            n, nf, diff = check_cutoff(ctx.db, task, plan, labels, t)
            status = "ok" if not diff else f"LEAK in {len(diff)} cols: {diff[:6]}"
            bad += bool(diff)
            print(f"[leakcheck] {name} cutoff={pd.Timestamp(t).date()} rows={n} feats={nf} {status}{tag}",
                  flush=True)
    print("[leakcheck] PASS" if not bad else f"[leakcheck] FAIL ({bad} cutoffs){tag}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
