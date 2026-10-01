"""The one report-only test run of the closing stage (EXPERIMENTS.md 7a and 7d).

    python harness/finaltest.py --final c9_tie --supp c13_fam,c14_fam3 --iter 7

Scores preds/final-<system>/ for the final system (chosen by rule before this runs) and the
supplementary systems in ONE run, with relbench.submit.evaluate_submission, which calls
relbench.submit.evaluate_task per CSV: the function CI runs (hosted test.parquet, fetched into
data/hf). Nothing here reads a test label directly. One results.tsv row per (system, task),
split "test". Then it packages the FINAL system only, with relbench.submit's own packager, into
submissions/reldfs-classification.zip, and checks it (12 CSVs, < 25 MB). Report only: nothing
is chosen on these numbers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from common import board_rows  # noqa: E402  (sets HF_HOME=data/hf)
from run import append  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--supp", default="")
    ap.add_argument("--iter", default="7")
    ap.add_argument("--note", default="final full-size pass CSVs (1M cap, refit train+val, 2-seed mean)",
                    help="round 3: describes the CSVs in the results.tsv rows")
    ap.add_argument("--rule", default="EXPERIMENTS.md 7a", help="round 3: where the final system was fixed")
    ap.add_argument("--package-only", action="store_true",
                    help="zip preds/final-<final>/ with relbench.submit's packager; no scoring")
    args = ap.parse_args()
    import relbench.submit as S

    if args.package_only:   # round 2: package without scoring (no test label is read)
        d = ROOT / "preds" / f"final-{args.final}"
        tasks = S.LEADERBOARD_TASKS["classification"]
        missing = [t for t in tasks if not (d / f"{t.replace('/', '__')}.csv").exists()]
        assert not missing, f"missing CSVs in {d}: {missing}"
        out = ROOT / "submissions" / "reldfs.zip"
        out.parent.mkdir(exist_ok=True)
        S._package(d, {"classification": {"valid": list(tasks)}}, ["classification"], [], out)
        z = ROOT / "submissions" / "reldfs-classification.zip"
        names = zipfile.ZipFile(z).namelist()
        print(f"[zip] {z} {z.stat().st_size} bytes, {len(names)} files")
        assert len(names) == 12 and z.stat().st_size < 25e6
        return 0

    systems = [args.final] + [s for s in args.supp.split(",") if s]
    res = {}
    for s in systems:
        d = ROOT / "preds" / f"final-{s}"
        csvs = sorted(p.name for p in d.glob("*.csv"))
        assert len(csvs) == 12, (s, csvs)
        print(f"\n######## {s} ({'FINAL, chosen by rule' if s == args.final else 'SUPPLEMENTARY, not chosen on test'})",
              flush=True)
        r = S.evaluate_submission(d, num_workers=1, verbose=True)
        res[s] = r
        fam = r["families"]["classification"]
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        role = (f"FINAL system (chosen by rule, {args.rule})" if s == args.final
                else "SUPPLEMENTARY system (reported, not chosen on test)")
        for t in fam["present"]:
            e = r["tasks"][t]
            ref = board_rows(t)
            append([now, args.iter, f"final:{s}", t, "test", e["metric_name"] or "roc_auc",
                    f"{e['metric']:.4f}" if e["metric"] is not None else "nan",
                    f"{ref.get('LightGBM', float('nan')):.4f}", f"{ref.get('GNN', float('nan')):.4f}",
                    f"{ref.get('Kapso (Leeroo Team)', float('nan')):.4f}", "", "",
                    f"TEST report-only, run once: relbench.submit.evaluate_submission -> evaluate_task "
                    f"(CI path); {role}; {args.note}"
                    + ("" if e["status"] == "ok" else f" ERROR {e['error']}")])
        print(f"[test] {s} classification aggregate {fam['aggregate']} complete={fam['complete']}", flush=True)
    rf = res[args.final]
    assert "classification" in rf["validated"], rf["families"]["classification"]["verdict"]
    out = ROOT / "submissions" / "reldfs.zip"
    out.parent.mkdir(exist_ok=True)
    S._package(ROOT / "preds" / f"final-{args.final}", rf["families"], ["classification"],
               rf.get("extra_files") or [], out)
    z = ROOT / "submissions" / "reldfs-classification.zip"
    names = zipfile.ZipFile(z).namelist()
    size = z.stat().st_size
    print(f"[zip] {z} {size} bytes ({size / 1e6:.2f} MB), {len(names)} files: {sorted(names)}")
    assert len(names) == 12 and all(n.endswith(".csv") for n in names) and size < 25e6
    return 0


if __name__ == "__main__":
    sys.exit(main())
