"""Score the final CSVs on test (report only) and package the submission zip.

    python harness/finaltest.py --final reldfs --supp h13,base [--record]   # scores test
    python harness/finaltest.py --final reldfs --package-only               # zip only, no scoring

Scoring: relbench.submit.evaluate_submission on <RELDFS_OUT>/preds/final-<system>/, which
calls relbench.submit.evaluate_task per CSV, the function the leaderboard's CI runs. It
downloads the hosted test tables (with labels) into HF_HOME; nothing here reads a test label
directly. RUNNING IT IS A TEST RUN: the recorded numbers come from one such run, made after
the final system had been fixed and its CSVs hashed (results/final-preds.sha256.tsv).

Packaging: the final system's 12 classification CSVs are zipped with relbench.submit's own
packager into <RELDFS_OUT>/submissions/reldfs-classification.zip. After scoring, the CSV set
comes from the evaluator's validation; with --package-only it is the 12 board tasks in the
board's order, with no validation and no scoring. The zip stores file modification times, so a
rebuilt zip differs in bytes from the recorded one while its 12 members are identical.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSIFICATION, OUT, append, board_cols, now  # noqa: E402  (sets HF_HOME)


def package(final: str, families: dict, extra) -> Path:
    import relbench.submit as S

    out = OUT / "submissions" / "reldfs.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    S._package(OUT / "preds" / f"final-{final}", families, ["classification"], extra or [], out)
    z = OUT / "submissions" / "reldfs-classification.zip"
    names = zipfile.ZipFile(z).namelist()
    size = z.stat().st_size
    print(f"[zip] {z} {size} bytes ({size / 1e6:.2f} MB), {len(names)} files")
    assert len(names) == 12 and all(n.endswith(".csv") for n in names) and size < 25e6
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="reldfs")
    ap.add_argument("--supp", default="")
    ap.add_argument("--step", default="7d")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--package-only", action="store_true")
    args = ap.parse_args()

    if args.package_only:
        d = OUT / "preds" / f"final-{args.final}"
        missing = [t for t in CLASSIFICATION if not (d / f"{t.replace('/', '__')}.csv").exists()]
        assert not missing, f"missing CSVs in {d}: {missing}"
        package(args.final, {"classification": {"valid": CLASSIFICATION}}, [])
        return 0

    import relbench.submit as S

    systems = [args.final] + [s for s in args.supp.split(",") if s]
    res = {}
    for s in systems:
        d = OUT / "preds" / f"final-{s}"
        csvs = sorted(p.name for p in d.glob("*.csv"))
        assert len(csvs) == 12, (s, csvs)
        role = "FINAL" if s == args.final else "SUPPLEMENTARY (not chosen on test)"
        print(f"\n######## {s} ({role})", flush=True)
        r = S.evaluate_submission(d, num_workers=1, verbose=True)
        res[s] = r
        fam = r["families"]["classification"]
        for t in fam["present"]:
            e = r["tasks"][t]
            if args.record:
                append([now(), args.step, f"final:{s}", t, "test", e["metric_name"] or "roc_auc",
                        f"{e['metric']:.4f}" if e["metric"] is not None else "nan", *board_cols(t), "", "",
                        f"TEST report-only: relbench.submit.evaluate_submission -> evaluate_task; {role}"
                        + ("" if e["status"] == "ok" else f" ERROR {e['error']}")])
        print(f"[test] {s} classification aggregate {fam['aggregate']} complete={fam['complete']}", flush=True)
    rf = res[args.final]
    assert "classification" in rf["validated"], rf["families"]["classification"]["verdict"]
    package(args.final, rf["families"], rf.get("extra_files"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
