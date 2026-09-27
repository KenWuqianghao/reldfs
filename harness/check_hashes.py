"""Compare preds/final-<system>/*.csv with results/final-preds.sha256.tsv.

    python harness/check_hashes.py [--systems reldfs,h13,base]

Prints one line per CSV (match, DIFFERENT or missing); exit 1 unless every listed CSV matches.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="reldfs,h13,base")
    args = ap.parse_args()
    want = set(args.systems.split(","))
    bad = 0
    with open(ROOT / "results" / "final-preds.sha256.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["system"] not in want:
                continue
            p = OUT / "preds" / f"final-{row['system']}" / row["file"]
            if not p.exists():
                status = "missing"
            else:
                status = "match" if hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256"] else "DIFFERENT"
            bad += status != "match"
            print(f"{row['system']:7s} {row['file']:34s} {status}")
    print("all match" if not bad else f"{bad} not matching")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
