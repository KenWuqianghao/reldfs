"""Build a slim local mirror of a RelBench dataset from the Hugging Face Hub.

    python harness/fetch.py rel-f1 [--tasks driver-dnf,driver-top3] [--keep-text]

Writes data/<dataset>/ in the exact RelBench folder layout (manifest.yaml, db/*.parquet,
tasks/<task>/{manifest.yaml,train,val,test}.parquet) so `relbench.load_dataset(<dir>)` works.

Disk rules (../docs/profile.md):
  * db tables are read remotely by column (HTTP range reads through HfFileSystem); string
    columns whose compressed size is >= 5 MB (review_text, Body, criteria, ...) are dropped
    unless --keep-text. rel-amazon's 7.2 GB review.parquet becomes ~120 MB this way.
  * the local test.parquet is written WITHOUT the target column. Test labels only ever live
    in the HF cache that relbench.submit.evaluate_task fetches (data/hf), and only the
    official evaluator reads them (harness/run.py --test).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RELDFS_DATA") or ROOT / "data").resolve()   # local mirrors and caches
os.environ.setdefault("HF_HOME", str(DATA / "hf"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

REPO = "datasets/stanford-star/relbench-v1"
TEXT_LIMIT = 5_000_000  # compressed bytes; bigger string columns are dropped


def disk_free_gb() -> float:
    return shutil.disk_usage(str(ROOT)).free / 1e9


def project_gb() -> float:
    tot = 0
    for p in ROOT.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                tot += p.stat().st_size
        except OSError:
            pass
    return tot / 1e9


def col_sizes(pf: pq.ParquetFile) -> dict:
    md, out = pf.metadata, {}
    for rg in range(md.num_row_groups):
        r = md.row_group(rg)
        for c in range(r.num_columns):
            cc = r.column(c)
            k = cc.path_in_schema.split(".")[0]
            out[k] = out.get(k, 0) + cc.total_compressed_size
    return out


def is_text(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--tasks", default="", help="comma list; default = all hosted tasks")
    ap.add_argument("--keep-text", action="store_true")
    ap.add_argument("--max-project-gb", type=float, default=2.5)
    args = ap.parse_args()

    fs = HfFileSystem()
    base = f"{REPO}/{args.dataset}"
    out = DATA / args.dataset
    (out / "db").mkdir(parents=True, exist_ok=True)
    (out / "manifest.yaml").write_text(fs.read_text(f"{base}/manifest.yaml"))
    manifest = yaml.safe_load((out / "manifest.yaml").read_text())

    # Rough pre-check: slim size of everything we are about to write.
    plan = []
    for name in manifest["tables"]:
        pf = pq.ParquetFile(fs.open(f"{base}/db/{name}.parquet", "rb", block_size=1 << 20))
        sizes, sch = col_sizes(pf), pf.schema_arrow
        keep = [
            c for c in sch.names
            if args.keep_text or not is_text(sch.field(c).type) or sizes.get(c, 0) < TEXT_LIMIT
        ]
        drop = [c for c in sch.names if c not in keep]
        plan.append((name, keep, drop, sum(sizes.get(c, 0) for c in keep)))
    est = sum(p[3] for p in plan) / 1e9
    print(f"[fetch] {args.dataset}: slim db ~{est:.2f} GB; disk free {disk_free_gb():.2f} GB;"
          f" project {project_gb():.2f} GB")
    if project_gb() + est > args.max_project_gb:
        print("[fetch] REFUSED: would push the project past the disk budget")
        return 2
    if disk_free_gb() - est < 0.5:
        print("[fetch] REFUSED: would leave < 0.5 GB free on the shared disk")
        return 2

    for name, keep, drop, _ in plan:
        dst = out / "db" / f"{name}.parquet"
        if dst.exists():
            continue
        pf = pq.ParquetFile(fs.open(f"{base}/db/{name}.parquet", "rb", block_size=8 << 20))
        tb = pf.read(columns=keep)
        pq.write_table(tb, dst, compression="zstd")
        print(f"[fetch]   db/{name}: {tb.num_rows} rows, dropped {drop or '-'}")

    tasks = [t for t in args.tasks.split(",") if t] or [
        Path(p).name for p in fs.ls(f"{base}/tasks", detail=False)
    ]
    for t in tasks:
        tdir = out / "tasks" / t
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / "manifest.yaml").write_text(fs.read_text(f"{base}/tasks/{t}/manifest.yaml"))
        tm = yaml.safe_load((tdir / "manifest.yaml").read_text())
        for split in ("train", "val", "test"):
            dst = tdir / f"{split}.parquet"
            if dst.exists():
                continue
            tb = pq.read_table(fs.open(f"{base}/tasks/{t}/{split}.parquet", "rb"))
            if split == "test":
                # Never keep test labels in the working mirror.
                for col in (tm.get("target_col"), tm.get("dst_entity_col")):
                    if col and col in tb.column_names:
                        tb = tb.drop(col)
            pq.write_table(tb, dst, compression="zstd")
        print(f"[fetch]   task {t}: ok")
    print(f"[fetch] done; project {project_gb():.2f} GB; disk free {disk_free_gb():.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
