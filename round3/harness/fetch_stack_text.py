"""Fetch rel-stack's users.AboutMe (v4): the one users column the slim mirror drops.

    python harness/fetch_stack_text.py

Writes data/rel-stack-text/users_about.parquet with Id and the length of AboutMe in
characters (0 for a missing or empty text). Only the length is kept.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RELDFS_DATA") or ROOT / "data").resolve()   # local mirrors and caches
os.environ.setdefault("HF_HOME", str(DATA / "hf"))

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402


def main() -> int:
    fs = HfFileSystem()
    pf = pq.ParquetFile(fs.open("datasets/stanford-star/relbench-v1/rel-stack/db/users.parquet", "rb",
                                block_size=8 << 20))
    print("[text] users columns:", pf.schema_arrow.names)
    tb = pf.read(columns=["Id", "AboutMe"])
    n = pc.fill_null(pc.utf8_length(tb.column("AboutMe")), 0)
    out = DATA / "rel-stack-text"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"Id": tb.column("Id"), "about_len": n}), out / "users_about.parquet", compression="zstd")
    print(f"[text] {tb.num_rows} users, {pc.sum(pc.greater(n, 0)).as_py()} with text -> {out / 'users_about.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
