"""Fetch rel-trial's study registration text (round 2's txt.score input; round3/README.md, step 8i).

    python harness/fetch_trial_text.py

Reads four columns of the hosted rel-trial studies table (nct_id, brief_title, official_title,
brief_summaries) by remote column reads and writes data/rel-trial-text/studies_text.parquet
(~85 MB). fetch.py drops these string columns from the slim mirror (>= 5 MB each). The text is
a static attribute of the study, like the mirror's other studies columns; the model fits on it
only point-in-time (candidates/c16_r3/model.py txt_scores). Written in round 3
to make the round-2 file reproducible; the round-2 file was produced by the same column read.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RELDFS_DATA") or ROOT / "data").resolve()   # local mirrors and caches
os.environ.setdefault("HF_HOME", str(DATA / "hf"))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

COLS = ["nct_id", "brief_title", "official_title", "brief_summaries"]


def main() -> int:
    fs = HfFileSystem()
    pf = pq.ParquetFile(fs.open("datasets/stanford-star/relbench-v1/rel-trial/db/studies.parquet", "rb",
                                block_size=8 << 20))
    tb = pf.read(columns=COLS)
    out = DATA / "rel-trial-text"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(tb, out / "studies_text.parquet", compression="zstd")
    print(f"[text] {tb.num_rows} studies -> {out / 'studies_text.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
