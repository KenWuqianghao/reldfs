"""Shared harness helpers: paths, the results log, board rows, the feature cache.

Paths (all overridable):
  RELDFS_DATA   local mirrors, one folder per dataset (default: <repo>/data)
  HF_HOME       Hugging Face cache; relbench.submit downloads the hosted test tables here
                when it scores (default: <RELDFS_DATA>/hf)
  RELDFS_OUT    predictions and submissions (default: <repo>)
  RELDFS_TMP    duckdb spill directory (default: <system temp dir>/reldfs-duckdb)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RELDFS_DATA") or ROOT / "data").resolve()
OUT = Path(os.environ.get("RELDFS_OUT") or ROOT).resolve()
os.environ.setdefault("HF_HOME", str(DATA / "hf"))
sys.path.insert(0, str(ROOT))   # run from a checkout without installing

import pandas as pd  # noqa: E402

RESULTS = Path(os.environ.get("RELDFS_RESULTS") or ROOT / "results.tsv")
HEADER = ("iso_time\tstep\tsystem\ttask\tsplit\tmetric\tvalue\tlgbm_board\tgnn_board"
          "\tkapso_board\tsecs\tpeak_rss_gb\tnotes\n")
BOARD_JSON = ROOT / "docs" / "leaderboard-2026-09-24.json"
REF_ROWS = ["Kapso (Leeroo Team)", "RT-PluRel (fine-tuned)", "GNN + LightGBM", "GNN", "LightGBM"]
CLASSIFICATION = [
    "rel-amazon/user-churn", "rel-amazon/item-churn", "rel-avito/user-visits",
    "rel-avito/user-clicks", "rel-event/user-repeat", "rel-event/user-ignore",
    "rel-f1/driver-dnf", "rel-f1/driver-top3", "rel-hm/user-churn",
    "rel-stack/user-engagement", "rel-stack/user-badge", "rel-trial/study-outcome",
]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append(row: list) -> None:
    """One row of results.tsv."""
    if not RESULTS.exists():
        RESULTS.write_text(HEADER)
    with RESULTS.open("a") as f:
        f.write("\t".join(str(x) for x in row) + "\n")


def board_rows(name: str) -> dict:
    """{row name: test metric} for one task from the leaderboard snapshot."""
    d = json.loads(BOARD_JSON.read_text())
    out = {}
    for e in d:
        for board in e.get("boards", {}).values():
            v = board.get("results", {}).get(name)
            if v is not None and e["name"] in REF_ROWS:
                out[e["name"]] = v
    return out


def board_cols(name: str) -> list:
    ref = board_rows(name)
    return [f"{ref.get(k, float('nan')):.4f}" for k in ("LightGBM", "GNN", "Kapso (Leeroo Team)")]


def load_ctx(name: str):
    from reldfs.data import load_task

    return load_task(DATA, name)


def cached_build(db, task, plan, lab: pd.DataFrame, tag: str, threads: int = 2):
    """reldfs.features.build memoised on disk under <RELDFS_DATA>/fcache/.

    The key hashes the feature module's source, the split tag and the label keys, so a changed
    engine or a different row set never reuses a stale matrix. Label values are not part of
    the key and are never stored. RELDFS_NOCACHE=1 skips reading, RELDFS_NOCACHE_WRITE=1
    skips writing.
    """
    import reldfs.features as F

    keys = lab[[task.entity_col, task.time_col]]
    h = hashlib.sha1(Path(F.__file__).read_bytes())
    h.update(f"{tag}|{len(keys)}".encode())
    h.update(pd.util.hash_pandas_object(keys, index=False).values.tobytes())
    d = DATA / "fcache"
    f = d / f"{h.hexdigest()[:20]}.parquet"
    if f.exists() and os.environ.get("RELDFS_NOCACHE") != "1":
        return pd.read_parquet(f)
    X = F.build(db, task, plan, lab, threads)
    if os.environ.get("RELDFS_NOCACHE_WRITE") != "1":
        d.mkdir(parents=True, exist_ok=True)
        X.to_parquet(f)
    return X


def score(task, df, pred) -> float:
    """The task's own metric (task.evaluate) on arbitrary labelled rows."""
    import numpy as np
    from relbench.base import Table

    tb = Table(df=df.reset_index(drop=True), fkey_col_to_pkey_table={}, pkey_col=None,
               time_col=task.time_col)
    return float(next(iter(task.evaluate(np.asarray(pred), tb).values())))


def rss_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
