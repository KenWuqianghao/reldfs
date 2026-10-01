"""Shared helpers: dataset/task loading from the slim local mirror, board numbers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RELDFS_DATA") or ROOT / "data").resolve()   # local mirrors and caches
os.environ.setdefault("HF_HOME", str(DATA / "hf"))

import pandas as pd  # noqa: E402

BOARD_JSON = ROOT / "docs" / "leaderboard-2026-09-24.json"
BOARD_KEYS = {"binary_classification": "classification", "regression": "regression",
              "recommendation": "recommendation"}
REF_ROWS = ["Kapso (Leeroo Team)", "RT-PluRel (fine-tuned)", "GNN + LightGBM", "GNN", "LightGBM"]


LEAN_BYTES = 60_000_000  # compressed db size above which load_ctx uses lean_db (RAM cap 6 GB)


def lean_db(ds_dir: Path, ds, task):
    """Memory-lean equivalent of task.get_db() for large mirrors.

    Same rows and values as relbench (Dataset.get_db -> upto(test_timestamp) inclusive,
    validate_and_correct_db's dangling-fkey nulling, the task's remove_columns), but read
    with a pyarrow row filter instead of a pandas copy, string columns as pandas categoricals
    instead of Python objects, and non-key integer columns downcast losslessly. relbench's
    own get_db peaks at 8.7 GB RSS on rel-event; this path stays under 2 GB.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from relbench.base import Database, Table

    man = ds.manifest
    ts = pd.Timestamp(ds.test_timestamp)
    drop = {}
    for t, c in task.remove_columns:
        drop.setdefault(t, set()).add(c)
    import pyarrow as pa
    import pyarrow.compute as pc

    pool = pa.default_memory_pool()
    tables = {}
    for name, spec in man.tables.items():
        pf = pq.ParquetFile(ds_dir / "db" / f"{name}.parquet")
        cols = [c for c in pf.schema_arrow.names if c not in drop.get(name, set())]
        mask = None
        if spec.time_col:   # Table.upto: rows with time <= test_timestamp (NaT dropped)
            tc = pf.read(columns=[spec.time_col]).column(0)
            mask = pc.fill_null(pc.less_equal(tc, pa.scalar(ts, type=tc.type)), False)
            del tc
        keys = {spec.pkey, *dict(spec.fkeys)}
        out = {}
        for c in cols:   # one column at a time keeps the peak near the final size
            a = pf.read(columns=[c]).column(0)
            if mask is not None:
                a = a.filter(mask)
            if pa.types.is_string(a.type) or pa.types.is_large_string(a.type):
                a = a.dictionary_encode()
            ser = a.to_pandas()
            del a
            if c not in keys and pd.api.types.is_integer_dtype(ser) and not pd.api.types.is_bool_dtype(ser):
                ser = pd.to_numeric(ser, downcast="integer")
            out[c] = ser.reset_index(drop=True)
            pool.release_unused()
        df = pd.DataFrame(out)
        del out
        tables[name] = Table(df=df, fkey_col_to_pkey_table=dict(spec.fkeys), pkey_col=spec.pkey,
                             time_col=spec.time_col)
    db = Database(tables)
    ds.validate_and_correct_db(db)
    return db


@dataclass
class Ctx:
    name: str            # "rel-f1/driver-dnf"
    dataset: object      # RelBenchDataset (local mirror)
    task: object
    db: object           # task.get_db(): rows up to test_timestamp, hidden cols dropped
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame   # target column absent (never present in the mirror)
    workers: int = 2
    seed: int = 0


def load_ctx(name: str, workers: int = 2, seed: int = 0) -> Ctx:
    import relbench

    ds_name, task_name = name.split("/")
    ds_dir = DATA / ds_name
    if not (ds_dir / "manifest.yaml").exists():
        raise SystemExit(f"missing {ds_dir}; run: python harness/fetch.py {ds_name} --tasks {task_name}")
    ds = relbench.load_dataset(str(ds_dir))
    task = ds.load_task(task_name)
    big = sum(f.stat().st_size for f in (ds_dir / "db").glob("*.parquet")) > LEAN_BYTES
    db = lean_db(ds_dir, ds, task) if (big or os.environ.get("RELDFS_LEAN") == "1") else task.get_db()
    tr = task.get_table("train").df.reset_index(drop=True)
    va = task.get_table("val").df.reset_index(drop=True)
    te = task.get_table("test", mask_input_cols=True).df.reset_index(drop=True)
    target = getattr(task, "target_col", None)
    assert target is None or target not in te.columns, "test labels leaked into the mirror"
    return Ctx(name, ds, task, db, tr, va, te, workers, seed)


def board_rows(name: str) -> dict:
    """{row name: test metric} for one task from the snapshot of leaderboard.json."""
    d = json.loads(BOARD_JSON.read_text())
    out = {}
    for e in d:
        for board in e.get("boards", {}).values():
            v = board.get("results", {}).get(name)
            if v is not None and e["name"] in REF_ROWS:
                out[e["name"]] = v
    return out


def cached_build(F, feat_path, db, task, plan, lab: pd.DataFrame, tag: str, threads: int = 2, extra=()):
    """F.build(...) memoised on disk under <RELDFS_DATA>/fcache/ (default data/, gitignored).

    The key hashes the feature module's source, the task name, the split tag and the label
    keys, so a changed feature engine or a different row set never reuses a stale matrix.
    Label values are not part of the key and are never stored.
    """
    import hashlib

    src = Path(feat_path).read_bytes()
    keys = lab[[task.entity_col, task.time_col]]
    h = hashlib.sha1(src)
    for e in extra:   # round 2: other source files the builder imports (e.g. domain.py)
        h.update(Path(e).read_bytes())
    h.update(f"{tag}|{len(keys)}".encode())  # tag = "<dataset>/<task>|<split>"
    h.update(pd.util.hash_pandas_object(keys, index=False).values.tobytes())
    d = DATA / "fcache"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{h.hexdigest()[:20]}.parquet"
    if f.exists() and os.environ.get("RELDFS_NOCACHE") != "1":
        return pd.read_parquet(f)
    X = F.build(db, task, plan, lab, threads)
    # RELDFS_NOCACHE_WRITE=1 (EXPERIMENTS.md, round 3): do not store the matrix. For the full-size final
    # pass, whose matrices (up to ~0.5 GB per split on amazon) would break the disk budget.
    if os.environ.get("RELDFS_NOCACHE_WRITE") != "1":
        X.to_parquet(f)
    return X
