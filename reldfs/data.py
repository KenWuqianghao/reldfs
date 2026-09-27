"""Load a RelBench task from a local mirror (the folder layout of the Hugging Face dataset).

A mirror is <data>/<dataset>/{manifest.yaml, db/*.parquet, tasks/<task>/{manifest.yaml,
train.parquet, val.parquet, test.parquet}}, as written by harness/fetch.py. relbench's own
``load_dataset(<dir>)`` reads it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LEAN_BYTES = 60_000_000  # compressed db size above which load_task uses lean_db


def lean_db(ds_dir: Path, ds, task):
    """Memory-lean equivalent of task.get_db() for large mirrors.

    Same rows and values as relbench (Dataset.get_db -> upto(test_timestamp) inclusive,
    validate_and_correct_db's dangling-fkey nulling, the task's remove_columns), but read
    with a pyarrow row filter instead of a pandas copy, string columns as pandas categoricals
    instead of Python objects, and non-key integer columns downcast losslessly. relbench's
    own get_db peaks at 8.7 GB RSS on rel-event; this path stays under 2.5 GB.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from relbench.base import Database, Table

    man = ds.manifest
    ts = pd.Timestamp(ds.test_timestamp)
    drop = {}
    for t, c in task.remove_columns:
        drop.setdefault(t, set()).add(c)

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
    dataset: object      # relbench dataset (local mirror)
    task: object
    db: object           # rows up to test_timestamp, the task's hidden columns dropped
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame   # input columns only (mask_input_cols=True)


def load_task(data_dir, name: str) -> Ctx:
    """Load dataset/task ``name`` (e.g. "rel-f1/driver-top3") from the mirror in data_dir."""
    import relbench

    ds_name, task_name = name.split("/")
    ds_dir = Path(data_dir) / ds_name
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
    assert target is None or target not in te.columns, "test labels present in the input table"
    return Ctx(name, ds, task, db, tr, va, te)
