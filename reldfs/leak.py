"""Time-leakage check for the feature builder.

For a cutoff t, the label rows at t are featurised twice:

  (a) with the database as the model sees it, in ONE batch together with up to 50 label rows
      of the latest cutoff (as in a real multi-cutoff build), so that batch-level state
      cannot hide a leak; only the rows at t are compared;
  (b) with every table that has a time column truncated to rows with time < t (the entity
      table is left whole: its rows carry the category vocabulary, and a row's own columns
      are static attributes).

Any difference means a feature read a row at or after t. The plan is built once from the
full database (vocabulary and guard caps only) and reused for both builds.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from . import features as F


def truncated(db, cutoff, keep: str):
    """A copy of db where every timed table except ``keep`` holds only rows with time < cutoff."""
    from relbench.base import Database

    tables = {}
    for name, tb in db.table_dict.items():
        t2 = copy.copy(tb)
        if tb.time_col is not None and name != keep:
            t2.df = tb.df[tb.df[tb.time_col] < cutoff]
        tables[name] = t2
    return Database(tables)


def pick_cutoffs(times, k: int = 8, seed: int = 0) -> list:
    """k - 1 random distinct cutoffs plus the latest one."""
    ts = np.sort(pd.unique(times))
    rng = np.random.RandomState(seed)
    return sorted(set(rng.choice(ts, size=min(k - 1, len(ts)), replace=False)) | {ts[-1]})


def check_cutoff(db, task, plan, labels: pd.DataFrame, t, max_rows: int = 300, threads: int = 2):
    """Compare (a) and (b) at cutoff t. Returns (n_rows, n_features, differing columns)."""
    tcol = task.time_col
    ts = np.sort(labels[tcol].unique())
    lab = labels[labels[tcol] == t]
    if len(lab) > max_rows:
        lab = lab.sample(max_rows, random_state=0)
    comp = labels[labels[tcol] == ts[-1]]
    comp = comp.sample(min(len(comp), 50), random_state=1) if t != ts[-1] else comp.iloc[:0]
    a = F.build(db, task, plan, pd.concat([lab, comp], ignore_index=True), threads).iloc[:len(lab)]
    b = F.build(truncated(db, pd.Timestamp(t), task.entity_table), task, plan, lab, threads)
    same = np.isclose(a.to_numpy(), b.to_numpy(), equal_nan=True, rtol=1e-9, atol=1e-9)
    diff = [c for c, ok in zip(a.columns, same.all(axis=0)) if not ok]
    return len(lab), a.shape[1], diff
