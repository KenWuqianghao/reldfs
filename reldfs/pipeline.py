"""End-to-end prediction for one task: features, per-system fits, 2-seed test predictions.

The feature matrices are built once per task (train sampled to MAX_TRAIN rows, val, test) in
row chunks; features are row-wise functions of (entity, cutoff) and the database, and the
guard's caps are fixed in make_plan on the full database, so the chunk size changes no value.
Each system then fits on its own column subset: per seed, early stopping on val and a refit on
train + val. The test prediction of a system is the mean of its seeds' refit predictions.
"""

from __future__ import annotations

import gc
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from . import features as F
from .families import task_drop
from .model import MAX_TRAIN, fit_refit, sample_train

CHUNK = 250_000


def trim() -> None:
    """Return freed heap pages to the OS between chunks. Changes no value."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
    try:
        import pyarrow as pa
        pa.default_memory_pool().release_unused()
    except Exception:
        pass


def build_matrix(db, task, plan, lab: pd.DataFrame, keep=None, chunk: int = CHUNK,
                 threads: int = 2) -> pd.DataFrame:
    """F.build over row chunks of lab. keep(columns) -> the column subset to retain."""
    parts = []
    cols = None
    for i in range(0, len(lab), chunk):
        X = F.build(db, task, plan, lab.iloc[i:i + chunk].reset_index(drop=True), threads)
        if cols is None:
            cols = list(X.columns) if keep is None else keep(list(X.columns))
        parts.append(X[cols])
        del X
        trim()
    out = pd.concat(parts, ignore_index=True)
    del parts
    trim()
    return out


def run_task(db, task, name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
             systems: Iterable[str] = ("reldfs",), seeds: Iterable[int] = (0, 1),
             max_train: int = MAX_TRAIN, chunk: int = CHUNK, log=print) -> Dict[str, dict]:
    """Returns {system: dict(cols, val_preds, iters, val_mean_pred, test_pred)}.

    val_preds are per seed (the early-stopped models); test_pred is the mean over seeds of
    the refit models' test predictions.
    """
    systems, seeds = list(systems), list(seeds)
    drops = {s: task_drop(name, s) for s in systems}
    plan = F.make_plan(db, task)
    tr = sample_train(train, max_train=max_train)

    def keep(cols: List[str]) -> List[str]:   # union of the systems' columns, engine order
        return [c for c in cols if any(not c.startswith(d) for d in drops.values())]

    Xtr = build_matrix(db, task, plan, tr, keep, chunk)
    Xva = build_matrix(db, task, plan, val, keep, chunk)
    Xte = build_matrix(db, task, plan, test, keep, chunk)
    log(f"[reldfs] {name} built train {Xtr.shape} val {Xva.shape} test {Xte.shape}")
    colsets = {s: [c for c in Xtr.columns if not c.startswith(drops[s])] for s in systems}
    # largest column set first; the sets are nested for the shipped systems, so the matrices
    # are shrunk in place between systems instead of copied
    order = sorted(systems, key=lambda s: -len(colsets[s]))
    done, out = {}, {}
    for s in order:
        cols = colsets[s]
        key = tuple(cols)
        if key not in done:
            if set(cols) <= set(Xtr.columns) and len(cols) < Xtr.shape[1]:
                Xtr, Xva, Xte = Xtr[cols], Xva[cols], Xte[cols]
                gc.collect()
            A, B, C = (Xtr, Xva, Xte) if list(Xtr.columns) == cols else (Xtr[cols], Xva[cols], Xte[cols])
            vp, tp, its = [], [], []
            for seed in seeds:
                r = fit_refit(task, A, tr, B, val, C, seed)
                vp.append(r["val"])
                tp.append(r["test"])
                its.append(r["best_iter"])
                log(f"[reldfs] {name} system={s} cols={len(cols)} seed={seed} iters={r['best_iter']}")
            done[key] = (vp, tp, its)
        vp, tp, its = done[key]
        out[s] = dict(cols=cols, val_preds=vp, iters=its, val_mean_pred=np.mean(vp, axis=0),
                      test_pred=np.mean(tp, axis=0), n_train=len(tr))
    return out
