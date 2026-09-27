"""LightGBM fit, early stopping on validation, and the refit on train + validation.

The model is a binary LightGBM with fixed default-style parameters on one thread. It is
early-stopped on the validation split; the test prediction comes from a second model fit on
train + validation for round(best_iter * (1 + n_val / n_train)) rounds. Only binary
classification was evaluated.
"""

from __future__ import annotations

import gc

import lightgbm as lgb
import numpy as np
import pandas as pd

MAX_TRAIN = 1_000_000   # train rows are sampled down to this many (random, RandomState(0))
# One OpenMP thread: on a shared, loaded 4-core machine two threads ran 18x slower from
# spin-waiting. The thread count can change LightGBM's floating-point summation order.
LGB_THREADS = 1
PARAMS = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=20, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbosity=-1)
MAX_ROUNDS = 3000
EARLY_STOP = 150


def sample_train(train: pd.DataFrame, target_col: str | None = None, max_train: int = MAX_TRAIN,
                 gating_cap: int = 0) -> pd.DataFrame:
    """The train rows a fit uses.

    gating_cap > 0 (cheaper comparison runs): at most gating_cap rows, sampled stratified by
    label with RandomState(0). Otherwise at most max_train rows sampled at random with
    RandomState(0) (the final runs). Row order is preserved in both cases.
    """
    tr = train
    if gating_cap and len(tr) > gating_cap:
        y = tr[target_col].to_numpy()
        rs = np.random.RandomState(0)
        idx = []
        for v in np.unique(y):
            iv = np.flatnonzero(y == v)
            idx.append(rs.choice(iv, size=int(round(gating_cap * len(iv) / len(y))), replace=False))
        tr = tr.iloc[np.sort(np.concatenate(idx))].reset_index(drop=True)
    elif len(tr) > max_train:
        tr = tr.iloc[np.sort(np.random.RandomState(0).permutation(len(tr))[:max_train])].reset_index(drop=True)
    return tr


def params(seed: int) -> dict:
    return dict(PARAMS, objective="binary", metric="auc", num_threads=LGB_THREADS, seed=seed)


def fit_refit(task, Xtr: pd.DataFrame, tr: pd.DataFrame, Xva: pd.DataFrame, va: pd.DataFrame,
              Xte=None, seed: int = 0, refit: bool = True) -> dict:
    """Fit with early stopping on (Xva, va); optionally refit on train + val and predict Xte.

    Xtr, Xva, Xte are float32 matrices with the same columns; tr and va hold the label rows
    (the task's time and target columns). Returns a dict with
      val          prediction of the early-stopped model on Xva
      test_norefit prediction of the early-stopped model on Xte (None without Xte)
      test         prediction of the refit model on Xte (None without Xte or refit)
      best_iter    the early-stopping round
    """
    y, tc = task.target_col, task.time_col
    p = params(seed)
    w = np.ones(len(tr))
    dtr = lgb.Dataset(Xtr, tr[y].to_numpy(), weight=w)
    dva = lgb.Dataset(Xva, va[y].to_numpy(), reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    it = max(1, m.best_iteration)
    out = dict(val=m.predict(Xva, num_iteration=it), test_norefit=None, test=None, best_iter=it)
    if Xte is not None:
        out["test_norefit"] = m.predict(Xte, num_iteration=it)
    del m, dtr, dva
    gc.collect()
    if not (refit and Xte is not None):
        return out
    lab = pd.concat([tr[[tc, y]], va[[tc, y]]], ignore_index=True)
    w2 = np.ones(len(lab))
    scale = 1.0 + len(va) / max(1, len(tr))
    # train + val as ONE C-contiguous float32 array, filled column by column. LightGBM would
    # otherwise copy a column-major concat into row-major order (one more full copy). Same
    # values and dtype as a pandas concat of the two frames.
    assert all(d_ == np.float32 for d_ in Xtr.dtypes) and all(d_ == np.float32 for d_ in Xva.dtypes)
    Xall = np.empty((len(Xtr) + len(Xva), Xtr.shape[1]), dtype=np.float32, order="C")
    for j, c in enumerate(Xtr.columns):
        Xall[:len(Xtr), j] = Xtr[c].to_numpy()
        Xall[len(Xtr):, j] = Xva[c].to_numpy()
    d2 = lgb.Dataset(Xall, lab[y].to_numpy(), weight=w2, params=p)
    d2.construct()
    del Xall
    gc.collect()
    m2 = lgb.train(p, d2, num_boost_round=max(1, int(round(it * scale))))
    out["test"] = m2.predict(Xte)
    del m2, d2
    gc.collect()
    return out
