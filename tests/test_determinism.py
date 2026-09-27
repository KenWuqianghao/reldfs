"""The feature builder is deterministic, row-wise, and family switches only drop columns."""

from __future__ import annotations

import numpy as np

import reldfs.features as F
from reldfs.families import FAMILIES, SYSTEMS, columns, task_drop
from reldfs.model import fit_refit, sample_train
from reldfs.pipeline import build_matrix


def same(a, b) -> bool:
    return list(a.columns) == list(b.columns) and np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_build_is_bit_identical_across_runs(toy):
    a = F.build(toy.db, toy.task, F.make_plan(toy.db, toy.task), toy.labels)
    b = F.build(toy.db, toy.task, F.make_plan(toy.db, toy.task), toy.labels)
    assert (a.dtypes == np.float32).all()
    assert same(a, b)


def test_chunking_and_row_order_change_no_value(toy):
    whole = F.build(toy.db, toy.task, toy.plan, toy.labels)
    chunked = build_matrix(toy.db, toy.task, toy.plan, toy.labels, chunk=97)
    assert same(whole, chunked)
    perm = np.random.RandomState(3).permutation(len(toy.labels))
    shuffled = F.build(toy.db, toy.task, toy.plan, toy.labels.iloc[perm].reset_index(drop=True))
    assert np.array_equal(shuffled.to_numpy(), whole.to_numpy()[perm], equal_nan=True)


def test_family_switches_only_drop_columns(toy):
    X = F.build(toy.db, toy.task, toy.plan, toy.labels.iloc[:80])
    for system, fams in SYSTEMS.items():
        for task in ("rel-avito/user-clicks", "rel-f1/driver-dnf", "rel-hm/user-churn"):
            cols = columns(X.columns, task, system)
            kept = set(fams.get(task, ()))
            assert cols == [c for c in X.columns if not c.startswith(task_drop(task, system))]
            for f in FAMILIES:
                assert any(c.startswith(f) for c in cols) == (f in kept)
            base = [c for c in X.columns if not c.startswith(FAMILIES)]
            assert [c for c in cols if not c.startswith(FAMILIES)] == base


def test_fit_is_deterministic(toy):
    X = F.build(toy.db, toy.task, toy.plan, toy.labels)
    t = toy.labels["timestamp"]
    tr_m, va_m, te_m = t < "2021-01-01", (t >= "2021-01-01") & (t < "2021-04-01"), t >= "2021-04-01"
    tr, va = toy.labels[tr_m].reset_index(drop=True), toy.labels[va_m].reset_index(drop=True)
    Xtr, Xva, Xte = (X[m].reset_index(drop=True) for m in (tr_m, va_m, te_m))
    r1 = fit_refit(toy.task, Xtr, tr, Xva, va, Xte, seed=0)
    r2 = fit_refit(toy.task, Xtr, tr, Xva, va, Xte, seed=0)
    assert r1["best_iter"] == r2["best_iter"] >= 1
    for k in ("val", "test", "test_norefit"):
        assert r1[k].shape == (len(Xva) if k == "val" else len(Xte),)
        assert np.array_equal(r1[k], r2[k])


def test_sample_train():
    import pandas as pd

    df = pd.DataFrame({"y": [0, 1] * 50, "x": range(100)})
    a = sample_train(df, "y", max_train=30)
    assert len(a) == 30 and a["x"].is_monotonic_increasing
    assert a.equals(sample_train(df, "y", max_train=30))
    s = sample_train(df, "y", gating_cap=20)
    assert len(s) == 20 and s["y"].sum() == 10
    assert sample_train(df, "y").equals(df)
