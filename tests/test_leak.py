"""The leak checker on the toy schema: the engine passes, and each planted leak is caught."""

from __future__ import annotations

import pandas as pd
import pytest

import reldfs.features as F
from reldfs.leak import check_cutoff, pick_cutoffs, truncated

K = 6


def run_check(toy):
    plan = F.make_plan(toy.db, toy.task)
    out = {}
    for t in pick_cutoffs(toy.labels[toy.task.time_col], K):
        _, _, diff = check_cutoff(toy.db, toy.task, plan, toy.labels, t)
        out[pd.Timestamp(t)] = diff
    return out


def test_every_family_has_columns(toy):
    X = F.build(toy.db, toy.task, toy.plan, toy.labels.iloc[:50])
    for prefix in ("own.", "orders.user_id.", "trk.products.", "gc.orders.reviews.", "pe.orders.", "ft.orders."):
        assert any(c.startswith(prefix) for c in X.columns), prefix


def test_truncated_drops_rows_at_and_after_cutoff(toy):
    t = pd.Timestamp("2021-01-01")
    db2 = truncated(toy.db, t, "users")
    assert (db2.table_dict["orders"].df["ts"] < t).all()
    assert (db2.table_dict["reviews"].df["ts"] < t).all()
    assert len(db2.table_dict["users"].df) == len(toy.db.table_dict["users"].df)


def test_engine_passes_leakcheck(toy):
    res = run_check(toy)
    assert len(res) >= 3
    assert all(not d for d in res.values()), {t: d[:5] for t, d in res.items() if d}


@pytest.mark.parametrize("plant, prefix", [
    ("ft", "ft."),
    ("pe", "pe."),
    ("gc-lookahead", "gc."),
    ("gc-drop_g", "gc."),
])
def test_planted_leak_is_caught(toy, monkeypatch, plant, prefix):
    if plant == "ft":
        monkeypatch.setenv("RELDFS_PLANT_FT", "1")
    elif plant == "pe":
        monkeypatch.setattr(F, "PE_LEAK", True)
    else:
        monkeypatch.setattr(F, "GC_LEAK", plant[3:])
    res = run_check(toy)
    flagged = [c for d in res.values() for c in d]
    assert flagged, f"planted leak {plant} was not detected"
    assert all(c.startswith(prefix) for c in flagged), sorted(set(flagged))[:5]
