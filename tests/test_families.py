"""The family-selection rule, and that it reproduces the recorded selections."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from reldfs.families import H13, RELDFS, VARIANTS, grid, select, task_drop, variant_keep

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results" / "grid"
TASKS = [
    "rel-amazon/user-churn", "rel-amazon/item-churn", "rel-avito/user-visits",
    "rel-avito/user-clicks", "rel-event/user-repeat", "rel-event/user-ignore",
    "rel-f1/driver-dnf", "rel-f1/driver-top3", "rel-hm/user-churn",
    "rel-stack/user-engagement", "rel-stack/user-badge", "rel-trial/study-outcome",
]
KEYS = [("bt1_refit", 0), ("bt1_refit", 1), ("bt2_refit", 0), ("bt2_refit", 1)]


def table(task, rows):
    """rows: {variant: [4 bt*_refit values]} -> results mapping."""
    R = {}
    for v, vals in rows.items():
        for (k, s), x in zip(KEYS, vals):
            R[(v, task, s, k)] = x
        R[(v, task, 0, "val")] = R[(v, task, 1, "val")] = 99.0   # never read by the rule
    return R


def test_all_four_deltas_must_be_positive():
    R = table("t", {"b0": [.5] * 4, "b1": [.6, .6, .6, .5], "b0gc": [.51] * 4})
    assert select(R, "b0", "t")[0] == "b0gc"


def test_largest_mean_wins_and_none_eligible_keeps_base():
    R = table("t", {"b0": [.5] * 4, "b1": [.52] * 4, "b1gc": [.51, .51, .51, .56]})
    assert select(R, "b0", "t")[0] == "b1gc"
    R = table("t", {"b0": [.5] * 4, "b1": [.4] * 4, "b1gc": [.6, .6, .6, .49]})
    assert select(R, "b0", "t")[0] == "b0"


def test_zero_delta_is_not_eligible_and_val_is_not_read():
    R = table("t", {"b0": [.5] * 4, "b1": [.5, .6, .6, .6]})
    assert select(R, "b0", "t")[0] == "b0"
    R = table("t", {"b0": [.5] * 4, "b1": [.51] * 4})
    R[("b1", "t", 0, "val")] = -1.0
    assert select(R, "b0", "t")[0] == "b1"


def test_incomplete_configuration_is_skipped():
    R = table("t", {"b0": [.5] * 4, "b1": [.9] * 4, "b0gc": [.51] * 4})
    del R[("b1", "t", 1, "bt2_refit")]
    best, rows = select(R, "b0", "t")
    assert best == "b0gc" and ("b1", None, None, False) in rows


def test_variants_and_grid():
    assert variant_keep("b0") == () and variant_keep("b1") == ("ft.",)
    assert set(variant_keep("b1gp")) == {"gc.", "pe.", "ft."}
    assert len(VARIANTS) == 8
    assert grid(["pe.", "ft."]) == ["b0", "b0pe", "b1", "b1pe"]
    assert grid(["ft."]) == ["b0", "b1"]
    assert len(grid(["gc.", "pe.", "ft."])) == 8


@pytest.mark.parametrize("name, args", [
    ("h13", sorted(GRID.glob("h13_*.bt"))),
    ("h13b", sorted(GRID.glob("h13_*.bt")) + sorted(GRID.glob("h13b_*.bt")) + [GRID / "h13b-hm-rerun.bt"]),
])
def test_famselect_reproduces_the_recorded_output(name, args):
    out = subprocess.run([sys.executable, str(ROOT / "harness" / "famselect.py"), "c13_fam:b0", *map(str, args)],
                         capture_output=True, text=True, check=True).stdout
    assert out == (GRID / f"{name}.select.txt").read_text()


@pytest.mark.parametrize("name, system", [("h13", H13), ("h13b", RELDFS)])
def test_shipped_switches_are_the_recorded_choices(name, system):
    chosen = {}
    for line in (GRID / f"{name}.select.txt").read_text().splitlines():
        if line.strip().startswith("->"):
            task, variant = line.strip()[3:].split(": ")
            chosen[task] = variant.split(":")[1]
    # the H13 grid had no configuration for amazon item-churn (gc.* and pe.* add no columns
    # there), so its selection lists 11 tasks; an unlisted task keeps the base
    missing = sorted(set(TASKS) - set(chosen))
    assert missing == (["rel-amazon/item-churn"] if name == "h13" else [])
    for t in TASKS:
        want = set(variant_keep(chosen[t])) if t in chosen else set()
        if name == "h13":      # H13 did not offer ft.*; its base dropped it
            assert "ft." not in want
        assert set(system.get(t, ())) == want, t
        assert set(task_drop(t, "reldfs" if system is RELDFS else "h13")) == {"gc.", "pe.", "ft."} - want
