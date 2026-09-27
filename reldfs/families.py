"""Per-task feature-family switches and the rule that chose them.

The feature engine computes every family on every schema. A *system* decides, per task, which
of the three switchable families it keeps:

  gc.*  grandchild aggregates (H9)
  pe.*  parent context of child rows (H7)
  ft.*  day-level recency (H10)

Everything else (1-hop windows, 2-hop track records, own columns) is always on. A task that
is not listed in a system's map keeps none of the three (the "base" columns).

The switches of the final system were chosen by :func:`select`, a rule fixed before any number
of that experiment existed. It reads only train-internal time backtests (bt1_refit and
bt2_refit, seeds 0 and 1), never the validation or test split. EXPERIMENTS.md has the history.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, Mapping, Tuple

FAMILIES: Tuple[str, ...] = ("gc.", "pe.", "ft.")

# Final system (H13b): chosen by select() on the recorded backtests (results/grid/).
RELDFS: Dict[str, Tuple[str, ...]] = {
    "rel-f1/driver-top3": ("ft.",),
    "rel-event/user-ignore": ("gc.",),
    "rel-avito/user-visits": ("pe.", "ft."),
    "rel-avito/user-clicks": ("gc.", "pe.", "ft."),
    "rel-hm/user-churn": ("ft.",),
    "rel-stack/user-badge": ("gc.", "ft."),
    "rel-amazon/user-churn": ("pe.", "ft."),
    "rel-amazon/item-churn": ("ft.",),
}

# Supplementary system H13: the same rule over {gc, pe} only (ft.* not offered).
H13: Dict[str, Tuple[str, ...]] = {
    "rel-avito/user-clicks": ("gc.", "pe."),
    "rel-event/user-ignore": ("gc.",),
    "rel-stack/user-badge": ("gc.",),
    "rel-amazon/user-churn": ("pe.",),
}

SYSTEMS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "reldfs": RELDFS,   # the final system
    "h13": H13,         # supplementary
    "base": {},         # supplementary: no switchable family on any task
}


def task_keep(task: str, system: str = "reldfs") -> Tuple[str, ...]:
    """Switchable family prefixes the system keeps on this task."""
    return tuple(SYSTEMS[system].get(task, ()))


def task_drop(task: str, system: str = "reldfs") -> Tuple[str, ...]:
    """Column prefixes the system drops on this task (str.startswith accepts the tuple)."""
    keep = task_keep(task, system)
    return tuple(p for p in FAMILIES if p not in keep)


def columns(all_columns: Iterable[str], task: str, system: str = "reldfs") -> list:
    """The system's columns for a task, in the engine's column order."""
    drop = task_drop(task, system)
    return [c for c in all_columns if not c.startswith(drop)]


# Backtest grid names: b0 = base, b1 = base + ft; suffix gc / pe / gp = + gc / + pe / + both.
VARIANTS: Dict[str, Tuple[str, ...]] = {}
for _b, _ft in (("b0", ("ft.",)), ("b1", ())):
    for _s, _keep in (("", ()), ("gc", ("gc.",)), ("pe", ("pe.",)), ("gp", ("gc.", "pe."))):
        VARIANTS[_b + _s] = tuple(p for p in ("gc.", "pe.") if p not in _keep) + _ft


def variant_keep(variant: str) -> Tuple[str, ...]:
    return tuple(p for p in FAMILIES if p not in VARIANTS[variant])


def grid(non_noop: Iterable[str]) -> list:
    """b0 plus every non-empty subset of a task's non-no-op families, as variant names."""
    fams = [f for f in FAMILIES if f in set(non_noop)]
    out = ["b0"]
    for r in range(1, len(fams) + 1):
        for sub in combinations(fams, r):
            name = next(v for v in VARIANTS if set(variant_keep(v)) == set(sub))
            out.append(name)
    return out


BT_KEYS = (("bt1_refit", 0), ("bt1_refit", 1), ("bt2_refit", 0), ("bt2_refit", 1))


def select(results: Mapping[tuple, float], base: str, task: str):
    """The selection rule, for one task.

    results maps (variant, task, seed, key) -> AUROC, where key is one of the backtest keys.
    For every non-base configuration, take the 4 deltas against the base on bt1_refit and
    bt2_refit, seeds 0 and 1. A configuration is eligible only if all 4 are > 0. The eligible
    configuration with the largest mean delta wins; with none eligible the task keeps the base.
    Only the bt*_refit keys are read.

    Returns (choice, rows) where rows lists (variant, deltas, mean, eligible) or
    (variant, None, None, False) for an incomplete configuration.
    """
    vs = sorted({k[0] for k in results if k[1] == task and k[0] != base})
    best, bestm = base, None
    rows = []
    for v in vs:
        try:
            d = [results[(v, task, s, k)] - results[(base, task, s, k)] for k, s in BT_KEYS]
        except KeyError:
            rows.append((v, None, None, False))
            continue
        ok = all(x > 0 for x in d)
        mean = sum(d) / 4
        rows.append((v, d, mean, ok))
        if ok and (bestm is None or mean > bestm):
            best, bestm = v, mean
    return best, rows
