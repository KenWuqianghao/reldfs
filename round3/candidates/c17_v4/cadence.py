"""v4: cadence family cd.* for the churn tasks (rel-hm, rel-amazon).

The label asks if an entity that was active in the last period (one timedelta) is active in
the next one. cd.* describes the entity's rhythm in units of that period, from event rows
before the seed only:
  counts in each of the last 12 periods and in older bands; active-period shares over the
  last 4 / 8 / 12 / 26 / 52 periods; the run of active periods up to now; the entity's own
  repeat rate (of its active periods, the share followed by an active period); distinct
  event days, recency, tenure; mean / sd / median / max gap between event days and the
  recency in units of the mean gap; value means; distinct items.

    python candidates/c17_v4/cadence.py --task rel-hm/user-churn --leakcheck   (RELDFS_PLANT=1 must FAIL)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLANT = os.environ.get("RELDFS_PLANT") == "1"
# task -> (event table, entity key, time column, numeric value columns, item column)
TASKS = {"rel-hm/user-churn": ("transactions", "customer_id", "t_dat", ("price", "sales_channel_id"), "article_id"),
         "rel-amazon/user-churn": ("review", "customer_id", "review_time", ("rating", "verified"), "product_id"),
         "rel-amazon/item-churn": ("review", "product_id", "review_time", ("rating", "verified"), "customer_id")}
NP = 12
BANDS = ((12, 26), (26, 52), (52, 104))
SHARES = (4, 8, 12, 26, 52)


def connect(db, name: str, threads: int = 4):
    import duckdb

    tb, key, tcol, nums, item = TASKS[name]
    ev_df = db.table_dict[tb].df  # noqa: F841
    con = duckdb.connect()
    con.execute(f"set threads={threads}")
    vals = ", ".join(f"cast({c} as double) v{i}" for i, c in enumerate(nums))
    con.execute(f"create table EV as select cast({key} as bigint) e, cast({tcol} as timestamp) ts, {vals}, "
                f"cast({item} as bigint) item from ev_df where {key} is not null")
    return con


def entity_features(con, name: str, t: pd.Timestamp, ents, td_days: float) -> pd.DataFrame:
    """One row per entity in `ents` with an event before t. Reads EV at ts < t only."""
    nums = TASKS[name][3]
    lit = f"TIMESTAMP '{t}'"
    cut = "true" if PLANT else f"x.ts < {lit}"
    con.register("ents", pd.DataFrame({"e": np.asarray(ents, dtype="int64")}).drop_duplicates())
    con.execute(f"""create or replace temp table cur as   -- not "ev": names are case-insensitive, it would shadow EV
        select x.e, x.ts, (epoch({lit}) - epoch(x.ts)) / 86400.0 age,
               cast(floor((epoch({lit}) - epoch(x.ts)) / 86400.0 / {td_days}) as int) p,
               {', '.join(f'x.v{i}' for i in range(len(nums)))}, x.item
        from EV x join ents using (e) where {cut}""")
    per_p = ", ".join(f"count(*) filter (p = {k}) n_p{k}" for k in range(NP))
    bands = ", ".join(f"count(*) filter (p >= {a} and p < {b}) n_b{a}" for a, b in BANDS)
    vals = ", ".join(f"avg(v{i}) v{i}_all, avg(v{i}) filter (p = 0) v{i}_p0" for i in range(len(nums)))
    q1 = f"""select e, count(*) n_all, {per_p}, {bands}, {vals}, count(distinct item) items_all,
               count(distinct item) filter (p = 0) items_p0, min(age) rec, max(age) tenure
        from cur group by e"""
    shares = ", ".join(f"count(*) filter (p < {k}) act{k}" for k in SHARES)
    q2 = f"""with per as (select e, p, count(*) n from cur group by 1, 2)
        select a.e, {shares.replace('p <', 'a.p <')},
               count(*) filter (a.p between 1 and 52) act_hist,
               count(*) filter (a.p between 1 and 52 and b.p is not null) act_pairs,
               min(a.p) filter (c.p is null) run_end
        from per a left join per b on a.e = b.e and b.p = a.p - 1
                   left join per c on a.e = c.e and c.p = a.p + 1
        group by a.e"""
    q3 = """with days as (select e, cast(ts as date) d, min(age) age, count(*) n from cur group by 1, 2),
             g as (select e, age, age - lag(age) over (partition by e order by age) gap from days)
        select e, count(*) n_days, count(*) filter (age <= 28) n_days28, avg(gap) gap_mean, stddev_samp(gap) gap_sd,
               median(gap) gap_med, max(gap) gap_max, min(gap) gap_min,
               arg_min(gap, age) filter (gap is not null) gap_last, avg(n) per_day
        from g join days using (e, age) group by e"""
    out = None
    for q in (q1, q2, q3):
        df = con.execute(q).df().set_index("e")
        out = df if out is None else out.join(df, how="left")
    return out.sort_index().astype("float32")


def derive(X: pd.DataFrame, td_days: float) -> pd.DataFrame:
    X = X.copy()
    for k in SHARES:
        X[f"share{k}"] = X[f"act{k}"] / k
    X["repeat_rate"] = (X["act_pairs"] + 1.0) / (X["act_hist"] + 2.0)
    X["run"] = X["run_end"] + 1                      # active periods in a row, counted from the last period
    X.loc[X["n_p0"].fillna(0) == 0, "run"] = 0
    X["due"] = X["rec"] / X["gap_mean"]
    X["due_med"] = X["rec"] / X["gap_med"]
    X["rec_td"] = X["rec"] / td_days
    X["freq"] = X["n_days"] / X["tenure"].clip(lower=1)
    X["p0_over_p1"] = (X["n_p0"] + 1) / (X["n_p1"] + 1)
    X["p0_share"] = X["n_p0"] / X["n_all"].clip(lower=1)
    return X.astype("float32").add_prefix("cd.")


def cd_features(db, task, name: str, lab: pd.DataFrame, con=None) -> pd.DataFrame:
    con = con or connect(db, name)
    ec, tc = task.entity_col, task.time_col
    td_days = task.timedelta.days
    parts = []
    for t, g in lab.groupby(tc, sort=False):
        F = entity_features(con, name, pd.Timestamp(t), g[ec].to_numpy(), td_days).reindex(g[ec].to_numpy())
        F.index = g.index
        parts.append(F)
    return derive(pd.concat(parts).loc[lab.index], td_days)


def main() -> int:
    import argparse

    sys.path.insert(0, str(ROOT / "harness"))
    from common import load_ctx
    from leakcheck import truncated

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--leakcheck", action="store_true")
    args = ap.parse_args()
    ctx = load_ctx(args.task)
    task, tc, ec = ctx.task, ctx.task.time_col, ctx.task.entity_col
    con = connect(ctx.db, args.task)
    seeds = [pd.Timestamp(ctx.test[tc].iloc[0]), pd.Timestamp(ctx.val[tc].iloc[0])] + \
        [pd.Timestamp(s) for s in np.random.RandomState(0).choice(np.sort(ctx.train[tc].unique())[4:], 2, replace=False)]
    lab = pd.concat([ctx.train, ctx.val, ctx.test], ignore_index=True)
    bad = 0
    for s in seeds:
        ents = lab.loc[lab[tc] == s, ec].to_numpy()[:20000]
        full = entity_features(con, args.task, s, ents, task.timedelta.days)
        c2 = connect(truncated(ctx.db, s, task.entity_table), args.task)
        cut = entity_features(c2, args.task, s, ents, task.timedelta.days)
        c2.close()
        same = full.shape == cut.shape and full.index.equals(cut.index) and bool(
            np.array_equal(full.to_numpy(), cut.to_numpy(), equal_nan=True))
        print(f"[leak] cd {s:%Y-%m-%d} entities={len(full)}/{len(cut)} cols={full.shape[1]} {'ok' if same else 'DIFF'}", flush=True)
        bad += not same
    print(f"[leak] {args.task} cd.*: {'PASS' if bad == 0 else 'FAIL'} ({len(seeds)} seed times, plant={int(PLANT)})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
