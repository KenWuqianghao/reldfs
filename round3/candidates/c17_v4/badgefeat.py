"""v4: rel-stack family bd.* (user-badge and user-engagement).

86% of the users who get a badge in a 91-day window have no post or comment in the 91 days
before it. Their badges are first-action badges of new accounts (Student, Supporter, Editor,
Autobiographer, Scholar, Informed, Teacher), the yearly Yearling badge (39% within two weeks
of the account anniversary), and view-count badges on old questions (Popular -> Notable ->
Famous). bd.* gives the model that state, from rows before the seed only:
  account age, days to the next account anniversary, the anniversary-year reputation proxy;
  per badge name: count and days since the last one; badges in the same 91 days one and two
  years earlier; posts, comments, edits and votes received in 7 / 30 / 91 / 365-day windows;
  the questions that do not yet have each view-count badge.

    python candidates/c17_v4/badgefeat.py --leakcheck    (RELDFS_PLANT=1 must FAIL)
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
# the badge names with the most awards in the year before the val seed (fixed list)
NAMES = ("Yearling", "Student", "Supporter", "Popular Question", "Editor", "Autobiographer", "Scholar",
         "Informed", "Notable Question", "Teacher", "Announcer", "Nice Answer", "Nice Question",
         "Famous Question", "Custodian", "Revival", "Commentator", "Necromancer", "Enlightened", "Curious",
         "Critic", "Good Answer", "Promoter", "Excavator", "Good Question", "Citizen Patrol", "Booster",
         "Favorite Question", "Benefactor", "Enthusiast")
ONCE = ("Student", "Supporter", "Editor", "Autobiographer", "Scholar", "Informed", "Teacher", "Commentator",
        "Critic", "Curious", "Citizen Patrol", "Excavator", "Promoter", "Benefactor", "Enthusiast")
WINS = (7, 30, 91, 365)


def _id(name: str) -> str:
    return name.lower().replace(" ", "_")


def connect(db, threads: int = 4):
    import duckdb

    T = {k: v.df for k, v in db.table_dict.items()}
    b_df, u_df, p_df, v_df, c_df, h_df = (T[k] for k in ("badges", "users", "posts", "votes", "comments", "postHistory"))  # noqa: F841
    con = duckdb.connect()
    con.execute(f"set threads={threads}")
    con.execute("create table B as select cast(UserId as bigint) u, cast(Name as varchar) bname, Class cls, "
                "cast(Date as timestamp) ts from b_df where UserId is not null")
    con.execute("create table U as select cast(Id as bigint) u, cast(CreationDate as timestamp) created from u_df")
    con.execute("create table P as select cast(Id as bigint) id, cast(OwnerUserId as bigint) u, PostTypeId pt, "
                "cast(CreationDate as timestamp) ts from p_df where OwnerUserId is not null")
    con.execute("create table V as select p.u, p.pt, p.id pid, v.VoteTypeId vt, cast(v.CreationDate as timestamp) ts "
                "from v_df v join P p on v.PostId = p.id")
    con.execute("create table C as select cast(UserId as bigint) u, cast(CreationDate as timestamp) ts "
                "from c_df where UserId is not null")
    con.execute("create table H as select cast(UserId as bigint) u, PostHistoryTypeId ht, "
                "cast(CreationDate as timestamp) ts from h_df where UserId is not null")
    return con


def user_features(con, t: pd.Timestamp) -> pd.DataFrame:
    """One row per user created before t. Every timed table is read at ts < t only."""
    lit = f"TIMESTAMP '{t}'"
    age = f"(epoch({lit}) - epoch(ts)) / 86400.0"

    def src(tb, leak=False):
        cut = "true" if (leak and PLANT) else f"ts < {lit}"
        return f"(select *, {age} age from {tb} where {cut})"

    def wins(expr, cond, name):
        c = f"{cond} and " if cond else ""
        return ", ".join(f"{expr} filter ({c}age <= {w}) {name}{w}" for w in WINS)

    per_name = ", ".join(f"count(*) filter (bname = '{n}') bn_{_id(n)}, min(age) filter (bname = '{n}') bl_{_id(n)}"
                         for n in NAMES)
    rep = "sum(case when vt = 2 then 10 when vt = 1 then 15 when vt = 3 then -2 else 0 end)"
    qs = [f"""select u, (epoch({lit}) - epoch(created)) / 86400.0 acc_age from U where created < {lit}""",
          f"""select u, count(*) b_n, {wins('count(*)', '', 'b_n')}, min(age) b_rec, count(distinct bname) b_names,
            count(*) filter (cls = 1) b_gold, count(*) filter (cls = 2) b_silver,
            count(*) filter (age >= 274 and age < 365) b_seas1, count(*) filter (age >= 639 and age < 730) b_seas2,
            count(*) filter (bname = 'Yearling' and age >= 274 and age < 365) y_seas1,
            count(*) filter (bname = 'Yearling' and age >= 639 and age < 730) y_seas2, {per_name}
          from {src('B', leak=True)} group by u""",
          f"""select u, count(*) filter (pt = 1) q_n, count(*) filter (pt = 2) a_n,
            {wins('count(*)', 'pt = 1', 'q_n')}, {wins('count(*)', 'pt = 2', 'a_n')}, min(age) p_rec,
            max(age) filter (pt = 1) q_old, min(age) filter (pt = 1) q_new,
            count(*) filter (pt = 1 and age > 365) q_gt1y
          from {src('P')} group by u""",
          f"""select u, count(*) filter (vt = 2 and pt = 1) uq_n, count(*) filter (vt = 2 and pt = 2) ua_n,
            count(*) filter (vt = 1) acc_n, count(*) filter (vt = 3) down_n, count(*) filter (vt = 5) fav_n,
            {wins('count(*)', 'vt = 2 and pt = 1', 'uq_n')}, {wins('count(*)', 'vt = 2 and pt = 2', 'ua_n')},
            {rep} rep_all, {wins(rep, '', 'rep_')}, min(age) filter (vt = 2) up_rec
          from {src('V')} group by u""",
          f"select u, count(*) c_n, {wins('count(*)', '', 'c_n')}, min(age) c_rec from {src('C')} group by u",
          f"select u, count(*) h_n, {wins('count(*)', '', 'h_n')}, min(age) h_rec from {src('H')} group by u",
          # reputation proxy since the last account anniversary (Yearling needs 200 in the year)
          f"""select v.u, {rep} rep_year from {src('V')} v join U on U.u = v.u
          where v.age <= ((epoch({lit}) - epoch(U.created)) / 86400.0) % 365.25 group by v.u"""]
    # per-post score (up - down votes before t) against the badge thresholds 1 / 10 / 25 / 100
    posts = f"""(select p.u, p.pt, p.id, (epoch({lit}) - epoch(p.ts)) / 86400.0 page,
                   coalesce(sum(case when v.vt = 2 then 1 when v.vt = 3 then -1 else 0 end), 0) score,
                   count(*) filter (v.vt = 2 and v.age <= 365) up365, count(*) filter (v.vt = 5) fav
            from P p left join {src('V')} v on v.pid = p.id where p.ts < {lit} group by all)"""
    thr = []
    for pt, tag in ((1, "pq"), (2, "pa")):
        f = f"pt = {pt}"
        thr += [f"count(*) filter ({f} and score <= 0) {tag}_s0", f"max(score) filter ({f}) {tag}_max",
                f"sum(page) filter ({f}) {tag}_expo", f"sum(up365) filter ({f}) {tag}_up365",
                f"count(*) filter ({f} and score between 7 and 9) {tag}_near10",
                f"count(*) filter ({f} and score between 20 and 24) {tag}_near25",
                f"count(*) filter ({f} and score between 80 and 99) {tag}_near100",
                f"count(*) filter ({f} and score >= 10) {tag}_ge10", f"count(*) filter ({f} and score >= 25) {tag}_ge25",
                f"max(up365 * 1.0 / (10 - score)) filter ({f} and score between 1 and 9) {tag}_r10",
                f"max(up365 * 1.0 / (25 - score)) filter ({f} and score between 10 and 24) {tag}_r25",
                f"max(up365) filter ({f} and score <= 0) {tag}_v0"]
    qs.append(f"select u, {', '.join(thr)}, count(*) filter (pt = 1 and fav between 18 and 24) pq_fav25 "
              f"from {posts} group by u")
    # view-badge projection: the question matched to the last Popular (Notable) badge reached 1000 (2500)
    # views at age A, so 2500 (10000) views are due near age 2.5 A (4 A)
    for name, nxt, mult in (("Popular Question", "not", 2.5), ("Notable Question", "fam", 4.0)):
        qs.append(f"""select b.u, (epoch({lit}) - epoch(max(q.ts))) / 86400.0
                   - {mult} * (epoch(any_value(b.blast)) - epoch(max(q.ts))) / 86400.0 proj_{nxt},
                 (epoch(any_value(b.blast)) - epoch(max(q.ts))) / 86400.0 reach_{nxt}
          from (select u, max(ts) blast from {src('B')} where bname = '{name}' group by u) b
          join P q on q.u = b.u and q.pt = 1 and q.ts < b.blast group by b.u""")
    out = None
    for sql in qs:
        df = con.execute(sql).df().set_index("u")
        out = df if out is None else out.join(df, how="left")
    return out.astype("float32")


def derive(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    cnt = [c for c in X.columns if not c.endswith(("_rec", "_old", "_new", "acc_age", "_max", "_r10", "_r25", "_v0"))
           and not c.startswith(("bl_", "proj_", "reach_"))]
    X[cnt] = X[cnt].fillna(0)
    ann = X["acc_age"] % 365.25
    X["to_anniv"] = 365.25 - ann
    X["anniv_in"] = (X["to_anniv"] <= 91).astype("float32")
    X["acc_years"] = np.floor(X["acc_age"] / 365.25)
    X["once_left"] = sum((X[f"bn_{_id(n)}"] == 0).astype("float32") for n in ONCE)
    X["q_unpop"] = X["q_n"] - X["bn_popular_question"]
    X["q_unnot"] = X["bn_popular_question"] - X["bn_notable_question"]
    X["q_unfam"] = X["bn_notable_question"] - X["bn_famous_question"]
    X["y_rate"] = X["bn_yearling"] / X["acc_years"].clip(lower=1)
    X["y_due"] = X["anniv_in"] * (X["rep_year"] >= 200).astype("float32")
    X["rep_gap"] = (200 - X["rep_year"]).clip(lower=0)
    X["last_any"] = X[["b_rec", "p_rec", "c_rec", "h_rec", "up_rec"]].min(axis=1)
    # a projection only counts while the next badge is still missing
    X.loc[X["bn_notable_question"] >= X["bn_popular_question"], "proj_not"] = np.nan
    X.loc[X["bn_famous_question"] >= X["bn_notable_question"], "proj_fam"] = np.nan
    X["student_open"] = ((X["bn_student"] == 0) & (X["pq_s0"] > 0)).astype("float32")
    X["teacher_open"] = ((X["bn_teacher"] == 0) & (X["pa_s0"] > 0)).astype("float32")
    X["niceq_open"] = X["pq_ge10"] - X["bn_nice_question"]
    X["nicea_open"] = X["pa_ge10"] - X["bn_nice_answer"]
    X["pq_uprate"] = X["pq_up365"] / X["q_n"].clip(lower=1)
    return X.astype("float32").add_prefix("bd.")


def bd_features(db, task, lab: pd.DataFrame, con=None) -> pd.DataFrame:
    """Row-aligned (index = lab.index) bd.* matrix."""
    con = con or connect(db)
    ec, tc = task.entity_col, task.time_col
    parts = []
    for t, g in lab.groupby(tc, sort=False):
        F = user_features(con, pd.Timestamp(t)).reindex(g[ec].to_numpy())
        F.index = g.index
        parts.append(F)
    return derive(pd.concat(parts).loc[lab.index])


def main() -> int:
    sys.path.insert(0, str(ROOT / "harness"))
    from common import load_ctx
    from leakcheck import truncated

    ctx = load_ctx("rel-stack/user-badge")
    task, tc = ctx.task, ctx.task.time_col
    con = connect(ctx.db)
    seeds = [pd.Timestamp(ctx.test[tc].iloc[0]), pd.Timestamp(ctx.val[tc].iloc[0])] + \
        [pd.Timestamp(s) for s in np.random.RandomState(0).choice(np.sort(ctx.train[tc].unique())[8:], 3, replace=False)]
    bad = 0
    for s in seeds:
        full = user_features(con, s)
        c2 = connect(truncated(ctx.db, s, task.entity_table))
        cut = user_features(c2, s)
        c2.close()
        same = full.shape == cut.shape and full.index.equals(cut.index) and bool(
            np.array_equal(full.to_numpy(), cut.to_numpy(), equal_nan=True))
        print(f"[leak] bd {s:%Y-%m-%d} users={len(full)}/{len(cut)} cols={full.shape[1]} {'ok' if same else 'DIFF'}", flush=True)
        bad += not same
    print(f"[leak] rel-stack/user-badge bd.*: {'PASS' if bad == 0 else 'FAIL'} ({len(seeds)} seed times, plant={int(PLANT)})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
