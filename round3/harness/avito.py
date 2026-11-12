"""v4: rel-avito with replayed daily anchors and window-bounded features.

    python harness/avito.py --task rel-avito/user-clicks [--arms B3] [--seeds 0,1]
        [--cfgs "150|num_leaves=7,min_child_samples=300"] [--leakcheck] [--no-folds]
        [--halflife 2] [--write-test] [--tag=-hl2] [--threads 4]

The hosted train table has 3 seed times (04-26, 04-30, 05-04) and the db holds 19 days
before the test seed. This script replays the task's label query (`task.make_table`) at
every daily anchor whose 4-day window ends at or before the scoring seed, and builds
per-user features from rows with time < anchor only.

  arms     A0 hosted seed times, window features; A1 daily anchors; A2 = A1 + static-group
           rates; A3 = A2 + ext (category / location priors, category shares, same-weekday
           lags); B0 / B1 the c16_r3 generic matrix on hosted / daily anchors;
           B3 = A3 + the generic matrix (the v4 user-clicks system); A5 / B5 = A3 / B3 + visit
           detail (B5 with --halflife 2 is the v4 user-visits system).
  val      "val model": anchors <= val_ts - 4d, fixed rounds; scored on the hosted val
           table. Val labels are never fit.
  folds    forward folds for tuning: for an eval anchor e, fit on anchors <= e - 4d and
           score the rows of e. FFx = mean over e in 05-01..05-10 without 05-08;
           LF = mean over the folds after val (05-09, 05-10).
  test     "test model": the same config on anchors <= test_ts - 4d. Written with
           --write-test to preds/r6-<arm>/, never scored here.

The db comes from load_ctx (rows <= test_timestamp). --leakcheck rebuilds the features at
k anchors from a db truncated before the anchor and requires equality; RELDFS_PLANT=1
drops the time filter of the click block and must make it FAIL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from backtest import score  # noqa: E402
from common import DATA, board_rows, cached_build, load_ctx  # noqa: E402
from fulldb import rebuilt_labels, same_labels  # noqa: E402
from leakcheck import truncated  # noqa: E402
from run import load_cand  # noqa: E402

WINS = (1, 2, 4, 7, 14)
TAUS = (1, 3, 7)
PLANT = os.environ.get("RELDFS_PLANT") == "1"
STATIC = ("UserAgentID", "UserAgentOSID", "UserDeviceID", "UserAgentFamilyID")
GP = ("r4", "r41", "rn4", "r48", "ctr", "ln4", "vr4", "vrn4")
# new = this file's window features; gp = static-group rates; ext = category / location priors,
# category shares, same-weekday lags; gen = the c16_r3 generic matrix
def _arm(daily, new=False, gp=False, ext=False, vx=False, gen=False):
    return dict(daily=daily, new=new, gp=gp, ext=ext, vx=vx, gen=gen)


ARMS = {"A0": _arm(False, new=True), "A1": _arm(True, new=True), "A2": _arm(True, new=True, gp=True),
        "A3": _arm(True, new=True, gp=True, ext=True),
        "B0": _arm(False, gen=True), "B1": _arm(True, gen=True),
        "B3": _arm(True, new=True, gp=True, ext=True, gen=True),   # B3 = the v4 user-clicks system
        "A5": _arm(True, new=True, gp=True, ext=True, vx=True),
        "B5": _arm(True, new=True, gp=True, ext=True, vx=True, gen=True)}   # B5 = the v4 user-visits system
EXT = ("pr_", "sh_", "wk_", "l37_")
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              verbose=-1)


def connect(db, threads: int):
    """DuckDB tables over the db's frames: SI (searches), IMP (impressions with the user), VS, PR."""
    import duckdb

    T = {k: v.df for k, v in db.table_dict.items()}
    si = T["SearchInfo"]
    q = si["SearchQuery"]
    si_df = pd.DataFrame({  # noqa: F841 (read by duckdb)
        "UserID": si["UserID"], "SearchID": si["SearchID"], "ts": si["SearchDate"],
        "logged": si["IsUserLoggedOn"], "hasq": (q.notna() & (q.astype(str) != "")).astype("int8"),
        "LocationID": si["LocationID"], "CategoryID": si["CategoryID"], "IPID": si["IPID"]})
    ss_df, vs_df, pr_df, ads_df = T["SearchStream"], T["VisitStream"], T["PhoneRequestsStream"], T["AdsInfo"]  # noqa: F841
    con = duckdb.connect()
    con.execute(f"set threads={threads}")
    con.execute("create table SI as select * replace (cast(ts as timestamp) as ts, cast(UserID as bigint) as UserID) "
                "from si_df where UserID is not null")
    cat_df = T["Category"]  # noqa: F841
    con.execute("""create table IMP as
        select s.UserID, x.SearchID, cast(x.SearchDate as timestamp) ts, x.ObjectType ot, x.Position pos,
               x.HistCTR hctr, coalesce(x.IsClick, 0) clk, x.AdID, s.CategoryID scat, s.LocationID sloc
        from ss_df x join SI s using (SearchID)""")
    con.execute("""create table VS as
        select cast(v.UserID as bigint) UserID, v.AdID, cast(v.ViewDate as timestamp) ts, a.CategoryID cat,
               a.Price price, c.ParentCategoryID pcat
        from vs_df v left join ads_df a using (AdID) left join cat_df c on a.CategoryID = c.CategoryID
        where v.UserID is not null""")
    ui_df = T["UserInfo"]  # noqa: F841
    con.execute("create table cat_tb as select CategoryID, ParentCategoryID from cat_df")
    con.execute("create table UI as select cast(UserID as bigint) UserID, "
                + ", ".join(f"coalesce({c}, -1) {c}" for c in STATIC) + " from ui_df")
    con.execute("create table PR as select cast(UserID as bigint) UserID, AdID, cast(PhoneRequestDate as timestamp) ts from pr_df where UserID is not null")
    return con


def cut_src(t: pd.Timestamp):
    """SQL for "rows of a table before t, with age in days". The only place a time filter is written."""
    lit = f"TIMESTAMP '{t}'"
    age = f"(epoch({lit}) - epoch(ts)) / 86400.0"

    def src(tb, leak=False):
        cut = "true" if (leak and PLANT) else f"ts < {lit}"
        return f"(select *, {age} age from {tb} where {cut})"
    return src


def group_features(con, t: pd.Timestamp) -> pd.DataFrame:
    """Per value of each static user column: smoothed rates over the users active before t.

    r4 / r41: share of the users with a search in the 4 days before t who clicked >= 2 / >= 1
    times in those days; rn4: the same among users whose first search is in those days (the
    analogue of a user with no history at the window start); r48: the window before; ctr:
    clicks per context impression; vr4 / vrn4: share of the users with a visit in the last 4
    days who visited >= 2 distinct ads (all / first seen in those days).
    """
    src = cut_src(t)
    base = f"""with u as (select UserID, max(age) first_age, count(*) filter (age <= 4) s4,
                         count(*) filter (age > 4 and age <= 8) s48 from {src('SI')} group by 1),
        k as (select UserID, coalesce(sum(clk) filter (age <= 4), 0) c4,
                     coalesce(sum(clk) filter (age > 4 and age <= 8), 0) c48, sum(clk) c,
                     count(*) filter (ot = 3) i from {src('IMP', leak=True)} group by 1),
        v as (select UserID, max(age) vfirst, count(distinct AdID) filter (age <= 4) va4,
                     count(*) filter (age <= 4) vn4 from {src('VS')} group by 1),
        j as (select ui.*, coalesce(u.s4, 0) s4, coalesce(u.s48, 0) s48, u.first_age,
                     coalesce(k.c4, 0) c4, coalesce(k.c48, 0) c48, coalesce(k.c, 0) c, coalesce(k.i, 0) i,
                     coalesce(v.vn4, 0) vn4, coalesce(v.va4, 0) va4, v.vfirst
              from UI ui left join u using (UserID) left join k using (UserID) left join v using (UserID))"""
    one = """select '{c}' col, {c} g, count(*) filter (s4 > 0) n4, count(*) filter (s4 > 0 and c4 >= 2) k4,
            count(*) filter (s4 > 0 and c4 >= 1) k41, count(*) filter (s4 > 0 and first_age <= 4) nn4,
            count(*) filter (s4 > 0 and first_age <= 4 and c4 >= 2) kn4, count(*) filter (s48 > 0) n48,
            count(*) filter (s48 > 0 and c48 >= 2) k48, sum(c) c, sum(i) i,
            count(*) filter (vn4 > 0) v4, count(*) filter (vn4 > 0 and va4 >= 2) vk4,
            count(*) filter (vn4 > 0 and vfirst <= 4) vnn4,
            count(*) filter (vn4 > 0 and vfirst <= 4 and va4 >= 2) vkn4
          from j group by {c}"""
    G = con.execute(base + " union all ".join(one.format(c=c) for c in STATIC)).df()
    G = G.sort_values(["col", "g"]).reset_index(drop=True)
    tot = G[G.col == STATIC[0]].sum(numeric_only=True)

    def sm(k, n, m):
        return (G[k] + m * tot[k] / max(tot[n], 1.0)) / (G[n] + m)

    out = G[["col", "g"]].copy()
    out["r4"], out["r41"], out["rn4"] = sm("k4", "n4", 30), sm("k41", "n4", 30), sm("kn4", "nn4", 30)
    out["r48"], out["ctr"], out["ln4"] = sm("k48", "n48", 30), sm("c", "i", 5000), np.log1p(G.n4)
    out["vr4"], out["vrn4"] = sm("vk4", "v4", 30), sm("vkn4", "vnn4", 30)
    return out


def ext_queries(t: pd.Timestamp, cats, parents) -> list:
    """Category / location priors, decayed category shares, and same-weekday lags, rows before t only.

    cats and parents come from the static Category table, so the column set does not depend on t.
    """
    src = cut_src(t)
    dow = f"cast(((dayofweek(ts) - dayofweek(TIMESTAMP '{t}') + 7) % 7) < 4 as int)"   # weekdays of the window
    w = "exp(-age / 4)"
    lag = "age > 3 and age <= 7"
    prior = ("select {k} g, sum(clk) / (count(*) filter (ot = 3) + 500.0) ctr, "
             "count(*) filter (ot = 3) * 1.0 / count(distinct SearchID) ctx from " + src("IMP") + " group by 1")
    return [f"""with cs as ({prior.format(k='scat')}), ls as ({prior.format(k='sloc')})
          select UserID, avg(cs.ctr) pr_cctr, avg(cs.ctr) filter (age <= 4) pr_cctr4, avg(cs.ctx) pr_cctx,
            avg(cs.ctx) filter (age <= 4) pr_cctx4, avg(ls.ctr) pr_lctr, avg(ls.ctr) filter (age <= 4) pr_lctr4,
            avg(ls.ctx) pr_lctx, count(*) filter ({lag}) l37_s, avg({dow}) wk_s, avg({dow}) filter (age <= 7) wk_s7,
            {', '.join(f"sum({w}) filter (CategoryID = {k}) / sum({w}) sh_c{k}" for k in cats)}
          from {src('SI')} s left join cs on s.CategoryID = cs.g left join ls on s.LocationID = ls.g
          group by UserID""",
            f"""select UserID, count(*) filter (ot = 3 and {lag}) l37_i, sum(clk) filter ({lag}) l37_c,
            avg({dow}) filter (clk = 1) wk_c from {src('IMP')} group by UserID""",
            f"""select UserID, count(*) filter ({lag}) l37_v, count(distinct AdID) filter ({lag}) l37_va,
            avg({dow}) wk_v, {', '.join(f"sum({w}) filter (pcat = {k}) / sum({w}) sh_vp{k}" for k in parents)}
          from {src('VS')} group by UserID"""]


def vx_queries(t: pd.Timestamp) -> list:
    """Visit detail: distinct ads per active day and per 4-day block aligned to t (the label, replayed)."""
    src = cut_src(t)
    return [f"""select UserID, avg(a) vx_dmean, max(a) vx_dmax, min(a) vx_dmin, count(*) filter (a = 1) vx_d1,
            avg(cast(a = 1 as int)) vx_d1share, avg(n * 1.0 / a) vx_rep
          from (select UserID, cast(ts as date) d, count(distinct AdID) a, count(*) n from {src('VS')} group by 1, 2)
          group by UserID""",
            f"""select UserID, count(*) vx_blocks, count(*) filter (a >= 2) vx_bpos, avg(cast(a >= 2 as int)) vx_brate,
            max(a) filter (b = 2) vx_b2, max(a) filter (b = 3) vx_b3, avg(a) vx_bmean
          from (select UserID, cast(floor(age / 4) as int) b, count(distinct AdID) a from {src('VS')} group by 1, 2)
          group by UserID"""]


def user_features(con, t: pd.Timestamp) -> pd.DataFrame:
    """One row per user with any row before t. Every aggregate reads rows with ts < t only."""
    src = cut_src(t)
    cat = con.execute("select distinct CategoryID c from cat_tb order by 1").df().c.tolist()
    par = con.execute("select distinct ParentCategoryID c from cat_tb where ParentCategoryID is not null order by 1").df().c.tolist()

    def wins(expr, cond, name):
        c = f"{cond} and " if cond else ""
        return ", ".join(f"{expr} filter ({c}age <= {w}) {name}{w}" for w in WINS)

    def lags(expr, name):
        return (f"{expr} filter (age > 4 and age <= 8) {name}l48, "
                f"{expr} filter (age > 8 and age <= 12) {name}l812")

    def decays(val, name):
        return ", ".join(f"sum({val} * exp(-age / {tau})) {name}e{tau}" for tau in TAUS)

    day = "cast(ts as date)"
    qs = [f"""select UserID, {wins('count(*)', '', 's_n')}, count(*) s_nall, {lags('count(*)', 's_')},
            count(distinct {day}) s_days, count(distinct {day}) filter (age <= 4) s_days4,
            count(distinct {day}) filter (age <= 7) s_days7,
            min(age) * 24 s_rec, max(age) s_age, {decays('1', 's_')},
            avg(logged) s_log, avg(logged) filter (age <= 4) s_log4,
            avg(hasq) s_q, avg(hasq) filter (age <= 4) s_q4,
            count(distinct CategoryID) s_ncat, count(distinct LocationID) s_nloc,
            count(distinct IPID) s_nip, count(distinct CategoryID) filter (age <= 4) s_ncat4,
            count(distinct LocationID) filter (age <= 4) s_nloc4,
            avg(hour(ts)) s_hour, stddev_samp(hour(ts)) s_hsd
          from {src('SI')} group by UserID""",
          f"""select UserID, {wins('count(*)', 'ot = 3', 'i_n')}, count(*) filter (ot = 3) i_nall,
            count(*) i_any, count(*) filter (age <= 4) i_any4,
            {wins('sum(clk)', '', 'c_n')}, sum(clk) c_nall, {lags('sum(clk)', 'c_')}, {decays('clk', 'c_')},
            min(age) filter (clk = 1) * 24 c_rec, count(distinct {day}) filter (clk = 1) c_days,
            avg(pos) filter (ot = 3) i_pos, avg(hctr) filter (ot = 3) i_hctr,
            avg(hctr) filter (ot = 3 and age <= 4) i_hctr4, max(hctr) i_hmax,
            sum(hctr) filter (ot = 3) i_hsum, sum(hctr) filter (ot = 3 and age <= 4) i_hsum4,
            sum(hctr * exp(-age / 3)) filter (ot = 3) i_hse3,
            avg(hctr) filter (clk = 1) c_hctr, avg(pos) filter (clk = 1) c_pos,
            count(distinct AdID) filter (clk = 1) c_ads
          from {src('IMP', leak=True)} group by UserID""",
          f"""select UserID, count(*) filter (n3 > 0) q_ctx, count(*) filter (n3 > 0 and age <= 4) q_ctx4,
            count(*) filter (k >= 1) q_c1, count(*) filter (k >= 2) q_c2,
            count(*) filter (k >= 1 and age <= 4) q_c14, max(k) q_kmax
          from (select UserID, SearchID, min(age) age, sum(clk) k, count(*) filter (ot = 3) n3
                from {src('IMP')} group by 1, 2) group by UserID""",
          f"""select UserID, max(k) d_kmax, count(*) filter (k >= 2) d_c2, max(n) d_imax
          from (select UserID, {day} d, sum(clk) k, count(*) filter (ot = 3) n
                from {src('IMP')} group by 1, 2) group by UserID""",
          f"""select UserID, {wins('count(*)', '', 'v_n')}, count(*) v_nall, {lags('count(*)', 'v_')},
            count(distinct AdID) v_ads, count(distinct AdID) filter (age <= 1) v_ads1,
            count(distinct AdID) filter (age <= 2) v_ads2, count(distinct AdID) filter (age <= 4) v_ads4,
            count(distinct AdID) filter (age <= 7) v_ads7,
            count(distinct AdID) filter (age > 4 and age <= 8) v_adsl48,
            min(age) * 24 v_rec, max(age) v_age, {decays('1', 'v_')},
            count(distinct {day}) v_days, count(distinct {day}) filter (age <= 4) v_days4,
            count(distinct cat) v_ncat, avg(price) v_price, median(price) v_pmed
          from {src('VS')} group by UserID""",
          f"""select UserID, count(*) filter (age <= 4) p_n4, count(*) filter (age <= 14) p_n14,
            count(*) p_nall, min(age) * 24 p_rec, count(distinct AdID) p_ads
          from {src('PR')} group by UserID"""]
    out = None
    for sql in qs + ext_queries(t, [int(c) for c in cat], [int(c) for c in par]) + vx_queries(t):
        df = con.execute(sql).df().set_index("UserID")
        out = df if out is None else out.join(df, how="outer")
    return out.astype("float32")


def derive(X: pd.DataFrame) -> pd.DataFrame:
    keep_nan = [c for c in X.columns if c.endswith(("_rec", "_age", "_hour", "_hsd", "_pos", "_hctr", "_hctr4",
                                                    "_hmax", "_price", "_pmed", "_log", "_log4", "_q", "_q4"))]
    keep_nan += [c for c in X.columns if c.startswith(("pr_", "wk_", "vx_"))]
    X = X.copy()
    cnt = [c for c in X.columns if c not in keep_nan]
    X[cnt] = X[cnt].fillna(0)
    hist = X[["s_age", "v_age"]].max(axis=1)
    X["hist"] = hist
    X["ctr_all"] = X.c_nall / (X.i_nall + 50)
    X["ctr4"] = X.c_n4 / (X.i_n4 + 50)
    X["ctr14"] = X.c_n14 / (X.i_n14 + 50)
    X["c_per_s"] = X.c_nall / (X.s_nall + 5)
    X["c_per_day"] = X.c_nall / (X.s_days + 1)
    X["c_lift"] = X.c_nall - X.i_hsum
    X["c_lift4"] = X.c_n4 - X.i_hsum4
    X["s_per_day"] = X.s_nall / X.s_days.clip(lower=1)
    X["s_frac4"] = X.s_n4 / (X.s_nall + 1)
    X["s_frac1"] = X.s_n1 / (X.s_n4 + 1)
    X["s_act"] = X.s_days / np.ceil(hist.fillna(1).clip(lower=1))
    X["v_per_day"] = X.v_nall / X.v_days.clip(lower=1)
    X["v_frac4"] = X.v_n4 / (X.v_nall + 1)
    X["v_ads_per_day"] = X.v_ads / X.v_days.clip(lower=1)
    X["c_frac4"] = X.c_n4 / (X.c_nall + 1)
    X["i_per_s"] = X.i_nall / (X.s_nall + 1)
    return X.astype("float32")


class Data:
    def __init__(self, name: str, threads: int):
        self.name = name
        self.ctx = ctx = load_ctx(name)
        self.task = task = ctx.task
        self.db = db = ctx.db
        self.test_ts = pd.Timestamp(ctx.dataset.test_timestamp)
        self.val_ts = pd.Timestamp(ctx.dataset.val_timestamp)
        self.td = task.timedelta
        for tb in db.table_dict.values():   # a mirror can hold the raw files; the cut must hold
            if tb.time_col:
                assert tb.df[tb.time_col].max() <= self.test_ts, "db not cut at test_timestamp"
        self.tr, self.va, self.te = ctx.train, ctx.val, ctx.test
        tc = task.time_col
        self.hosted = sorted(pd.to_datetime(self.tr[tc].unique()))
        first = min(self.hosted)
        self.anchors = list(pd.date_range(first, self.test_ts - self.td, freq="D"))
        assert self.val_ts in self.anchors
        self.lab = rebuilt_labels(task, db, self.anchors)
        assert same_labels(task, self.lab[self.lab[tc].isin(self.hosted)], self.tr), "make_table != hosted train"
        assert same_labels(task, self.lab[self.lab[tc] == self.val_ts], self.va), "make_table != hosted val"
        self.con = connect(db, threads)
        self.static = db.table_dict["UserInfo"].df.set_index("UserID")[list(STATIC)].astype("float32")
        self.src = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]

    def feats_at(self, t: pd.Timestamp, con=None, cache=True) -> pd.DataFrame:
        f = DATA / "fcache" / f"avito-{self.src}-{'plant-' if PLANT else ''}{t:%Y%m%d}.parquet"
        if cache and con is None and f.exists():
            return pd.read_parquet(f)
        X = user_features(con or self.con, t)
        if cache and con is None:
            f.parent.mkdir(parents=True, exist_ok=True)
            X.to_parquet(f)
        return X

    def groups_at(self, t: pd.Timestamp) -> pd.DataFrame:
        f = DATA / "fcache" / f"avito-{self.src}-{'plant-' if PLANT else ''}gp-{t:%Y%m%d}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        G = group_features(self.con, t)
        f.parent.mkdir(parents=True, exist_ok=True)
        G.to_parquet(f)
        return G

    def matrix(self, lab: pd.DataFrame) -> pd.DataFrame:
        """Row-aligned feature matrix for a label frame (entity, time)."""
        tc, ec = self.task.time_col, self.task.entity_col
        S = self.static.reindex(lab[ec].to_numpy())
        S.index = lab.index
        parts, gparts = [], []
        for t, g in lab.groupby(tc, sort=False):
            F = self.feats_at(pd.Timestamp(t))
            X = F.reindex(g[ec].to_numpy())
            X.index = g.index
            parts.append(X)
            G = self.groups_at(pd.Timestamp(t))
            cols = []
            for c in STATIC:
                Gc = G[G.col == c].set_index("g")[list(GP)]
                P = Gc.reindex(S.loc[g.index, c].fillna(-1).to_numpy())
                P.index = g.index
                P.columns = [f"gp_{c}_{x}" for x in GP]
                cols.append(P)
            gparts.append(pd.concat(cols, axis=1))
        X = derive(pd.concat(parts).loc[lab.index])
        return pd.concat([X, S, pd.concat(gparts).loc[lab.index].astype("float32")], axis=1)


def generic(D: Data, lab: pd.DataFrame, tag: str) -> pd.DataFrame:
    """The current system's (c16_r3) feature matrix for any label frame, columns prefixed "gen__"."""
    mod = load_cand("c16_r3")
    plan = mod.F.make_plan(D.db, D.task)
    mod.ID_COLS[:] = mod.id_cols(D.db, D.task)
    parts = []
    for t, g in lab.groupby(D.task.time_col, sort=False):   # one seed time per build: the engine's
        X = cached_build(mod.F, mod.HERE / "features.py", D.db, D.task, plan, g.reset_index(drop=True),  # duckdb limits
                         f"{D.name}|{tag}|r6", 2, extra=(mod.HERE / "domain.py",))
        X.index = g.index
        parts.append(X)
    X = pd.concat(parts).loc[lab.index]
    X = X[mod.select_cols(X.columns, mod.variant_cfg(D.name, "base"))]
    odd = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
    assert not odd, f"non-numeric generic columns: {odd[:8]}"
    return X.add_prefix("gen__")


def leakcheck(D: Data, k: int) -> int:
    tc, ec = D.task.time_col, D.task.entity_col
    seeds = [D.test_ts, D.val_ts] + list(np.random.RandomState(0).choice(D.anchors, k - 2, replace=False))
    bad = 0
    for t in seeds:
        t = pd.Timestamp(t)
        full = user_features(D.con, t)
        con = connect(truncated(D.db, t, D.task.entity_table), 2)
        cut = user_features(con, t)
        gf, gc = group_features(D.con, t), group_features(con, t)
        gsame = gf.shape == gc.shape and gf[["col", "g"]].equals(gc[["col", "g"]]) and bool(
            np.array_equal(gf[list(GP)].to_numpy(), gc[list(GP)].to_numpy(), equal_nan=True))
        same = gsame and full.shape == cut.shape and full.index.equals(cut.index) and bool(
            np.allclose(full.to_numpy(), cut.to_numpy(), equal_nan=True, rtol=0, atol=0))
        nbad = 0 if same else int((~np.isclose(full.reindex(cut.index).to_numpy(), cut.to_numpy(),
                                               equal_nan=True, rtol=0, atol=0)).any(axis=0).sum()) \
            if full.shape[1] == cut.shape[1] else -1
        con.close()
        print(f"[leak] {t:%Y-%m-%d} users={len(full)}/{len(cut)} cols={full.shape[1]} groups={len(gf)}/{len(gc)} "
              f"{'ok' if same else f'DIFF in {nbad} user columns, group table same={gsame}'}", flush=True)
        bad += not same
    print(f"[leak] {D.name}: {'PASS' if bad == 0 else 'FAIL'} ({len(seeds)} seed times, plant={int(PLANT)})")
    return 1 if bad else 0


def fit(X, y, rounds, seed, threads, params, w=None):
    import lightgbm as lgb

    p = dict(PARAMS, **params, seed=seed, num_threads=threads)
    return lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=rounds)


def parse_cfg(text: str):
    """"<rounds>|k=v,k=v" -> (rounds, {k: v})."""
    r, _, kv = text.partition("|")
    out = {}
    for item in filter(None, kv.split(",")):
        k, v = item.split("=")
        out[k] = float(v) if "." in v else int(v)
    return int(r), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--arms", default="B3")
    ap.add_argument("--cfgs", default="150|num_leaves=7,min_child_samples=300", help='";"-separated "<rounds>|k=v,..." LightGBM overrides')
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iter", default="12")
    ap.add_argument("--leakcheck", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-folds", action="store_true", help="official val only")
    ap.add_argument("--write-test", action="store_true")
    ap.add_argument("--halflife", type=float, default=0.0, help="recency weight half-life in days (0 = uniform)")
    ap.add_argument("--tag", default="", help="suffix of the preds/ folder")
    args = ap.parse_args()
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    D = Data(args.task, args.threads)
    task, tc, yc = D.task, D.task.time_col, D.task.target_col
    print(f"[r6] {D.name} anchors {D.anchors[0]:%m-%d}..{D.anchors[-1]:%m-%d} ({len(D.anchors)}) "
          f"hosted {[f'{t:%m-%d}' for t in D.hosted]} rebuilt rows {len(D.lab)} ({time.time() - t0:.0f}s)", flush=True)
    if args.leakcheck:
        return leakcheck(D, args.k)

    Xall = D.matrix(D.lab)
    Xva, Xte = D.matrix(D.va), D.matrix(D.te)
    y = D.lab[yc].to_numpy().astype(int)
    if any(ARMS[a]["gen"] for a in args.arms.split(",")):
        Xall = pd.concat([Xall, generic(D, D.lab, "anchors")], axis=1)
        Xva = pd.concat([Xva, generic(D, D.va, "val")], axis=1)
        Xte = pd.concat([Xte, generic(D, D.te, "test")], axis=1)
        print(f"[r6] with generic columns: {Xall.shape[1]} ({time.time() - t0:.0f}s)", flush=True)
    cold = lambda X: float(X["hist"].isna().mean())  # noqa: E731
    print(f"[r6] features {Xall.shape[1]}; cold rows: anchors {cold(Xall):.3f} val {cold(Xva):.3f} "
          f"test {cold(Xte):.3f} ({time.time() - t0:.0f}s)", flush=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    brow = [board_rows(D.name).get(k, "") for k in ("LightGBM", "GNN", "Kapso (Leeroo Team)")]
    evals = [a for a in D.anchors if a - D.td >= D.anchors[1] and a != D.val_ts]
    late = [a for a in evals if a > D.val_ts]
    rows = []
    for arm in args.arms.split(","):
        spec = ARMS[arm]
        pool = D.anchors if spec["daily"] else D.hosted + [D.val_ts]   # hosted at test = train + val
        kind = lambda c: ("gen" if c.startswith("gen__") else "gp" if c.startswith("gp_")  # noqa: E731
                          else "ext" if c.startswith(EXT) else "vx" if c.startswith("vx_") else "new")
        use = [c for c in Xall.columns if spec[kind(c)]]

        def rows_upto(last):
            m = D.lab[tc].isin([a for a in pool if a <= last]).to_numpy()
            w = None
            if args.halflife:   # weight 0.5 per `halflife` days before the last fitted anchor
                age = (D.lab.loc[m, tc].max() - D.lab.loc[m, tc]).dt.days.to_numpy()
                w = 0.5 ** (age / args.halflife)
            return Xall.loc[m, use], y[m], w

        for ci, ctext in enumerate(args.cfgs.split(";")):
            rounds, params = parse_cfg(ctext)
            tag = f"{arm} cfg={ctext}" + (f" hl={args.halflife:g}" if args.halflife else "")
            pv, pt = [], []
            for seed in seeds:
                ff = {}
                if spec["daily"] and not args.no_folds:
                    for e in evals:
                        me = (D.lab[tc] == e).to_numpy()
                        Xf, yf, wf = rows_upto(e - D.td)
                        m = fit(Xf, yf, rounds, seed, args.threads, params, wf)
                        ff[e] = roc_auc_score(y[me], m.predict(Xall.loc[me, use]))
                Xf, yf, wf = rows_upto(D.val_ts - D.td)
                mv = fit(Xf, yf, rounds, seed, args.threads, params, wf)
                p = mv.predict(Xva[use])
                pv.append(p)
                s = score(task, D.va, p)
                ffx = float(np.mean(list(ff.values()))) if ff else float("nan")
                lf = float(np.mean([ff[e] for e in late])) if ff else float("nan")
                print(f"[r6] {D.name} {tag} seed={seed} FFx={ffx:.4f} LF={lf:.4f} val={s:.4f} | "
                      + " ".join(f"{e:%m-%d}={v:.4f}" for e, v in ff.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
                rows.append([f"r6:{arm}", "val", f"{s:.4f}", f"cfg={ctext} seed={seed} FFx={ffx:.4f} LF={lf:.4f}"])
                if args.write_test:
                    Xf, yf, wf = rows_upto(D.test_ts - D.td)
                    pt.append(fit(Xf, yf, rounds, seed, args.threads, params, wf).predict(Xte[use]))
                if seed == seeds[0] and ci == 0:
                    warm = Xva["hist"].notna().to_numpy()
                    yva = D.va[yc].to_numpy().astype(int)
                    print(f"[r6]   val warm auc={roc_auc_score(yva[warm], p[warm]):.4f} cold auc="
                          f"{roc_auc_score(yva[~warm], p[~warm]):.4f} mean pred warm {p[warm].mean():.4f} cold {p[~warm].mean():.4f}")
                    imp = pd.Series(mv.feature_importance("gain"), index=use).sort_values(ascending=False)
                    print("[r6]   top gain: " + ", ".join(f"{k}={v / imp.sum():.3f}" for k, v in imp.head(12).items()))
            s = score(task, D.va, np.mean(pv, axis=0))
            print(f"[r6] {D.name} {tag}: val pooled over {len(seeds)} seeds = {s:.4f}", flush=True)
            rows.append([f"r6:{arm}", "val", f"{s:.4f}", f"cfg={ctext} {len(seeds)}-seed pooled prediction"])
            if args.write_test:
                from relbench.submit import write_prediction_table

                pred = np.mean(pt, axis=0)
                assert len(pred) == len(D.te) and np.all((pred >= 0) & (pred <= 1))
                d = ROOT / "preds" / f"r6-{arm}{args.tag}"
                d.mkdir(parents=True, exist_ok=True)
                write_prediction_table(task, pred, d / f"{D.name.replace('/', '__')}.csv")
                print(f"[r6] {D.name} {tag}: test CSV written to preds/r6-{arm}/ (not scored); "
                      f"mean pred test cold {pred[Xte['hist'].isna().to_numpy()].mean():.4f} "
                      f"warm {pred[Xte['hist'].notna().to_numpy()].mean():.4f}")
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = ROOT / "runs" / f"it{args.iter}" / f"rows-{D.name.replace('/', '__')}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for cand, split, val, note in rows:
            f.write("\t".join([now, args.iter, cand, D.name, split, "roc_auc", val, *map(str, brow), "", "",
                               f"harness/avito.py src={D.src} {note}"]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
