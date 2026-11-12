"""c15_dom = c14_fam3 features + hand-written per-dataset dom.* columns (domain.py), appended last (round 2).

(c13_fam docstring follows.) c13_fam = c12_fine (H10 ft.*) + H9 gc.* (c8_gchild) + H7 pe.* (c11_penr): per-task family switches (H13, EXPERIMENTS.md 6b-6c).

The feature code of each family is copied verbatim from its candidate. Column order: children,
H1 track, gc.* (H9), pe.* (H7), own columns, ft.* (H10). Dropping a family's prefix therefore
gives exactly that family-less candidate's matrix in its own column order (e.g. dropping
"gc." and "pe." gives c12_fine; also dropping "ft." gives c9_tie). model.py picks, per task, which
of gc.* / pe.* stay on (TASK_FAMILIES); that choice is made on the backtests only (EXPERIMENTS.md 6b).

(c12_fine docstring follows.) c12_fine = c9_tie + H10 "ft.*": fine-grained time on the entity's own timed children (EXPERIMENTS.md 6a).

For every timed child C of the entity (the plan's children with a time column), C's rows are
first reduced to distinct (entity, day) pairs with a row count, and a day d enters a label row
(e, t) only when d + 1 day <= t (every row of that day is strictly before t; RelBench cutoffs
are all at midnight, so this is exactly C.time < t). Features, prefix ft.<C>.<fk>.:
  n_7d, n_30d     rows in the last 7 / 30 days (skipped when equal to an existing
                  w x timedelta window, e.g. 30 days on f1, 7 days on event and hm)
  ad_<k>d         distinct active days in the last 7, 30, 1 x td and 4 x td days (deduplicated)
  ad_all          distinct active days, all time
  gap_mean        mean gap between consecutive active days = (last - first) / (ad_all - 1)
  gap_last        gap between the two most recent active days
All windows and choices were fixed before any val number. The ft.* columns are appended AFTER
every c9_tie column, so model variant "noft" (drop_prefix "ft.") is c9_tie in the same column
order. Planted leak (leakcheck only): RELDFS_PLANT_FT=1 lets a day enter when d < t + td.

(c9_tie docstring follows.) c9_tie = c7_guard + a deterministic tie-break for H1's "plast" (EXPERIMENTS.md 3b; infrastructure).

plast is now arg_max over (link time, parent id) instead of link time alone. With tied link
times (rel-hm: several articles bought on one day; some avito visits and f1 races) duckdb picked
an arbitrary parent, two builds disagreed, and the leakcheck FAILED on rel-hm
(trk.article.*.plast; a tie, not a leak). Features equal c8_gchild's minus its gc.* columns
(the "nogc" ablation of EXPERIMENTS.md 3b), in the same order.

(c7_guard docstring follows.) c7_guard = c4_det + a scale-up (fan-out) guard for the H1 track-record paths.

Guard (EXPERIMENTS.md 2b). Every decision is taken in make_plan from full-db statistics (like the
category vocabulary), so it is identical for every build and for the leakcheck's truncated
db; every cap is leak-safe by construction:
  * LC_CAP: at most LC_CAP of the entity's own link rows per label row, most recent first
    (only rows with C.time < t are ranked).
  * p_cap: at most K distinct parents per label row (most recent last link < t first), chosen
    so that the parent-level block stays under GUARD_Q_BYTES per label row.
  * bucket: path times (G.time) rounded UP to the next midnight when a path's event table
    would pass GUARD_BYTES; a rounded-up time is >= the true one, so "< t" stays strict.
  * edge_cap: path B keeps, per (parent, timedelta bucket of the first link) and per
    (entity e', bucket), only the K earliest links (by first link time, then id). Earliest-K
    within a bucket is truncation-invariant: the kept links with time < t are the same
    whether or not later links exist.
  * drop: a path still over budget at K = 1 is skipped (logged).
On f1, trial and event every guard is a no-op (features identical to c4_det).

(c4_det docstring follows.) c4_det = c2_track features, returned as float32 for determinism.

Automated, time-aware relational feature synthesis (deep-feature-synthesis style).

Everything is derived from the schema graph in the RelBench manifest; nothing is task- or
dataset-specific. For label rows (entity id e, cutoff time t):

  own      the entity row's own columns: numbers, low/mid-cardinality categories (codes),
           dates as days before t; the entity's parents (fkeys) contribute their columns too.
  child    for every table C holding a fkey to the entity table, aggregate the rows of C
           with C.time < t (STRICTLY before the cutoff) for that entity:
             count over all time and over windows {1, 4, 16} x task timedelta,
             days since last / first row,
             per numeric column: mean / min / max / std over all time, mean over the
             4 x timedelta window, and the value of the most recent row,
             per categorical column: counts of its 8 most frequent values over all time.
           Tables with no time column are aggregated in full (they are static).

  track    (H1, EXPERIMENTS.md 1b) 2-hop "track record" through STATIC parents P (tables without a
           time column: sponsors, facilities, constructors, ...). For every P the entity
           links to through a link table C (child of E holding a fkey to P; the largest such
           C per P), at cutoff t:
             A  P's own children G (fkey to P, timed), rows with G.time < t:
                counts (all, 4 x timedelta window) and mean of each numeric column;
             B  P's other entities E' (through C rows with C.time < t) and their children G
                (fkey to E, timed), rows with G.time < t: same statistics
                (the sponsor's / facility's / constructor's record on earlier entities).
           A row of B enters at max(first C time for (P, E'), G.time), so it counts only
           when both hops are strictly before t. Per label row the P-level values are
           reduced over the entity's own C rows (C.time < t): mean, max and, when C rows of
           one entity carry different times, the value at the most recent C row ("last").
           Implemented as per-parent cumulative sums + ASOF joins (entity-keyed; no
           cutoff x table join).

Time leakage: the only rows that reach a feature satisfy C.time < t. harness/leakcheck.py
re-derives the features of sampled cutoffs from a database truncated at t and asserts
equality. A plan (which columns / which top values) is built once from the task's db;
it holds vocabulary only, never values.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get("RELDFS_DATA") or Path(__file__).resolve().parents[2] / "data")

import domain as _domain   # round 2 (c15_dom): hand-written per-dataset features, prefix dom.

TOPK = 8
MAX_CAT_CARD = 200     # child categorical columns with more distinct values are skipped
OWN_CAT_CARD = 5000    # entity/parent object columns: category codes up to this card
WINDOWS = (1, 4, 16)   # multiples of the task timedelta
TRACK_WIN = 4          # H1 recent-record window, in task timedeltas
TRACK_MAX_NUM = 12     # numeric columns per G table in the track record
GUARD_BYTES = 0.25e9   # guard: max bytes of one path's event table (rows x 8 x (3 + 2 k)); f1/trial max 33 MB
GUARD_Q_BYTES = 6000   # guard: max bytes per label row of one parent's pre-reduction block
LC_CAP = 5000          # guard: max link rows per label row (max on f1/trial/event: 3511)
CAPS = (1000, 500, 200, 100, 50, 20, 10, 5, 2, 1)
PE_MAX_NUM = 6         # H7: numeric parent columns per (child, parent) path
PE_MAX_CAT = 10        # H7: categorical parent columns per (child, parent) path
PE_LIST_POS = 3        # H7: list-valued parent columns: elements 0..PE_LIST_POS-1 (+ length)
PE_LEAK = os.environ.get("RELDFS_PLANT_PE", "") == "1"   # planted-leak switch for the leakcheck ONLY
FT_DAYS = (7, 30)       # H10 absolute windows (days)
GUARD_LOG: list = []   # plan-time estimates and build-time cost per path (for reports)


def _is_num(s: pd.Series) -> bool:
    return (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)) and not \
        pd.api.types.is_datetime64_any_dtype(s)


def _q(c: str) -> str:
    return '"' + c.replace('"', '""') + '"'


def make_plan(db, task) -> dict:
    ent = task.entity_table
    plan = {"entity": ent, "children": [], "timedelta_days": max(1, task.timedelta.days)}
    for name, tb in db.table_dict.items():
        fks = [c for c, p in tb.fkey_col_to_pkey_table.items() if p == ent]
        if not fks or name == ent:
            continue
        keys = {tb.pkey_col, *tb.fkey_col_to_pkey_table, tb.time_col}
        df = tb.df
        num, cat = [], {}
        for c in df.columns:
            if c in keys or c is None:
                continue
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                continue
            nun = s.nunique(dropna=True)
            if _is_num(s) and nun > 12:
                num.append(c)
            elif 1 < nun <= MAX_CAT_CARD:
                cat[c] = list(s.value_counts().index[:TOPK])
            elif _is_num(s) and nun > 1:
                num.append(c)
        for fk in fks:
            plan["children"].append(dict(table=name, fk=fk, time=tb.time_col, num=num, cat=cat))
    plan["track"] = _track_plan(db, task, plan)
    plan["gchild"] = _gchild_plan(db, task, plan)   # H9 (c8_gchild)
    plan["penr"] = _penr_plan(db, task, plan)       # H7 (c11_penr)
    plan["domain"] = _domain.detect(db, task)        # round 2: hand-written dom.* (per dataset)
    return plan


def _parent_frame(pt, need=None) -> pd.DataFrame:
    """H7: a static parent's columns, list-valued ones expanded (length, elements 0..k-1).

    need: the derived column names to keep (None = all)."""
    df = pt.df
    df = df.reset_index(drop=True)
    out = {pt.pkey_col: df[pt.pkey_col].astype("int64")}
    for c in df.columns:
        if c == pt.pkey_col:
            continue
        if need is not None and not any(n == c or n.startswith(c + "#") or n.startswith(c + "@") for n in need):
            continue
        s = df[c]
        if s.dtype == object:
            nn = s.dropna()
            if len(nn) and isinstance(nn.iloc[0], (list, tuple, np.ndarray)):
                out[c + "#len"] = s.map(lambda a: float(len(a)) if isinstance(a, (list, tuple, np.ndarray)) else np.nan).astype("float64")
                for i in range(PE_LIST_POS):
                    out[f"{c}@{i}"] = s.map(lambda a, i=i: str(a[i]) if isinstance(a, (list, tuple, np.ndarray)) and len(a) > i else None)
                continue
        out[c] = s.astype(object) if isinstance(s.dtype, pd.CategoricalDtype) else s
    return pd.DataFrame(out)


def _penr_plan(db, task, plan) -> list:
    ent = task.entity_table
    out = []
    for ch in plan["children"]:
        if not ch["time"]:
            continue
        tb = db.table_dict[ch["table"]]
        for pfk, ptab in tb.fkey_col_to_pkey_table.items():
            if pfk == ch["fk"] or ptab == ent:
                continue
            pt = db.table_dict[ptab]
            if pt.time_col is not None or pt.pkey_col is None:
                continue
            P = _parent_frame(pt)
            pk = pt.pkey_col
            w = tb.df[pfk].dropna().astype("int64").value_counts()
            W = pd.DataFrame({pk: w.index.to_numpy(), "__w": w.to_numpy()}).merge(P, on=pk, how="inner")
            pfks = set(pt.fkey_col_to_pkey_table)
            num, cat = [], {}
            for c in P.columns:
                if c == pk:
                    continue
                s = P[c]
                nun = s.nunique(dropna=True)
                if 1 < nun <= MAX_CAT_CARD:
                    if len(cat) >= PE_MAX_CAT:
                        continue
                    # near-bijection with a chosen column (a relabelling, e.g. product_type_no /
                    # product_type_name, 132 / 131 values): one determines the other and their
                    # cardinalities are within 10 %. A coarser level of a hierarchy is kept.
                    dup = False
                    for d in cat:
                        nd = P[d].nunique(dropna=True)
                        pair = len(P[[c, d]].dropna().drop_duplicates())
                        if pair == max(nun, nd) and min(nun, nd) >= 0.9 * max(nun, nd):
                            dup = True
                            break
                    if dup:
                        continue
                    vc = W.groupby(c, dropna=True)["__w"].sum()
                    vc = pd.DataFrame({"v": vc.index, "w": vc.to_numpy(), "k": vc.index.astype(str)})
                    vc = vc.sort_values(["w", "k"], ascending=[False, True], kind="mergesort")
                    cat[c] = list(vc["v"].iloc[:TOPK])
                elif c not in pfks and _is_num(s) and nun > 1 and len(num) < PE_MAX_NUM:
                    num.append(c)
            if num or cat:
                out.append(dict(table=ch["table"], fk=ch["fk"], time=ch["time"], pfk=pfk, ptab=ptab, pk=pk,
                                num=num, cat=cat))
    return out


def _penr(con, db, plan, td: int) -> tuple[list, list]:
    """H7 features: child rows (C.time < t) joined to their static parent's columns."""
    frames, counts = [], []
    for sp in plan.get("penr", []):
        pt = db.table_dict[sp["ptab"]]
        P = _parent_frame(pt, need=list(sp["num"]) + list(sp["cat"]))
        Q = pd.DataFrame({"pk": P[sp["pk"]].astype("int64").to_numpy()})
        for i, c in enumerate(sp["num"]):
            Q[f"x{i}"] = pd.to_numeric(P[c], errors="coerce").astype("float64").to_numpy()
        for j, (c, vals) in enumerate(sp["cat"].items()):
            code = {v: k for k, v in enumerate(vals)}
            Q[f"k{j}"] = P[c].astype(object).map(lambda v: -1 if pd.isna(v) else code.get(v, -1)).astype("int8").to_numpy()
        tb = db.table_dict[sp["table"]]
        con.register("pc", tb.df[[sp["fk"], sp["time"], sp["pfk"]]])
        con.register("pq", Q)
        T = f"pc.{_q(sp['time'])}"
        pid = f"pc.{_q(sp['pfk'])}"
        pre = f"pe.{sp['table']}.{sp['fk']}.{sp['pfk']}."
        sel = []
        for i, c in enumerate(sp["num"]):
            x = f"pq.x{i}"
            sel += [f"avg({x}) AS {_q(pre + c + '.mean')}", f"min({x}) AS {_q(pre + c + '.min')}",
                    f"max({x}) AS {_q(pre + c + '.max')}", f"stddev_samp({x}) AS {_q(pre + c + '.std')}",
                    f"avg({x}) FILTER (WHERE {T} >= l.__t - to_days({4 * td})) AS {_q(pre + c + '.mean_4td')}",
                    f"arg_max({x}, ({T}, {pid})) AS {_q(pre + c + '.last')}"]
        for j, (c, vals) in enumerate(sp["cat"].items()):
            for i in range(len(vals)):
                sel.append(f"count(*) FILTER (WHERE pq.k{j} = {i}) AS {_q(pre + c + f'.v{i}')}")
                sel.append(f"(count(*) FILTER (WHERE pq.k{j} = {i}))::DOUBLE / count(*) AS {_q(pre + c + f'.s{i}')}")
                counts.append(pre + c + f".v{i}")
        tf = "" if PE_LEAK else f" AND {T} < l.__t"   # PLANTED LEAK when RELDFS_PLANT_PE=1
        q = (f"SELECT l.__rid, {', '.join(sel)} FROM lab l JOIN pc ON pc.{_q(sp['fk'])} = l.__eid{tf}"
             f" JOIN pq ON pq.pk = {pid} GROUP BY l.__rid")
        frames.append(con.execute(q).df())
        con.unregister("pc")
        con.unregister("pq")
    return frames, counts



def _num_cols(tb) -> list:
    keys = {tb.pkey_col, *tb.fkey_col_to_pkey_table, tb.time_col}
    out = []
    for c in tb.df.columns:
        if c in keys or c is None:
            continue
        s = tb.df[c]
        if _is_num(s) and s.nunique(dropna=True) > 1:
            out.append(c)
    return out[:TRACK_MAX_NUM]


def _track_plan(db, task, plan) -> list:
    ent = task.entity_table
    links = {}
    for name, tb in db.table_dict.items():
        if name == ent:
            continue
        efks = [c for c, p in tb.fkey_col_to_pkey_table.items() if p == ent]
        if not efks:
            continue
        for pfk, ptab in tb.fkey_col_to_pkey_table.items():
            if ptab == ent or db.table_dict[ptab].time_col is not None:
                continue
            cand = (len(tb.df), name, efks[0], pfk, ptab)
            if ptab not in links or cand > links[ptab]:
                links[ptab] = cand
    # children of E with a time column and numeric columns (path B targets)
    gb = []
    for ch in plan["children"]:
        tb = db.table_dict[ch["table"]]
        num = _num_cols(tb)
        if ch["time"] and num:
            gb.append(dict(table=ch["table"], fk=ch["fk"], time=ch["time"], num=num))
    out = []
    for ptab, (_, ctab, efk, pfk, _) in sorted(links.items()):
        ct = db.table_dict[ctab]
        last = False
        if ct.time_col is not None:   # do an entity's C rows carry different times?
            d = ct.df[[efk, ct.time_col]].dropna()
            d = d[d[efk].isin(d[efk].drop_duplicates().head(2000))]
            last = bool((d.groupby(efk)[ct.time_col].nunique() > 1).mean() > 0.1)
        ga = []
        for gname, gt in db.table_dict.items():
            if gt.time_col is None:
                continue
            for gfk, gp in gt.fkey_col_to_pkey_table.items():
                if gp == ptab:
                    ga.append(dict(table=gname, fk=gfk, time=gt.time_col, num=_num_cols(gt)))
        out.append(dict(ptab=ptab, link=ctab, efk=efk, pfk=pfk, ctime=ct.time_col, last=last,
                        A=[dict(x) for x in ga], B=[dict(x) for x in gb]))
    _guard_plan(db, out, plan["timedelta_days"])
    return out


def _ceil_day(x: str) -> str:
    return f"(CASE WHEN {x} = date_trunc('day', {x}) THEN {x} ELSE date_trunc('day', {x}) + INTERVAL 1 DAY END)"


def _width(g) -> int:
    return 8 * (3 + 2 * len(g["num"]))


def _guard_plan(db, out, td: int) -> None:
    """Scale-up guard: size estimates on the full db -> per-path caps stored in the plan."""
    import time as _time

    con = duckdb.connect()
    tmp = DATA_DIR / "ddbtmp"
    con.execute(f"SET threads=2; SET memory_limit='3GB'; SET preserve_insertion_order=false;"
                f" SET temp_directory='{tmp}'; SET max_temp_directory_size='500MB'")
    for tr in out:
        t0 = _time.time()
        C = db.table_dict[tr["link"]]
        ccols = [tr["efk"], tr["pfk"]] + ([tr["ctime"]] if tr["ctime"] else [])
        con.register("tc", C.df[ccols])
        ef, pf = _q(tr["efk"]), _q(tr["pfk"])
        ct = _q(tr["ctime"]) if tr["ctime"] else "NULL::TIMESTAMP"
        nn = f"{ef} IS NOT NULL AND {pf} IS NOT NULL"
        g = dict(lc_cap=None, p_cap=None, edge_cap=None)
        mx = con.execute(f"SELECT max(n) FROM (SELECT count(*) n FROM tc WHERE {nn} GROUP BY {ef})").fetchone()[0] or 0
        if mx > LC_CAP:
            g["lc_cap"] = LC_CAP
        qw = sum(8 * (2 + 2 * len(x["num"])) for x in tr["A"] + tr["B"])
        dp = con.execute(f"SELECT count(DISTINCT {pf}) d FROM tc WHERE {nn} GROUP BY {ef}").df()["d"].to_numpy()
        if len(dp) and dp.mean() * qw > GUARD_Q_BYTES:
            g["p_cap"] = next((k for k in CAPS if np.minimum(dp, k).mean() * qw <= GUARD_Q_BYTES), 1)
        log = dict(path=tr["ptab"], link=tr["link"], max_links=int(mx), mean_parents=float(dp.mean()) if len(dp) else 0.0,
                   q_bytes_per_row=float(dp.mean() * qw) if len(dp) else 0.0, **g, specs=[])
        for spec in tr["A"]:
            G = db.table_dict[spec["table"]]
            con.register("tg", G.df[[spec["fk"], spec["time"]]])
            fk, gt = _q(spec["fk"]), _q(spec["time"])
            w = f"{fk} IS NOT NULL AND {gt} IS NOT NULL"
            rows = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT {fk}, {gt} FROM tg WHERE {w})").fetchone()[0]
            spec["bucket"], spec["drop"] = False, False
            rows2 = rows
            if rows * _width(spec) > GUARD_BYTES:
                spec["bucket"] = True
                rows2 = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT {fk}, {_ceil_day(gt)} FROM tg WHERE {w})").fetchone()[0]
                if rows2 * _width(spec) > GUARD_BYTES:   # shed numeric columns before dropping the path
                    k = int((GUARD_BYTES / (8 * rows2) - 3) // 2)
                    if k < 0:
                        spec["drop"] = True
                    else:
                        spec["num"] = spec["num"][:k]
            log["specs"].append(dict(kind="A", table=spec["table"], rows=int(rows), rows_after=int(rows2),
                                     bucket=spec["bucket"], drop=spec["drop"], k=len(spec["num"])))
            con.unregister("tg")
        if tr["B"]:
            con.execute(f"CREATE OR REPLACE TEMP TABLE gcp AS SELECT {pf} AS p, {ef} AS e2, min({ct}) AS ct FROM tc"
                        f" WHERE {nn} GROUP BY 1, 2")
            bk = f"floor(epoch(ct) / {td * 86400})"
            con.execute(f"CREATE OR REPLACE TEMP TABLE gcpr AS SELECT *, row_number() OVER (PARTITION BY p, {bk}"
                        f" ORDER BY ct, e2) AS rp, row_number() OVER (PARTITION BY e2, {bk} ORDER BY ct, p) AS re FROM gcp")

            def est(spec, bucket, k):
                tt = _ceil_day(_q(spec["time"])) if bucket else _q(spec["time"])
                capw = f" WHERE rp <= {k} AND re <= {k}" if k else ""
                return con.execute(f"WITH T AS (SELECT {_q(spec['fk'])} AS e2, count(DISTINCT {tt}) AS T FROM tg"
                                   f" WHERE {_q(spec['fk'])} IS NOT NULL AND {_q(spec['time'])} IS NOT NULL GROUP BY 1)"
                                   f" SELECT coalesce(sum(T), 0) FROM (SELECT e2 FROM gcpr{capw}) JOIN T USING (e2)").fetchone()[0]

            need = []
            for spec in tr["B"]:
                G = db.table_dict[spec["table"]]
                con.register("tg", G.df[[spec["fk"], spec["time"]]])
                rows = est(spec, False, None)
                spec["bucket"], spec["drop"] = False, False
                rows2 = rows
                if rows * _width(spec) > GUARD_BYTES:
                    spec["bucket"] = True
                    rows2 = est(spec, True, None)
                    if rows2 * _width(spec) > GUARD_BYTES:
                        need.append(spec)
                spec["_est"] = (int(rows), int(rows2))
                con.unregister("tg")
            if need:   # one edge cap per path: the largest K that brings every such spec under budget
                ok = {}
                for spec in need:
                    G = db.table_dict[spec["table"]]
                    con.register("tg", G.df[[spec["fk"], spec["time"]]])
                    ok[id(spec)] = {k: est(spec, True, k) for k in CAPS}
                    con.unregister("tg")
                keep = [x for x in need if ok[id(x)][1] * _width(x) <= GUARD_BYTES]
                for x in need:
                    if x not in keep:
                        x["drop"] = True
                if keep:
                    g["edge_cap"] = next(k for k in CAPS if all(ok[id(x)][k] * _width(x) <= GUARD_BYTES for x in keep))
                for x in need:
                    x["_est"] = x["_est"] + (int(ok[id(x)][g["edge_cap"] or 1]),)
            log["edge_cap"] = g["edge_cap"]
            for spec in tr["B"]:
                log["specs"].append(dict(kind="B", table=spec["table"], rows=spec["_est"][0], rows_bucket=spec["_est"][1],
                                         rows_capped=spec["_est"][2] if len(spec["_est"]) > 2 else None,
                                         bucket=spec["bucket"], drop=spec["drop"]))
                del spec["_est"]
        con.unregister("tc")
        tr["guard"] = g
        log["plan_secs"] = round(_time.time() - t0, 1)
        GUARD_LOG.append(log)
    con.close()


def _track(con, db, plan, td: int) -> tuple[list, list]:
    """H1 features; returns (frames aligned to lab.__rid, count column names)."""
    W = TRACK_WIN * td
    frames, counts = [], []
    for tr in plan.get("track", []):
        C = db.table_dict[tr["link"]]
        ccols = [tr["efk"], tr["pfk"]] + ([tr["ctime"]] if tr["ctime"] else [])
        con.register("tc", C.df[ccols])
        ef, pf = f"tc.{_q(tr['efk'])}", f"tc.{_q(tr['pfk'])}"
        ctf = f"tc.{_q(tr['ctime'])}" if tr["ctime"] else None
        vis = f" AND {ctf} < l.__t" if ctf else ""
        # label rows x their own C rows (entity-keyed), and the (P, t) pairs to evaluate
        import time as _time
        import resource as _res
        t0 = _time.time()
        gd = tr.get("guard", {})
        con.execute(f"CREATE OR REPLACE TEMP TABLE lc AS SELECT l.__rid, l.__t, {pf} AS p, "
                    f"{ctf if ctf else 'NULL::TIMESTAMP'} AS ct FROM lab l JOIN tc ON {ef} = l.__eid{vis}"
                    f" WHERE {pf} IS NOT NULL")
        if gd.get("lc_cap"):   # guard: the most recent LC_CAP link rows per label row (all < t)
            con.execute(f"CREATE OR REPLACE TEMP TABLE lc AS SELECT * FROM lc QUALIFY row_number() OVER"
                        f" (PARTITION BY __rid ORDER BY ct DESC NULLS LAST, p) <= {gd['lc_cap']}")
        if gd.get("p_cap"):    # guard: the K most recently linked distinct parents per label row
            con.execute(f"CREATE OR REPLACE TEMP TABLE lc AS SELECT lc.* FROM lc JOIN (SELECT __rid, p FROM"
                        f" (SELECT __rid, p, max(ct) AS m FROM lc GROUP BY 1, 2) QUALIFY row_number() OVER"
                        f" (PARTITION BY __rid ORDER BY m DESC NULLS LAST, p) <= {gd['p_cap']}) k USING (__rid, p)")
        n_lc = con.execute("SELECT count(*) FROM lc").fetchone()[0]
        con.execute("CREATE OR REPLACE TEMP TABLE q AS SELECT DISTINCT p, __t AS t FROM lc")
        con.execute("CREATE OR REPLACE TEMP TABLE rel AS SELECT DISTINCT p FROM q")
        if tr["B"]:
            con.execute(f"CREATE OR REPLACE TEMP TABLE cp AS SELECT {pf} AS p, {ef} AS e2, "
                        f"{'min(' + ctf + ')' if ctf else 'NULL::TIMESTAMP'} AS ct FROM tc "
                        f"WHERE {pf} IN (SELECT p FROM rel) AND {ef} IS NOT NULL GROUP BY 1, 2")
            if gd.get("edge_cap"):   # guard: earliest-K links per (parent, bucket) and per (e', bucket)
                bk = f"floor(epoch(ct) / {td * 86400})"
                k = gd["edge_cap"]
                # ranks over ALL links of the parent / of e' (not only relevant parents), as planned
                con.execute(f"CREATE OR REPLACE TEMP TABLE cpa AS SELECT {pf} AS p, {ef} AS e2, "
                            f"{'min(' + ctf + ')' if ctf else 'NULL::TIMESTAMP'} AS ct FROM tc "
                            f"WHERE {pf} IS NOT NULL AND {ef} IS NOT NULL GROUP BY 1, 2")
                con.execute(f"CREATE OR REPLACE TEMP TABLE cp AS SELECT p, e2, ct FROM (SELECT *, row_number() OVER"
                            f" (PARTITION BY p, {bk} ORDER BY ct, e2) AS rp, row_number() OVER (PARTITION BY e2, {bk}"
                            f" ORDER BY ct, p) AS re FROM cpa) WHERE rp <= {k} AND re <= {k} AND p IN (SELECT p FROM rel)")
                con.execute("DROP TABLE cpa")
        pieces = []
        for kind, specs in (("A", tr["A"]), ("B", tr["B"])):
            for g in specs:
                if g.get("drop"):
                    continue
                G = db.table_dict[g["table"]]
                con.register("tg", G.df[[g["fk"], g["time"]] + g["num"]])
                gt = _ceil_day(f"tg.{_q(g['time'])}") if g.get("bucket") else f"tg.{_q(g['time'])}"
                sums = "".join(f", sum(tg.{_q(x)}::DOUBLE) AS s{i}, count(tg.{_q(x)}) AS c{i}"
                               for i, x in enumerate(g["num"]))
                if kind == "A":
                    con.execute(f"CREATE OR REPLACE TEMP TABLE ev AS SELECT tg.{_q(g['fk'])} AS p, {gt} AS e,"
                                f" count(*) AS n{sums} FROM tg WHERE tg.{_q(g['time'])} IS NOT NULL AND"
                                f" tg.{_q(g['fk'])} IN (SELECT p FROM rel) GROUP BY 1, 2")
                else:
                    ssum = "".join(f", sum(s{i}) AS s{i}, sum(c{i}) AS c{i}" for i in range(len(g["num"])))
                    con.execute(f"CREATE OR REPLACE TEMP TABLE gg AS SELECT tg.{_q(g['fk'])} AS e2, {gt} AS gt,"
                                f" count(*) AS n{sums} FROM tg WHERE tg.{_q(g['time'])} IS NOT NULL AND tg.{_q(g['fk'])}"
                                f" IN (SELECT e2 FROM cp) GROUP BY 1, 2")
                    con.execute("CREATE OR REPLACE TEMP TABLE ev AS SELECT cp.p, CASE WHEN cp.ct IS NULL OR "
                                "gg.gt > cp.ct THEN gg.gt ELSE cp.ct END AS e, sum(n) AS n" + ssum +
                                " FROM cp JOIN gg USING (e2) GROUP BY 1, 2")
                k = len(g["num"])
                win = "OVER (PARTITION BY p ORDER BY e ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
                cs = ", ".join([f"sum(n) {win} AS n"] + [f"sum(s{i}) {win} AS s{i}, sum(c{i}) {win} AS c{i}"
                                                         for i in range(k)])
                con.execute(f"CREATE OR REPLACE TEMP TABLE cum AS SELECT p, e, {cs} FROM ev")
                cols = ["n"] + [f"s{i}" for i in range(k)] + [f"c{i}" for i in range(k)]
                a = ", ".join(f"a.{c} AS a_{c}" for c in cols)
                b = ", ".join(f"b.{c} AS b_{c}" for c in cols)
                st = con.execute(
                    f"WITH qa AS (SELECT q.p, q.t, {a} FROM q ASOF LEFT JOIN cum a ON q.p = a.p AND q.t > a.e),"
                    f" qb AS (SELECT q.p, q.t, {b} FROM q ASOF LEFT JOIN cum b ON q.p = b.p AND"
                    f" q.t - to_days({W}) > b.e) SELECT * FROM qa JOIN qb USING (p, t)").df()
                pre = f"trk.{tr['ptab']}.{kind}.{g['table']}."
                o = pd.DataFrame({"p": st["p"], "t": st["t"]})
                na, nb = st["a_n"].fillna(0), st["b_n"].fillna(0)
                o[pre + "n"] = na
                o[pre + f"n_{TRACK_WIN}td"] = na - nb
                for i, x in enumerate(g["num"]):
                    sa, ca = st[f"a_s{i}"].fillna(0), st[f"a_c{i}"].fillna(0)
                    sb, cb = st[f"b_s{i}"].fillna(0), st[f"b_c{i}"].fillna(0)
                    o[pre + f"{x}.mean"] = (sa / ca.where(ca > 0)).astype("float64")
                    o[pre + f"{x}.mean_{TRACK_WIN}td"] = ((sa - sb) / (ca - cb).where(ca - cb > 0)).astype("float64")
                pieces.append(o.set_index(["p", "t"]))
                con.unregister("tg")
        con.unregister("tc")
        if not pieces:
            continue
        P = pd.concat(pieces, axis=1).reset_index()
        fcols = [c for c in P.columns if c not in ("p", "t")]
        con.register("pst", P)
        red = []
        for c in fcols:
            qc = _q(c)
            red += [f"avg(s.{qc}) AS {_q(c + '.pmean')}", f"max(s.{qc}) AS {_q(c + '.pmax')}"]
            if tr["last"]:
                red.append(f"arg_max(s.{qc}, (lc.ct, lc.p)) AS {_q(c + '.plast')}")  # ties in ct broken by p (EXPERIMENTS.md 3b)
        agg = con.execute(f"SELECT lc.__rid, count(DISTINCT lc.p) AS {_q('trk.' + tr['ptab'] + '.nparents')}, "
                          f"{', '.join(red)} FROM lc JOIN pst s ON s.p = lc.p AND s.t = lc.__t GROUP BY lc.__rid").df()
        con.unregister("pst")
        frames.append(agg)
        counts.append("trk." + tr["ptab"] + ".nparents")
        GUARD_LOG.append(dict(path=tr["ptab"], build_secs=round(_time.time() - t0, 1), lc_rows=int(n_lc),
                              q_rows=int(len(P)), q_cols=len(fcols), label_rows=int(len(agg)),
                              rss_gb=round(_res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1e6, 2)))
        print(f"[guard] {GUARD_LOG[-1]}", flush=True)
    return frames, counts


def _own_frame(db, task) -> tuple[pd.DataFrame, list]:
    """Entity row + its direct parents' rows, flattened. Dates kept raw (made relative later)."""
    et = db.table_dict[task.entity_table]
    df = et.df.copy()
    out = pd.DataFrame({"__eid": df[et.pkey_col].astype("int64").values})
    dates = []

    def add(prefix, frame, skip):
        for c in frame.columns:
            if c in skip:
                continue
            s = frame[c]
            col = f"{prefix}{c}"
            if pd.api.types.is_datetime64_any_dtype(s):
                out[col] = s.values
                dates.append(col)
            elif _is_num(s):
                out[col] = pd.to_numeric(s, errors="coerce").astype("float64").values
            else:
                s = s.astype("string")
                if s.nunique() <= OWN_CAT_CARD:
                    out[col] = s.astype("category").cat.codes.astype("float64").replace(-1, np.nan).values

    add("own.", df, {et.pkey_col, *et.fkey_col_to_pkey_table})
    for fk, ptab in et.fkey_col_to_pkey_table.items():
        pt = db.table_dict[ptab]
        if pt.time_col is not None:   # parent rows with a time are not static: skip in c0
            continue
        pdf = pt.df
        m = df[[fk]].merge(pdf, how="left", left_on=fk, right_on=pt.pkey_col)
        add(f"par.{fk}.", m, {fk, pt.pkey_col, *pt.fkey_col_to_pkey_table})
    return out, dates


GC_BYTES = 1.0e9      # H9 guard: max bytes of one path's (entity, vt) table; a day bucket drops the last
                      # partial day (ceil(vt) = t), so it is kept for tables that really need it
GC_MAX_NUM = 12       # H9: numeric G columns per path
GC_MAX_IND = 4        # H9: low-cardinality G columns per path (top-8 indicators each)
GC_LEAK = os.environ.get("RELDFS_PLANT_GC", "")   # planted-leak switch for the leakcheck ONLY


def _gchild_plan(db, task, plan) -> list:
    """H9 paths E <- C <- G (C has a pkey, G a fkey to C); decisions from full-db stats."""
    ent = task.entity_table
    out = []
    con = None
    for ch in plan["children"]:
        C = db.table_dict[ch["table"]]
        if C.pkey_col is None or ch["table"] == ent:
            continue
        for gname, G in db.table_dict.items():
            if gname == ent:
                continue
            for gfk, gp in G.fkey_col_to_pkey_table.items():
                if gp != ch["table"] or (C.time_col is None and G.time_col is None):
                    continue
                keys = {G.pkey_col, *G.fkey_col_to_pkey_table, G.time_col}
                num, ind = [], {}
                for c in G.df.columns:
                    if c in keys or c is None:
                        continue
                    s = G.df[c]
                    if pd.api.types.is_datetime64_any_dtype(s):
                        continue
                    nun = s.nunique(dropna=True)
                    if _is_num(s) and (nun == 2 or nun > 20):
                        if len(num) < GC_MAX_NUM:
                            num.append(c)
                    elif 2 < nun <= (20 if _is_num(s) else MAX_CAT_CARD) and len(ind) < GC_MAX_IND:
                        ind[c] = list(s.value_counts().index[:TOPK])
                spec = dict(ctab=ch["table"], efk=ch["fk"], cpk=C.pkey_col, ctime=C.time_col,
                            gtab=gname, gfk=gfk, gtime=G.time_col, num=num, ind=ind, bucket=False, drop=False)
                # guard: size of the (entity, vt) table
                if con is None:
                    con = duckdb.connect()
                    tmp = DATA_DIR / "ddbtmp"
                    con.execute(f"SET threads=2; SET memory_limit='3GB'; SET preserve_insertion_order=false;"
                                f" SET temp_directory='{tmp}'; SET max_temp_directory_size='500MB'")
                con.register("tc", C.df[[C.pkey_col, ch["fk"]] + ([C.time_col] if C.time_col else [])])
                con.register("tg", G.df[[gfk] + ([G.time_col] if G.time_col else [])])
                vt = _gc_vt(spec)
                frm = (f"FROM tg JOIN tc ON tg.{_q(gfk)} = tc.{_q(C.pkey_col)} WHERE tc.{_q(ch['fk'])} IS NOT NULL"
                       f" AND {vt} IS NOT NULL")
                width = 8 * (2 + 5 * len(num) + sum(len(v) for v in ind.values()))
                rows = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT tc.{_q(ch['fk'])}, {vt} {frm})").fetchone()[0]
                rows2 = rows
                if rows * width > GC_BYTES:
                    spec["bucket"] = True
                    rows2 = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT tc.{_q(ch['fk'])}, {_ceil_day(vt)}"
                                        f" {frm})").fetchone()[0]
                    while rows2 * width > GC_BYTES and (spec["ind"] or spec["num"]):
                        if spec["ind"]:
                            spec["ind"].pop(list(spec["ind"])[-1])
                        else:
                            spec["num"].pop()
                        width = 8 * (2 + 5 * len(spec["num"]) + sum(len(v) for v in spec["ind"].values()))
                    spec["drop"] = rows2 * width > GC_BYTES
                con.unregister("tc")
                con.unregister("tg")
                GUARD_LOG.append(dict(gchild=f"{ch['table']}<-{gname}.{gfk}", rows=int(rows), rows_after=int(rows2),
                                      bucket=spec["bucket"], drop=spec["drop"], k=len(spec["num"]), ind=len(spec["ind"])))
                out.append(spec)
    if con is not None:
        con.close()
    return out


def _gc_vt(g) -> str:
    """Virtual time of a V row: the later of the two hops (NULL when a timed hop has no time)."""
    ct = f"tc.{_q(g['ctime'])}" if g["ctime"] else None
    gt = f"tg.{_q(g['gtime'])}" if g["gtime"] else None
    if GC_LEAK == "drop_g" and ct:      # PLANTED LEAK: ignore the G hop's time
        gt = None
    if ct and gt:
        return f"(CASE WHEN {ct} IS NULL OR {gt} IS NULL THEN NULL WHEN {gt} > {ct} THEN {gt} ELSE {ct} END)"
    return ct or gt


def _gchild(con, db, plan, td: int) -> tuple[list, list]:
    """H9 features; returns (frames with __rid, count column names)."""
    import time as _time
    import resource as _res

    frames, counts = [], []
    con.execute("CREATE OR REPLACE TEMP TABLE q9 AS SELECT DISTINCT __eid AS e, __t AS t FROM lab")
    shift = f" + to_days({td})" if GC_LEAK == "lookahead" else ""   # PLANTED LEAK: one window ahead
    for g in plan.get("gchild", []):
        if g["drop"]:
            continue
        t0 = _time.time()
        C, G = db.table_dict[g["ctab"]], db.table_dict[g["gtab"]]
        con.register("tc", C.df[[g["cpk"], g["efk"]] + ([g["ctime"]] if g["ctime"] else [])])
        con.register("tg", G.df[[g["gfk"]] + ([g["gtime"]] if g["gtime"] else []) + g["num"] + list(g["ind"])])
        vt = _gc_vt(g)
        if g["bucket"]:
            vt = _ceil_day(vt)
        cols, sel = [], ["count(*) AS n"]
        for i, x in enumerate(g["num"]):
            v = f"tg.{_q(x)}::DOUBLE"
            sel += [f"sum({v}) AS s{i}", f"count({v}) AS c{i}", f"sum({v} * {v}) AS q{i}", f"max({v}) AS m{i}"]
            cols += [f"s{i}", f"c{i}", f"q{i}"]
        j = 0
        for x, vals in g["ind"].items():
            for v in vals:
                lit = f"'{str(v).replace(chr(39), chr(39) * 2)}'"
                sel.append(f"count(*) FILTER (WHERE tg.{_q(x)}::VARCHAR = {lit}) AS i{j}")
                cols.append(f"i{j}")
                j += 1
        con.execute(f"CREATE OR REPLACE TEMP TABLE v9 AS SELECT tc.{_q(g['efk'])} AS e, {vt} AS vt, {', '.join(sel)}"
                    f" FROM tg JOIN tc ON tg.{_q(g['gfk'])} = tc.{_q(g['cpk'])} WHERE tc.{_q(g['efk'])} IN"
                    f" (SELECT e FROM q9) AND {vt} IS NOT NULL GROUP BY 1, 2")
        con.unregister("tc")
        con.unregister("tg")
        win = "OVER (PARTITION BY e ORDER BY vt ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
        cum = ", ".join([f"sum(n) {win} AS n", f"min(vt) OVER (PARTITION BY e) AS vfirst"] +
                        [f"sum({c}) {win} AS {c}" for c in cols] +
                        [f"max(m{i}) {win} AS m{i}" for i in range(len(g["num"]))])
        con.execute(f"CREATE OR REPLACE TEMP TABLE cum9 AS SELECT e, vt, {cum} FROM v9")
        con.execute("DROP TABLE v9")
        allc = ["n"] + cols
        parts = []
        for tag, off in (("a", 0), ("b1", 1), ("b4", 4), ("b16", 16)):
            keep = allc + (["vt", "vfirst"] + [f"m{i}" for i in range(len(g["num"]))] if tag == "a" else [])
            if tag in ("b1", "b16"):
                keep = ["n"] + [c for c in cols if c.startswith("s") or c.startswith("i")]
            sel = ", ".join(f"x.{c} AS {tag}_{c}" for c in keep)
            cond = f"q9.t{shift} > x.vt" if off == 0 else f"q9.t - to_days({off * td}){shift} > x.vt"
            parts.append(f"(SELECT q9.e, q9.t, {sel} FROM q9 ASOF LEFT JOIN cum9 x ON q9.e = x.e AND {cond})")
        sql = parts[0] + " a " + " ".join(f"JOIN {p} {nm} USING (e, t)" for p, nm in zip(parts[1:], ("b1", "b4", "b16")))
        st = con.execute(f"SELECT * FROM {sql}").df()
        con.execute("DROP TABLE cum9")
        pre = f"gc.{g['ctab']}.{g['gtab']}.{g['gfk']}."
        o = pd.DataFrame({"e": st["e"], "t": st["t"]})
        na = st["a_n"].fillna(0)
        o[pre + "n"] = na
        for w, tag in ((1, "b1"), (4, "b4"), (16, "b16")):
            o[pre + f"n_{w}td"] = na - st[f"{tag}_n"].fillna(0)
        tt = pd.to_datetime(st["t"])
        o[pre + "days_since_last"] = (tt - pd.to_datetime(st["a_vt"])).dt.days.astype("float64")
        o[pre + "days_since_first"] = (tt - pd.to_datetime(st["a_vfirst"])).dt.days.astype("float64").where(na > 0)
        f = lambda c: st[c].fillna(0)  # noqa: E731
        for i, x in enumerate(g["num"]):
            p = pre + x
            sa, ca, qa = f(f"a_s{i}"), f(f"a_c{i}"), f(f"a_q{i}")
            s4, c4 = sa - f(f"b4_s{i}"), ca - f(f"b4_c{i}")
            mean = sa / ca.where(ca > 0)
            o[p + ".mean"] = mean
            o[p + ".std"] = np.sqrt(((qa - ca * mean * mean) / (ca - 1).where(ca > 1)).clip(lower=0))
            o[p + ".max"] = st[f"a_m{i}"]
            o[p + ".sum"] = sa
            o[p + ".mean_4td"] = s4 / c4.where(c4 > 0)
            o[p + ".sum_1td"] = sa - f(f"b1_s{i}")
            o[p + ".sum_4td"] = s4
        j = 0
        for x, vals in g["ind"].items():
            for k in range(len(vals)):
                p = pre + f"{x}.v{k}"
                ia = f(f"a_i{j}")
                o[p + ".sum"] = ia
                o[p + ".share"] = ia / na.where(na > 0)
                o[p + ".sum_4td"] = ia - f(f"b4_i{j}")
                counts += [p + ".sum", p + ".sum_4td"]
                j += 1
        counts += [pre + "n"] + [pre + f"n_{w}td" for w in WINDOWS] + [pre + x + s for x in g["num"]
                                                                        for s in (".sum", ".sum_1td", ".sum_4td")]
        con.register("o9", o)
        agg = con.execute("SELECT l.__rid, o9.* EXCLUDE (e, t) FROM lab l JOIN o9 ON o9.e = l.__eid AND o9.t = l.__t").df()
        con.unregister("o9")
        frames.append(agg)
        GUARD_LOG.append(dict(gchild=pre, build_secs=round(_time.time() - t0, 1), q_rows=int(len(o)),
                              cols=int(o.shape[1] - 2), rss_gb=round(_res.getrusage(_res.RUSAGE_SELF).ru_maxrss / 1e6, 2)))
        print(f"[gchild] {GUARD_LOG[-1]}", flush=True)
    con.execute("DROP TABLE q9")
    return frames, counts



def _fine(con, db, plan, td: int) -> tuple[list, list]:
    """H10 ft.* features; returns (frames with __rid, count column names)."""
    import os as _os

    plant = _os.environ.get("RELDFS_PLANT_FT") == "1"
    frames, counts = [], []
    old = {w * td for w in WINDOWS}
    nwin = [k for k in FT_DAYS if k not in old]
    adwin = sorted({*FT_DAYS, td, 4 * td})
    for ch in plan["children"]:
        if not ch["time"]:
            continue
        tb = db.table_dict[ch["table"]]
        con.register("c", tb.df[[ch["fk"], ch["time"]]])
        fk, T = f"c.{_q(ch['fk'])}", f"c.{_q(ch['time'])}"
        con.execute(f"CREATE OR REPLACE TEMP TABLE dd AS SELECT {fk} AS e, CAST(date_trunc('day', {T}::TIMESTAMP)"
                    f" AS TIMESTAMP) AS d, count(*) AS n FROM c WHERE {fk} IS NOT NULL AND {T} IS NOT NULL"
                    f" AND {fk} IN (SELECT DISTINCT __eid FROM lab) GROUP BY 1, 2")
        con.unregister("c")
        p = f"ft.{ch['table']}.{ch['fk']}."
        sel = [f"sum(dd.n) FILTER (WHERE dd.d >= l.__t - to_days({k})) AS {_q(p + f'n_{k}d')}" for k in nwin]
        sel += [f"count(*) FILTER (WHERE dd.d >= l.__t - to_days({k})) AS {_q(p + f'ad_{k}d')}" for k in adwin]
        sel += [f"count(*) AS {_q(p + 'ad_all')}",
                f"date_diff('day', min(dd.d), max(dd.d)) / nullif(count(*) - 1, 0) AS {_q(p + 'gap_mean')}",
                f"date_diff('day', list_extract(max(dd.d, 2), 2), max(dd.d)) AS {_q(p + 'gap_last')}"]
        cond = f"dd.d < l.__t + to_days({td})" if plant else "dd.d + INTERVAL 1 DAY <= l.__t"
        agg = con.execute(f"SELECT l.__rid, {', '.join(sel)} FROM lab l JOIN dd ON dd.e = l.__eid AND {cond}"
                          f" GROUP BY l.__rid").df()
        frames.append(agg)
        counts += [p + f"n_{k}d" for k in nwin] + [p + f"ad_{k}d" for k in adwin] + [p + "ad_all"]
    con.execute("DROP TABLE IF EXISTS dd")
    return frames, counts


def build(db, task, plan, lab: pd.DataFrame, threads: int = 2) -> pd.DataFrame:
    """lab: DataFrame with columns [entity_col, time_col]; returns features row-aligned."""
    L = pd.DataFrame({
        "__rid": np.arange(len(lab), dtype=np.int64),
        "__eid": lab[task.entity_col].astype("int64").values,
        "__t": pd.to_datetime(lab[task.time_col]).values,
    })
    con = duckdb.connect()
    tmp = DATA_DIR / "ddbtmp"  # bounded spill (disk is shared)
    con.execute(f"SET threads={threads}; SET memory_limit='4GB'; SET preserve_insertion_order=false;"
                f" SET temp_directory='{tmp}'; SET max_temp_directory_size='500MB'")
    con.register("lab", L)
    feats = [L[["__rid"]]]
    td = plan["timedelta_days"]
    count_cols = []
    for ch in plan["children"]:
        tb = db.table_dict[ch["table"]]
        cols = [ch["fk"]] + ([ch["time"]] if ch["time"] else []) + ch["num"] + list(ch["cat"])
        con.register("c", tb.df[cols])
        p = f"{ch['table']}.{ch['fk']}."
        sel = [f"count(*) AS {_q(p + 'n')}"]
        if ch["time"]:
            T = f"c.{_q(ch['time'])}"
            for w in WINDOWS:
                sel.append(f"count(*) FILTER (WHERE {T} >= l.__t - to_days({w * td})) AS {_q(p + f'n_{w}td')}")
            sel.append(f"date_diff('day', max({T}), l.__t) AS {_q(p + 'days_since_last')}")
            sel.append(f"date_diff('day', min({T}), l.__t) AS {_q(p + 'days_since_first')}")
        for c in ch["num"]:
            x = f"c.{_q(c)}::DOUBLE"
            sel += [f"avg({x}) AS {_q(p + c + '.mean')}", f"min({x}) AS {_q(p + c + '.min')}",
                    f"max({x}) AS {_q(p + c + '.max')}", f"stddev_samp({x}) AS {_q(p + c + '.std')}"]
            if ch["time"]:
                sel.append(f"avg({x}) FILTER (WHERE {T} >= l.__t - to_days({4 * td})) AS {_q(p + c + '.mean_4td')}")
                sel.append(f"arg_max({x}, {T}) AS {_q(p + c + '.last')}")
        for c, vals in ch["cat"].items():
            for i, v in enumerate(vals):
                lit = f"'{str(v).replace(chr(39), chr(39) * 2)}'"
                sel.append(f"count(*) FILTER (WHERE c.{_q(c)}::VARCHAR = {lit}) AS {_q(p + c + f'.v{i}')}")
        cond = f"c.{_q(ch['fk'])} = l.__eid" + (f" AND c.{_q(ch['time'])} < l.__t" if ch["time"] else "")
        q = f"SELECT l.__rid, {', '.join(sel)} FROM lab l JOIN c ON {cond} GROUP BY l.__rid, l.__t"
        count_cols += [p + "n"] + [p + f"n_{w}td" for w in WINDOWS if ch["time"]] + \
            [p + c + f".v{i}" for c, vals in ch["cat"].items() for i in range(len(vals))]
        agg = con.execute(q).df()
        feats.append(L[["__rid"]].merge(agg, on="__rid", how="left").drop(columns="__rid"))
        con.unregister("c")
    tf, tcount = _track(con, db, plan, td)
    for agg in tf:
        feats.append(L[["__rid"]].merge(agg, on="__rid", how="left").drop(columns="__rid"))
    count_cols += tcount
    gf, gcount = _gchild(con, db, plan, td)   # H9 (c8_gchild)
    for agg in gf:
        feats.append(L[["__rid"]].merge(agg, on="__rid", how="left").drop(columns="__rid"))
    count_cols += gcount
    pf, pcount = _penr(con, db, plan, td)     # H7 (c11_penr)
    for agg in pf:
        feats.append(L[["__rid"]].merge(agg, on="__rid", how="left").drop(columns="__rid"))
    count_cols += pcount
    ff, fcount = _fine(con, db, plan, td)   # H10 (EXPERIMENTS.md 6a)
    fine = [L[["__rid"]].merge(agg, on="__rid", how="left").drop(columns="__rid") for agg in ff]
    con.close()
    X = pd.concat(feats, axis=1).drop(columns="__rid")
    # counts are 0, not NaN, when an entity has no earlier child rows
    X[count_cols] = X[count_cols].fillna(0)

    own, dates = _own_frame(db, task)
    O = L[["__eid", "__t"]].merge(own, on="__eid", how="left")
    for c in dates:
        O[c] = (O["__t"] - pd.to_datetime(O[c])).dt.days.astype("float64")
    O = O.drop(columns=["__eid", "__t"])
    X = pd.concat([X.reset_index(drop=True), O.reset_index(drop=True)]
                  + [f.reset_index(drop=True) for f in fine], axis=1)
    if fcount:   # ft counts are 0, not NaN, when an entity has no earlier child rows
        X[fcount] = X[fcount].fillna(0)
    if plan.get("domain"):   # round 2: dom.* appended after every c14 column
        D = _domain.build(db, task, plan["domain"], lab)
        X = pd.concat([X.reset_index(drop=True), D.reset_index(drop=True)], axis=1)
    X.columns = [str(c) for c in X.columns]
    # float32: duckdb's parallel float sums differ run to run in the last ulp (~1e-14 rel.),
    # which moved LightGBM bin edges (trial val 0.6939 vs 0.6889 for the same features).
    # Rounding to float32 makes the matrix reproducible (and halves its memory).
    return X.astype("float32")
