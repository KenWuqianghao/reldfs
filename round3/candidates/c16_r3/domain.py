"""Hand-written domain features ("dom.*"), second submission round ("round 2"; round3/README.md, step 8a).

These are a PER-DATASET code path, unlike the schema-generic engine in features.py. They are
switched on per task by model.TASK_FAMILIES (prefix "dom.") and are recorded as such.

Every feature of a label row (entity e, cutoff t) reads only db rows with time < t (strict),
the same rule as the engine; harness/leakcheck.py checks it, and RELDFS_PLANT_DOM=1 plants a
leak (rows with time < t + timedelta are admitted) so that the checker must FAIL.

rel-f1 (entity drivers):
  driver     DNF rate (statusId != 1, the task's own definition) over the last 3 / 5 / 10
             results, the last 365 days and all time; top-3 finish rate (positionOrder <= 3)
             over the last 5 / 10; points, grid and finishing order over the last 3 / 5;
             qualifying position last / mean of last 3 / 5 / best of last 5, top-3 and pole
             rates over the last 5 / 10 sessions and the last 365 days;
             standings (position, points, wins) as of the last standings row before t, whether
             that row is from t's season, and the final position of the previous season.
  constructor the driver's current constructor (latest result or qualifying row before t):
             DNF rate, grid, points and top-3 finish rate over its last 10 results (both
             cars), mean qualifying position and top-3 rate over its last 10 sessions, and
             its constructor-standings position / points as of the last row before t.
  relative   driver minus constructor qualifying mean (the teammate gap); the driver's and
             the constructor's percentile ranks among the drivers / constructors active in
             the 365 days before t (qualifying mean of the last 5, points of the last 5).
  calendar   day of year of t; races in the 30 days before t; races so far in t's season;
             races in (t - 365 d, t - 335 d], the same 30-day window one year earlier (the
             schedule density a year ago; the future calendar itself is off-limits).
  circuit    the circuits of the races in that window one year earlier, and the driver's
             all-time DNF rate / mean qualifying position / top-3 qualifying rate at them
             (a causal proxy for circuit-specific history).
  era.*      (round 3, step 9b; its own prefix "era.", so every round-2 variant keeps its
             exact matrix) era-normalised outcomes. Each past result is split by the share of
             the race's laps it completed (lapfrac = laps / the race's max laps; known when
             the race ends, so only rows with date < t are read): fin (statusId 1), lapped
             (not fin, lapfrac >= .9), ret (not fin, 0 < lapfrac < .9), nost (laps 0: did not
             qualify / start / first-lap out). The field's rates over all results in the 365
             days before t; the driver's rates over the last 10 results and the last 365 days,
             each minus the field's; the current constructor's rates over its last 10 results
             minus the field's; the driver's mean lapfrac over the last 5 / 10 results minus
             the field's; grid and qualifying position as a share of the field size of that
             race (mean of the last 5); the driver's DNF rates (k10, 365 d) minus the field's
             DNF rate.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PLANT = os.environ.get("RELDFS_PLANT_DOM", "") == "1"   # planted-leak switch for the leakcheck ONLY


def detect(db, task) -> str | None:
    t = set(db.table_dict)
    if {"results", "qualifying", "standings", "constructor_standings", "races"} <= t and task.entity_table == "drivers":
        return "rel-f1"
    if {"event_attendees", "user_friends", "events"} <= t and task.entity_table == "users":
        return "rel-event"
    if {"SearchInfo", "SearchStream", "AdsInfo"} <= t and task.entity_table == "UserInfo":
        return "rel-avito"
    return None


def _lastk_mean(df, key, cols, k):
    """df sorted by (key, time); mean of cols over each key's last k rows."""
    return df.groupby(key, sort=False).tail(k).groupby(key)[cols].mean()


def _f1(db, task, lab: pd.DataFrame) -> pd.DataFrame:
    ecol, tcol = task.entity_col, task.time_col
    td = pd.Timedelta(days=max(1, task.timedelta.days))
    R = db.table_dict["results"].df[["driverId", "constructorId", "raceId", "date", "statusId",
                                     "positionOrder", "grid", "points"]].copy()
    R["dnf"] = (R["statusId"] != 1).astype("float64")
    R["top3f"] = (R["positionOrder"] <= 3).astype("float64")
    for c in ("grid", "points", "positionOrder"):
        R[c] = R[c].astype("float64")
    R.loc[R["grid"] <= 0, "grid"] = np.nan   # grid 0 = pit-lane start / unknown
    # era.* (round 3): outcome split by the share of the race's laps completed (per race, past rows only)
    laps = db.table_dict["results"].df["laps"].astype("float64").to_numpy()
    R["lapfrac"] = laps / pd.Series(laps).groupby(R["raceId"].to_numpy()).transform("max").replace(0, np.nan).to_numpy()
    fin = R["statusId"] == 1
    R["e_fin"] = fin.astype("float64")
    R["e_lap"] = (~fin & (R["lapfrac"] >= 0.9)).astype("float64")
    R["e_ret"] = (~fin & (R["lapfrac"] > 0) & (R["lapfrac"] < 0.9)).astype("float64")
    R["e_nost"] = (~fin & ~(R["lapfrac"] > 0)).astype("float64")
    R["gridn"] = R["grid"] / R.groupby("raceId")["grid"].transform("count").replace(0, np.nan)
    R = R.sort_values(["date", "raceId", "driverId"], kind="mergesort").reset_index(drop=True)
    Q = db.table_dict["qualifying"].df[["driverId", "constructorId", "raceId", "date", "position"]].copy()
    Q["position"] = Q["position"].astype("float64")
    Q["qtop3"] = (Q["position"] <= 3).astype("float64")
    Q["qpole"] = (Q["position"] <= 1).astype("float64")
    Q["qn"] = Q["position"] / Q.groupby("raceId")["position"].transform("count").replace(0, np.nan)
    Q = Q.sort_values(["date", "raceId", "driverId"], kind="mergesort").reset_index(drop=True)
    S = db.table_dict["standings"].df[["driverId", "raceId", "date", "position", "points", "wins"]].copy()
    S = S.sort_values(["date", "raceId", "driverId"], kind="mergesort").reset_index(drop=True)
    CS = db.table_dict["constructor_standings"].df[["constructorId", "raceId", "date", "position", "points"]].copy()
    CS = CS.sort_values(["date", "raceId", "constructorId"], kind="mergesort").reset_index(drop=True)
    RA = db.table_dict["races"].df[["raceId", "date", "circuitId"]].copy()
    RA = RA.sort_values(["date", "raceId"], kind="mergesort").reset_index(drop=True)
    for df in (R, Q, S, CS, RA):
        df["date"] = pd.to_datetime(df["date"])

    L = pd.DataFrame({"__rid": np.arange(len(lab)), "e": lab[ecol].astype("int64").to_numpy(),
                      "t": pd.to_datetime(lab[tcol]).to_numpy()})
    out = []
    for t, grp in L.groupby("t", sort=True):
        lim = t + td if PLANT else t
        Rt, Qt = R[R["date"] < lim], Q[Q["date"] < lim]
        St, CSt, RAt = S[S["date"] < lim], CS[CS["date"] < lim], RA[RA["date"] < lim]
        drv = grp["e"].unique()
        F = pd.DataFrame(index=pd.Index(drv, name="e"))
        Rd = Rt[Rt["driverId"].isin(drv)].sort_values(["driverId", "date", "raceId"], kind="mergesort")
        Qd = Qt[Qt["driverId"].isin(drv)].sort_values(["driverId", "date", "raceId"], kind="mergesort")
        for k in (3, 5, 10):
            F[f"dom.d.dnf_k{k}"] = _lastk_mean(Rd, "driverId", ["dnf"], k)["dnf"]
        F["dom.d.dnf_all"] = Rd.groupby("driverId")["dnf"].mean()
        R365 = Rd[Rd["date"] >= t - pd.Timedelta(days=365)]
        F["dom.d.dnf_365"] = R365.groupby("driverId")["dnf"].mean()
        F["dom.d.n_365"] = R365.groupby("driverId").size()
        for k in (5, 10):
            F[f"dom.d.top3f_k{k}"] = _lastk_mean(Rd, "driverId", ["top3f"], k)["top3f"]
        for k in (3, 5):
            m = _lastk_mean(Rd, "driverId", ["points", "grid", "positionOrder"], k)
            F[f"dom.d.pts_k{k}"] = m["points"]
            F[f"dom.d.grid_k{k}"] = m["grid"]
            F[f"dom.d.pord_k{k}"] = m["positionOrder"]
        F["dom.d.q_last"] = Qd.groupby("driverId")["position"].last()
        for k in (3, 5):
            F[f"dom.d.q_k{k}"] = _lastk_mean(Qd, "driverId", ["position"], k)["position"]
        F["dom.d.q_best5"] = Qd.groupby("driverId", sort=False).tail(5).groupby("driverId")["position"].min()
        for k in (5, 10):
            m = _lastk_mean(Qd, "driverId", ["qtop3", "qpole"], k)
            F[f"dom.d.qtop3_k{k}"] = m["qtop3"]
            F[f"dom.d.qpole_k{k}"] = m["qpole"]
        F["dom.d.qtop3_365"] = Qd[Qd["date"] >= t - pd.Timedelta(days=365)].groupby("driverId")["qtop3"].mean()
        Sd = St[St["driverId"].isin(drv)]
        last = Sd.groupby("driverId").tail(1).set_index("driverId")
        F["dom.d.st_pos"] = last["position"].astype("float64")
        F["dom.d.st_pts"] = last["points"].astype("float64")
        F["dom.d.st_wins"] = last["wins"].astype("float64")
        F["dom.d.st_same_season"] = (last["date"].dt.year == t.year).astype("float64")
        prev = Sd[Sd["date"].dt.year < t.year].groupby("driverId").tail(1).set_index("driverId")
        F["dom.d.st_prev_season_pos"] = prev["position"].astype("float64")
        # current constructor: the latest result or qualifying row before t (ties: later race)
        a = pd.concat([Rd[["driverId", "date", "raceId", "constructorId"]],
                       Qd[["driverId", "date", "raceId", "constructorId"]]], ignore_index=True)
        a = a.sort_values(["driverId", "date", "raceId"], kind="mergesort")
        con = a.groupby("driverId")["constructorId"].last()
        F["dom.c.id_known"] = con.reindex(F.index).notna().astype("float64")
        Rc = Rt.sort_values(["constructorId", "date", "raceId", "driverId"], kind="mergesort")
        Qc = Qt.sort_values(["constructorId", "date", "raceId", "driverId"], kind="mergesort")
        cr = _lastk_mean(Rc, "constructorId", ["dnf", "grid", "points", "top3f"], 10)
        cq = _lastk_mean(Qc, "constructorId", ["position", "qtop3"], 10)
        cs = CSt.groupby("constructorId").tail(1).set_index("constructorId")
        cidx = con.reindex(F.index)
        for src, col, name in ((cr, "dnf", "dnf_k10"), (cr, "grid", "grid_k10"), (cr, "points", "pts_k10"),
                               (cr, "top3f", "top3f_k10"), (cq, "position", "q_k10"), (cq, "qtop3", "qtop3_k10"),
                               (cs, "position", "st_pos"), (cs, "points", "st_pts")):
            F[f"dom.c.{name}"] = src[col].astype("float64").reindex(cidx.to_numpy()).to_numpy()
        F["dom.r.q_minus_con"] = F["dom.d.q_k5"] - F["dom.c.q_k10"]
        # field-relative: percentile ranks among drivers / constructors active in the last 365 days
        Qa = Qt[Qt["date"] >= t - pd.Timedelta(days=365)].sort_values(["driverId", "date", "raceId"], kind="mergesort")
        Ra = Rt[Rt["date"] >= t - pd.Timedelta(days=365)].sort_values(["driverId", "date", "raceId"], kind="mergesort")
        qa = _lastk_mean(Qa, "driverId", ["position"], 5)["position"].rank(pct=True)
        pa = _lastk_mean(Ra, "driverId", ["points"], 5)["points"].rank(pct=True, ascending=False)
        F["dom.r.q_k5_rank"] = qa.reindex(F.index)
        F["dom.r.pts_k5_rank"] = pa.reindex(F.index)
        Qac = Qa.sort_values(["constructorId", "date", "raceId", "driverId"], kind="mergesort")
        ca = _lastk_mean(Qac, "constructorId", ["position"], 10)["position"].rank(pct=True)
        F["dom.r.con_q_rank"] = ca.reindex(cidx.to_numpy()).to_numpy()
        # calendar (cutoff-level; past races only)
        F["dom.t.doy"] = float(t.dayofyear)
        F["dom.t.races_30"] = float(((RAt["date"] >= t - pd.Timedelta(days=30))).sum())
        F["dom.t.round"] = float((RAt["date"].dt.year == t.year).sum())
        ly = RAt[(RAt["date"] > t - pd.Timedelta(days=365)) & (RAt["date"] <= t - pd.Timedelta(days=365) + td)]
        F["dom.t.races_ly"] = float(len(ly))
        # circuit proxy: the driver's record at the circuits of last year's same window
        circ = set(ly["circuitId"].dropna().astype("int64"))
        if circ:
            rid = set(RAt.loc[RAt["circuitId"].isin(circ), "raceId"])
            F["dom.x.dnf"] = Rd[Rd["raceId"].isin(rid)].groupby("driverId")["dnf"].mean()
            qx = Qd[Qd["raceId"].isin(rid)].groupby("driverId")
            F["dom.x.q_mean"] = qx["position"].mean()
            F["dom.x.qtop3"] = qx["qtop3"].mean()
        else:
            F["dom.x.dnf"] = np.nan
            F["dom.x.q_mean"] = np.nan
            F["dom.x.qtop3"] = np.nan
        # era.* (round 3): era-normalised outcomes, the field = every result in the 365 days before t
        Rf = Rt[Rt["date"] >= t - pd.Timedelta(days=365)]
        oc = ["e_fin", "e_lap", "e_ret", "e_nost"]
        fld = Rf[oc + ["dnf", "lapfrac"]].mean() if len(Rf) else pd.Series(np.nan, index=oc + ["dnf", "lapfrac"])
        for c in oc:
            F[f"era.f.{c[2:]}_365"] = float(fld[c])
        F["era.f.dnf_365"] = float(fld["dnf"])
        dk = _lastk_mean(Rd, "driverId", oc + ["dnf"], 10)
        d3 = Rd[Rd["date"] >= t - pd.Timedelta(days=365)].groupby("driverId")[oc + ["dnf"]].mean()
        for c in oc:
            F[f"era.d.{c[2:]}_k10_rel"] = dk[c] - fld[c]
            F[f"era.d.{c[2:]}_365_rel"] = d3[c] - fld[c]
        F["era.d.dnf_k10_rel"] = dk["dnf"] - fld["dnf"]
        F["era.d.dnf_365_rel"] = d3["dnf"] - fld["dnf"]
        for k in (5, 10):
            F[f"era.d.lapfrac_k{k}_rel"] = _lastk_mean(Rd, "driverId", ["lapfrac"], k)["lapfrac"] - fld["lapfrac"]
        F["era.d.gridn_k5"] = _lastk_mean(Rd, "driverId", ["gridn"], 5)["gridn"]
        F["era.d.qn_k5"] = _lastk_mean(Qd, "driverId", ["qn"], 5)["qn"]
        ck = _lastk_mean(Rc, "constructorId", oc, 10)
        for c in oc:
            F[f"era.c.{c[2:]}_k10_rel"] = (ck[c] - fld[c]).reindex(cidx.to_numpy()).to_numpy()
        g = grp[["__rid", "e"]].merge(F.reset_index(), on="e", how="left")
        out.append(g.drop(columns="e"))
    D = pd.concat(out, ignore_index=True).sort_values("__rid").reset_index(drop=True)
    assert (D["__rid"].to_numpy() == np.arange(len(lab))).all()
    return D.drop(columns="__rid").astype("float64")


def _event(db, task, lab: pd.DataFrame) -> pd.DataFrame:
    """rel-event (entity users). event_attendees rows are dated by their event's start_time
    (constant per event), so a row enters only when start_time < t.
      status windows  rows per status (invited / yes / maybe / no) and in total over the last
                      7, 14, 28, 56 days and all time; the invited share over 28 days and all
      weekly history  invited rows in each of the last 4 weeks; of the last 8 weeks, how many
                      had more than 2 invited rows (the task's label rule rebuilt from db rows
                      before t); weeks since the last such week
      recency         days since the last invited / yes row
      popularity      over the user's events of the last 28 days and all time: mean log size
                      (all attendee rows of the event) and mean invited share of the event
      friends         (user_friends, static) the friends' invited / yes rows in the last 7 and
                      28 days, and the number of friends with any row in the last 28 days
    """
    import duckdb

    ecol, tcol = task.entity_col, task.time_col
    td = max(1, task.timedelta.days)
    ea = db.table_dict["event_attendees"].df
    EA = pd.DataFrame({"event": ea["event"].to_numpy(), "st": ea["status"].astype(object).to_numpy(),
                       "u": ea["user_id"].to_numpy(), "ts": pd.to_datetime(ea["start_time"]).to_numpy()})
    EP = EA.groupby("event", sort=False).agg(n_all=("st", "size"), n_inv=("st", lambda s: (s == "invited").sum()),
                                             ts=("ts", "first")).reset_index()
    EP["lsize"] = np.log1p(EP["n_all"].astype("float64"))
    EP["ishare"] = EP["n_inv"] / EP["n_all"]
    EU = EA[EA["u"].notna()].copy()
    EU["u"] = EU["u"].astype("int64")
    EU = EU.merge(EP[["event", "lsize", "ishare"]], on="event", how="left")
    uf = db.table_dict["user_friends"].df
    FR = pd.DataFrame({"u": uf["user"].to_numpy(), "f": uf["friend"].to_numpy()})
    FR = FR[FR["f"].notna() & FR["u"].notna()].astype("int64").drop_duplicates()
    L = pd.DataFrame({"rid": np.arange(len(lab), dtype=np.int64), "e": lab[ecol].astype("int64").to_numpy(),
                      "t": pd.to_datetime(lab[tcol]).to_numpy()})
    con = duckdb.connect()
    con.execute("SET threads=2")
    for n, df in (("L", L), ("EU", EU), ("FR", FR)):
        con.register(n, df)
    lim = f"l.t + to_days({td})" if PLANT else "l.t"
    sel = []
    for w in (7, 14, 28, 56, None):
        wf = f" AND a.ts >= l.t - to_days({w})" if w else ""
        tag = f"{w}d" if w else "all"
        for s in ("invited", "yes", "maybe", "no"):
            sel.append(f"count(*) FILTER (WHERE a.st = '{s}'{wf}) AS \"dom.e.{s}_{tag}\"")
        sel.append(f"count(*) FILTER (WHERE true{wf}) AS \"dom.e.tot_{tag}\"")
    sel += ["date_diff('day', max(a.ts) FILTER (WHERE a.st = 'invited'), l.t) AS \"dom.e.days_inv\"",
            "date_diff('day', max(a.ts) FILTER (WHERE a.st = 'yes'), l.t) AS \"dom.e.days_yes\"",
            "avg(a.lsize) FILTER (WHERE a.ts >= l.t - to_days(28)) AS \"dom.e.pop_28d\"",
            "avg(a.lsize) AS \"dom.e.pop_all\"",
            "avg(a.ishare) FILTER (WHERE a.ts >= l.t - to_days(28)) AS \"dom.e.ishare_28d\"",
            "avg(a.ishare) AS \"dom.e.ishare_all\""]
    A = con.execute(f"SELECT l.rid, {', '.join(sel)} FROM L l JOIN EU a ON a.u = l.e AND a.ts < {lim} "
                    "GROUP BY l.rid, l.t").df()
    W = con.execute(f"""
        WITH wk AS (SELECT l.rid, CAST(floor(date_diff('second', a.ts, l.t) / 604800.0) AS INTEGER) AS j,
                           count(*) AS n
                    FROM L l JOIN EU a ON a.u = l.e AND a.ts < {lim} AND a.st = 'invited'
                         AND a.ts >= l.t - to_days(56)
                    GROUP BY 1, 2)
        SELECT rid,
               sum(n) FILTER (WHERE j = 0) AS "dom.e.inv_w1", sum(n) FILTER (WHERE j = 1) AS "dom.e.inv_w2",
               sum(n) FILTER (WHERE j = 2) AS "dom.e.inv_w3", sum(n) FILTER (WHERE j = 3) AS "dom.e.inv_w4",
               count(*) FILTER (WHERE n > 2 AND j BETWEEN 0 AND 7) AS "dom.e.pos_weeks8",
               min(j) FILTER (WHERE n > 2) AS "dom.e.weeks_since_pos"
        FROM wk GROUP BY rid""").df()
    Fq = con.execute(f"""
        SELECT l.rid,
               count(*) FILTER (WHERE a.st = 'invited' AND a.ts >= l.t - to_days(7)) AS "dom.f.inv_7d",
               count(*) FILTER (WHERE a.st = 'invited' AND a.ts >= l.t - to_days(28)) AS "dom.f.inv_28d",
               count(*) FILTER (WHERE a.st = 'yes' AND a.ts >= l.t - to_days(7)) AS "dom.f.yes_7d",
               count(*) FILTER (WHERE a.st = 'yes' AND a.ts >= l.t - to_days(28)) AS "dom.f.yes_28d",
               count(DISTINCT a.u) FILTER (WHERE a.ts >= l.t - to_days(28)) AS "dom.f.active_28d"
        FROM L l JOIN FR f ON f.u = l.e JOIN EU a ON a.u = f.f AND a.ts < {lim}
             AND a.ts >= l.t - to_days(28)
        GROUP BY l.rid""").df()
    con.close()
    D = L[["rid"]]
    for part in (A, W, Fq):
        D = D.merge(part, on="rid", how="left")
    D = D.sort_values("rid").reset_index(drop=True)
    cnt = [c for c in D.columns if c != "rid" and not any(k in c for k in ("days_", "pop_", "ishare", "weeks_since"))]
    D[cnt] = D[cnt].fillna(0)
    for w in ("28d", "all"):
        D[f"dom.e.irate_{w}"] = D[f"dom.e.invited_{w}"] / D[f"dom.e.tot_{w}"].where(D[f"dom.e.tot_{w}"] > 0)
    return D.drop(columns="rid").astype("float64")


def _avito(db, task, lab: pd.DataFrame) -> pd.DataFrame:
    """rel-avito (entity UserInfo). The data start on 2015-04-25, so the first train cutoff
    (04-26) has one day of history and test (05-14) has 19: the family uses windows of at
    most 8 days and rates, whose meaning does not grow with the history length.
      search    the user's searches (SearchInfo) in the last 1, 2, 4, 8 days
      clicks    over the user's searches before t: context-ad impressions (SearchStream rows
                with ObjectType 3, the only ones with IsClick), clicks, in the last 1, 2, 4, 8
                days; clicks per search and click-through rate over 4 and 8 days; days since
                the last click; whether the last 4 days had more than one click (the task's
                label rule rebuilt from rows before t)
      ad CTR    mean / max HistCTR of the context ads shown to the user in the last 4 / 8 days
      category  click-through rate of each search category over ALL users' searches before
                t; the user's mean of it over their searches of the last 8 days
      rate.*    (round 3, step 9g; its own prefix "rate.") historical click-through rates
                of groups, over ALL users' searches before t, smoothed toward the global CTR
                before t ((clicks + 50 g) / (impressions + 50)): by search location, by IP, by
                device / browser family / OS (UserInfo); the user's mean location and IP
                CTR over their searches of the last 8 days and all time, their device group
                CTRs, their all-time searches / impressions / clicks and smoothed CTR (also
                minus the global CTR), the share of logged-on searches and the distinct search
                days in the last 8 days, and the global CTR itself.
    """
    import duckdb

    ecol, tcol = task.entity_col, task.time_col
    td = max(1, task.timedelta.days)
    si = db.table_dict["SearchInfo"].df
    ss = db.table_dict["SearchStream"].df
    SI = pd.DataFrame({"sid": si["SearchID"].to_numpy(), "u": si["UserID"].to_numpy(),
                       "ts": pd.to_datetime(si["SearchDate"]).to_numpy(), "cat": si["CategoryID"].to_numpy(),
                       "loc": si["LocationID"].to_numpy(), "ip": si["IPID"].to_numpy(),
                       "logged": pd.to_numeric(si["IsUserLoggedOn"], errors="coerce").to_numpy()})
    ui = db.table_dict["UserInfo"].df
    UD = pd.DataFrame({"u": ui["UserID"].to_numpy(), "dev": ui["UserDeviceID"].to_numpy(),
                       "uaf": ui["UserAgentFamilyID"].to_numpy(), "os": ui["UserAgentOSID"].to_numpy()})
    SS = pd.DataFrame({"sid": ss["SearchID"].to_numpy(), "ot": ss["ObjectType"].to_numpy(),
                       "clk": ss["IsClick"].to_numpy(), "hctr": ss["HistCTR"].to_numpy()})
    L = pd.DataFrame({"rid": np.arange(len(lab), dtype=np.int64), "e": lab[ecol].astype("int64").to_numpy(),
                      "t": pd.to_datetime(lab[tcol]).to_numpy()})
    con = duckdb.connect()
    con.execute("SET threads=2; SET memory_limit='3GB'")
    for n, df in (("L", L), ("SI", SI), ("SS", SS), ("UD", UD)):
        con.register(n, df)
    # one row per search: its context impressions, clicks, HistCTR sum / max
    con.execute("""CREATE TEMP TABLE S AS
        SELECT SI.sid, SI.u, SI.ts, SI.cat, any_value(SI.loc) AS loc, any_value(SI.ip) AS ip,
               any_value(SI.logged) AS logged,
               coalesce(sum(CASE WHEN SS.ot = 3 THEN 1 ELSE 0 END), 0)::DOUBLE AS nctx,
               coalesce(sum(CASE WHEN SS.ot = 3 THEN SS.clk ELSE 0 END), 0)::DOUBLE AS nclk,
               sum(CASE WHEN SS.ot = 3 THEN SS.hctr END) AS hsum, max(CASE WHEN SS.ot = 3 THEN SS.hctr END) AS hmax
        FROM SI LEFT JOIN SS ON SS.sid = SI.sid GROUP BY 1, 2, 3, 4""")
    lim = f"l.t + to_days({td})" if PLANT else "l.t"
    sel = []
    for w in (1, 2, 4, 8):
        f = f"s.ts >= l.t - to_days({w})"
        sel += [f"count(*) FILTER (WHERE {f}) AS \"dom.a.srch_{w}d\"",
                f"coalesce(sum(s.nctx) FILTER (WHERE {f}), 0) AS \"dom.a.ctx_{w}d\"",
                f"coalesce(sum(s.nclk) FILTER (WHERE {f}), 0) AS \"dom.a.clk_{w}d\""]
    for w in (4, 8):
        f = f"s.ts >= l.t - to_days({w})"
        sel += [f"sum(s.hsum) FILTER (WHERE {f}) / nullif(sum(s.nctx) FILTER (WHERE {f}), 0) AS \"dom.a.hctr_mean_{w}d\"",
                f"max(s.hmax) FILTER (WHERE {f}) AS \"dom.a.hctr_max_{w}d\""]
    sel.append("date_diff('hour', max(s.ts) FILTER (WHERE s.nclk > 0), l.t) / 24.0 AS \"dom.a.days_clk\"")
    A = con.execute(f"SELECT l.rid, {', '.join(sel)} FROM L l JOIN S s ON s.u = l.e AND s.ts < {lim}"
                    f" AND s.ts >= l.t - to_days(8) GROUP BY l.rid, l.t").df()
    D2 = con.execute(f"SELECT l.rid, max(s.ts) FILTER (WHERE s.nclk > 0) AS lc FROM L l JOIN S s ON s.u = l.e"
                     f" AND s.ts < {lim} GROUP BY l.rid").df()
    # category CTR over all users' searches before each cutoff (few distinct cutoffs)
    cuts = pd.DataFrame({"t": np.unique(L["t"].to_numpy())})
    con.register("C", cuts)
    climit = f"C.t + to_days({td})" if PLANT else "C.t"
    con.execute(f"""CREATE TEMP TABLE CC AS SELECT C.t, s.cat, sum(s.nclk) / nullif(sum(s.nctx), 0) AS cctr
        FROM C JOIN S s ON s.ts < {climit} GROUP BY 1, 2""")
    B = con.execute(f"""SELECT l.rid, avg(cc.cctr) AS "dom.a.cat_ctr_8d" FROM L l JOIN S s ON s.u = l.e
        AND s.ts < {lim} AND s.ts >= l.t - to_days(8) JOIN CC cc ON cc.t = l.t AND cc.cat = s.cat GROUP BY l.rid""").df()
    # rate.* (round 3): smoothed group CTRs over all users' searches before each cutoff
    M = 50.0
    con.execute(f"""CREATE TEMP TABLE G AS SELECT C.t, sum(s.nclk) / nullif(sum(s.nctx), 0) AS g
        FROM C JOIN S s ON s.ts < {climit} GROUP BY 1""")
    con.execute("CREATE TEMP TABLE SU AS SELECT s.*, UD.dev, UD.uaf, UD.os FROM S s LEFT JOIN UD ON UD.u = s.u")
    for key in ("loc", "ip", "dev", "uaf", "os"):
        con.execute(f"""CREATE TEMP TABLE R_{key} AS SELECT C.t, s.{key} AS k,
            (sum(s.nclk) + {M} * any_value(G.g)) / (sum(s.nctx) + {M}) AS r
            FROM C JOIN SU s ON s.ts < {climit} JOIN G ON G.t = C.t WHERE s.{key} IS NOT NULL GROUP BY 1, 2""")
    RU = con.execute(f"""SELECT l.rid,
            avg(rl.r) FILTER (WHERE s.ts >= l.t - to_days(8)) AS "rate.u.loc_8d", avg(rl.r) AS "rate.u.loc_all",
            avg(ri.r) FILTER (WHERE s.ts >= l.t - to_days(8)) AS "rate.u.ip_8d", avg(ri.r) AS "rate.u.ip_all",
            count(*) AS "rate.u.srch_all", sum(s.nctx) AS "rate.u.ctx_all", sum(s.nclk) AS "rate.u.clk_all",
            (sum(s.nclk) + {M} * any_value(G.g)) / (sum(s.nctx) + {M}) AS "rate.u.ctr_all",
            avg(s.logged) FILTER (WHERE s.ts >= l.t - to_days(8)) AS "rate.u.logged_8d",
            count(DISTINCT date_trunc('day', s.ts)) FILTER (WHERE s.ts >= l.t - to_days(8)) AS "rate.u.days_8d"
        FROM L l JOIN S s ON s.u = l.e AND s.ts < {lim} JOIN G ON G.t = l.t
             LEFT JOIN R_loc rl ON rl.t = l.t AND rl.k = s.loc LEFT JOIN R_ip ri ON ri.t = l.t AND ri.k = s.ip
        GROUP BY l.rid""").df()
    RD = con.execute("""SELECT l.rid, G.g AS "rate.g.ctr", rd.r AS "rate.u.dev", ra.r AS "rate.u.uaf", ro.r AS "rate.u.os"
        FROM L l JOIN G ON G.t = l.t LEFT JOIN UD ON UD.u = l.e
             LEFT JOIN R_dev rd ON rd.t = l.t AND rd.k = UD.dev LEFT JOIN R_uaf ra ON ra.t = l.t AND ra.k = UD.uaf
             LEFT JOIN R_os ro ON ro.t = l.t AND ro.k = UD.os""").df()
    con.close()
    D = L[["rid", "t"]].merge(A, on="rid", how="left").merge(B, on="rid", how="left").merge(D2, on="rid", how="left")
    D = D.sort_values("rid").reset_index(drop=True)
    cnt = [c for c in D.columns if c.startswith(("dom.a.srch", "dom.a.ctx", "dom.a.clk"))]
    D[cnt] = D[cnt].fillna(0)
    D["dom.a.days_clk"] = (D["t"] - D["lc"]).dt.total_seconds() / 86400.0
    for w in (4, 8):
        D[f"dom.a.cps_{w}d"] = D[f"dom.a.clk_{w}d"] / D[f"dom.a.srch_{w}d"].where(D[f"dom.a.srch_{w}d"] > 0)
        D[f"dom.a.ctr_{w}d"] = D[f"dom.a.clk_{w}d"] / D[f"dom.a.ctx_{w}d"].where(D[f"dom.a.ctx_{w}d"] > 0)
    D["dom.a.lab_last"] = (D["dom.a.clk_4d"] > 1).astype("float64")
    R = L[["rid"]].merge(RU, on="rid", how="left").merge(RD, on="rid", how="left").sort_values("rid")
    R = R.reset_index(drop=True)
    for c in ("rate.u.srch_all", "rate.u.ctx_all", "rate.u.clk_all", "rate.u.days_8d"):
        R[c] = R[c].fillna(0)
    R["rate.u.ctr_rel"] = R["rate.u.ctr_all"] - R["rate.g.ctr"]
    D = pd.concat([D.drop(columns=["rid", "t", "lc"]).reset_index(drop=True), R.drop(columns="rid")], axis=1)
    return D.astype("float64")


BUILDERS = {"rel-f1": _f1, "rel-event": _event, "rel-avito": _avito}


def build(db, task, kind: str, lab: pd.DataFrame) -> pd.DataFrame:
    return BUILDERS[kind](db, task, lab)
