"""v3 (round 5): rel-f1 with the full database.

    python harness/fulldb.py --task rel-f1/driver-dnf --variant "base+domonly+era+noid+robust" \
        [--seeds 0,1] [--leakcheck] [--write-test] [--iter 11]

Leaderboard rule 1 allows `get_db(upto_test_timestamp=False)`: a prediction may use every
database row dated strictly before its own seed time. The rel-f1 test seeds run from 2010-03
to 2013-03, but the default db ends at 2010-01-01, so rounds 1-3 built every test feature
from data up to 3 years old. This script measures, on val only, what the fresh rows are worth
and writes test CSVs for the arms (test is never scored here).

Arms (val number -> the test CSV built the same way):
  stale   val features from the db cut at val_timestamp (the regime the old test CSVs were in)
  static  features cut at each row's own seed time; one model (val: fit on train; test: refit
          on train+val)
  roll    one model per seed time t, fit on every label row whose window ended strictly
          before t (seed + timedelta < t). val: train + earlier val rows. test: train + val +
          gap + rows rebuilt with task.make_table from db rows dated before t.

The plan (category vocabulary) always comes from the default db; the full db only supplies
rows, and features.build / domain.py cut them at `time < seed` per label row. --leakcheck
verifies that on test-era seeds (features from the full db == features from the db truncated
before the seed) and that every rebuilt label a seed may use is unchanged by that truncation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from backtest import score  # noqa: E402
from common import board_rows, cached_build, load_ctx  # noqa: E402
from leakcheck import truncated  # noqa: E402
from rollbt import combine  # noqa: E402
from run import append, load_cand  # noqa: E402

VAL_ARMS = ("stale", "static", "roll")
TEST_ARMS = ("static", "roll")


def rebuilt_labels(task, db, seeds) -> pd.DataFrame:
    df = task.make_table(db, pd.Series(pd.to_datetime(list(seeds)))).df
    df = df[[task.time_col, task.entity_col, task.target_col]].copy()
    df[task.target_col] = df[task.target_col].astype("int64")
    return df.sort_values([task.time_col, task.entity_col]).reset_index(drop=True)


def same_labels(task, a: pd.DataFrame, b: pd.DataFrame) -> bool:
    cols = [task.time_col, task.entity_col, task.target_col]
    a = a[cols].sort_values(cols[:2]).reset_index(drop=True)
    b = b[cols].sort_values(cols[:2]).reset_index(drop=True)
    return len(a) == len(b) and bool((a.to_numpy() == b.to_numpy()).all())


class Data:
    """Label frames and feature matrices for one task, every split row-aligned."""

    def __init__(self, mod, name: str, dense: int = 1):
        self.mod = mod
        self.dense = dense
        self.ctx = ctx = load_ctx(name)
        self.task = task = ctx.task
        self.full = task.get_db(upto_test_timestamp=False)
        tc, td = task.time_col, task.timedelta
        self.td = td
        self.val_ts = pd.Timestamp(ctx.dataset.val_timestamp)
        self.test_ts = pd.Timestamp(ctx.dataset.test_timestamp)
        assert max(len(t.df) for t in self.full.table_dict.values()) > \
            max(len(t.df) for t in ctx.db.table_dict.values()), "full db is not larger than the default db"

        self.tr, self.va, self.te = ctx.train, ctx.val, ctx.test
        # make_table must reproduce the hosted train / val labels before its output is trusted
        assert same_labels(task, rebuilt_labels(task, ctx.db, self.tr[tc].unique()), self.tr), "train rebuild differs"
        assert same_labels(task, rebuilt_labels(task, ctx.db, self.va[tc].unique()), self.va), "val rebuild differs"
        gap_seeds = pd.date_range(self.va[tc].max() + td, self.test_ts - td, freq=td)
        self.gap = rebuilt_labels(task, ctx.db, gap_seeds)        # windows end <= test_timestamp
        grid = pd.date_range(self.test_ts, periods=task.num_eval_timestamps, freq=td)
        self.ext = rebuilt_labels(task, self.full, grid)          # used per seed t only where seed + td < t
        k_ext = set(map(tuple, self.ext[[tc, task.entity_col]].to_numpy().tolist()))
        k_te = set(map(tuple, self.te[[tc, task.entity_col]].to_numpy().tolist()))
        assert k_te == k_ext, f"rebuilt test-grid keys differ from the test table ({len(k_ext)} vs {len(k_te)})"

        self.plan = mod.F.make_plan(ctx.db, task)
        mod.ID_COLS[:] = mod.id_cols(ctx.db, task)
        stale_db = ctx.db.upto(self.val_ts)
        self.X = {
            "tr": self.feats(ctx.db, self.tr, "train|r5"),
            "va": self.feats(ctx.db, self.va, "val|r5"),
            "va_stale": self.feats(stale_db, self.va, "val|r5stale"),
            "gap": self.feats(ctx.db, self.gap, "gap|r5"),
            "ext": self.feats(self.full, self.ext, "ext|r5full"),
            "te": self.feats(self.full, self.te, "test|r5full"),
        }
        # dense seeds: extra label rows at seeds shifted back by j * timedelta / dense (j = 1..dense-1),
        # rebuilt from the default db. Every shifted window ends before its unshifted one does.
        shifts = [td * j / dense for j in range(1, dense)]
        self.dtr = self.dva = self.tr.iloc[:0]
        if shifts:
            self.dtr = rebuilt_labels(task, ctx.db, [s - d for s in self.tr[tc].unique() for d in shifts])
            dva_seeds = [s - d for s in self.va[tc].unique() for d in shifts]
            assert min(dva_seeds) > self.val_ts and self.dtr[tc].max() + td <= self.val_ts
            self.dva = rebuilt_labels(task, ctx.db, dva_seeds)
            self.X["dtr"] = self.feats(ctx.db, self.dtr, f"dtr{dense}|r5")
            self.X["dva"] = self.feats(ctx.db, self.dva, f"dva{dense}|r5")

    def feats(self, db, lab, tag):
        m = self.mod
        return cached_build(m.F, m.HERE / "features.py", db, self.task, self.plan, lab,
                            f"{self.ctx.name}|{tag}", self.ctx.workers, extra=(m.HERE / "domain.py",))


def leakcheck(D: Data, k: int = 6) -> int:
    """Full-db features and rebuilt labels at test-era seeds must not read rows at or after the seed."""
    task, tc, F = D.task, D.task.time_col, D.mod.F
    ts = np.sort(D.te[tc].unique())
    pick = sorted(set(np.random.RandomState(0).choice(ts, size=min(k - 1, len(ts)), replace=False)) | {ts[-1]})
    bad = 0
    for t in pick:
        t = pd.Timestamp(t)
        lab = D.te[D.te[tc] == t]
        comp = D.te[D.te[tc] == ts[-1]] if t != ts[-1] else D.te.iloc[:0]
        a = F.build(D.full, task, D.plan, pd.concat([lab, comp], ignore_index=True), 2).iloc[:len(lab)]
        cut = truncated(D.full, t, task.entity_table)
        b = F.build(cut, task, D.plan, lab, 2)
        same = np.isclose(a.to_numpy(), b.to_numpy(), equal_nan=True, rtol=1e-9, atol=1e-9)
        diff = [c for c, ok in zip(a.columns, same.all(axis=0)) if not ok]
        usable = D.ext[D.ext[tc] + D.td < t]
        lab_ok = same_labels(task, rebuilt_labels(task, cut, usable[tc].unique()), usable) if len(usable) else True
        bad += bool(diff) or not lab_ok
        print(f"[leak5] {D.ctx.name} seed={t.date()} rows={len(lab)} feats={a.shape[1]} "
              f"{'ok' if not diff else f'LEAK in {len(diff)} cols: {diff[:6]}'}; "
              f"rebuilt labels usable at this seed: {len(usable)} rows {'ok' if lab_ok else 'CHANGED BY TRUNCATION'}")
    print("[leak5] PASS" if not bad else f"[leak5] FAIL ({bad} seeds)")
    return bad


def run_cfg(D: Data, cfg: dict, seed: int, write_test: bool, roll: bool = True) -> dict:
    """run_cfg_fits memoised on disk (runs/r5memo/): a stopped TabPFN run resumes at the next fit."""
    import hashlib

    h = hashlib.sha1(repr((D.ctx.name, sorted(cfg.items(), key=str), seed, write_test, roll,
                           [len(x) for x in (D.tr, D.va, D.gap, D.ext, D.te)])).encode())
    if D.dense > 1:
        h.update(f"dense{D.dense}|{len(D.dtr)}|{len(D.dva)}".encode())
    for f in ("features.py", "domain.py", "model.py"):
        h.update((D.mod.HERE / f).read_bytes())
    path = ROOT / "runs" / "r5memo" / f"{D.ctx.name.replace('/', '__')}-{h.hexdigest()[:16]}.npz"
    if path.exists():
        return dict(np.load(path))
    res = run_cfg_fits(D, cfg, seed, write_test, roll)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **res)
    return res


def run_cfg_fits(D: Data, cfg: dict, seed: int, write_test: bool, roll: bool) -> dict:
    import lightgbm as lgb

    mod, task = D.mod, D.task
    tc, y, td = task.time_col, task.target_col, D.td
    X = D.X
    cols = mod.select_cols(X["tr"].columns, cfg)
    pfn = cfg.get("learner") == "pfn"
    res = {}
    Xtr, tr = X["tr"], D.tr
    if D.dense > 1:
        Xtr = pd.concat([X["tr"], X["dtr"]], ignore_index=True)
        tr = pd.concat([D.tr, D.dtr], ignore_index=True)
    if pfn:
        res["val:static"], res["val:stale"] = mod._pfn_one(
            task, Xtr[cols], tr, [X["va"][cols], X["va_stale"][cols]], cfg, seed)
        it = 0
    else:
        out, _, it, _ = mod.fit_predict(task, Xtr, tr, X["va"], D.va, [X["va"], X["va_stale"]],
                                        dict(cfg, refit=False), seed)
        res["val:static"], res["val:stale"] = out

    def fit_pred(parts, Xout):
        Xfit = pd.concat([X[k][cols] if m is None else X[k][cols][m] for k, _, m in parts], ignore_index=True)
        lab = pd.concat([f[[tc, y]] if m is None else f[[tc, y]][m] for _, f, m in parts], ignore_index=True)
        if pfn:
            return mod._pfn_one(task, Xfit, lab, [Xout[cols]], cfg, seed)[0]
        p = dict(mod.PARAMS, **(cfg.get("params") or {}))
        p.update(objective="binary", num_threads=mod.LGB_THREADS, seed=seed)
        w = mod.weights(lab[tc], cfg)
        keep = w > 0
        rounds = max(1, int(round(it * keep.sum() / len(tr))))   # the refit rule: rounds scale with rows
        m = lgb.train(p, lgb.Dataset(Xfit[keep], lab[y].to_numpy()[keep], weight=w[keep]), num_boost_round=rounds)
        return m.predict(Xout[cols])

    pv = np.full(len(D.va), np.nan)
    for t in np.sort(D.va[tc].unique()) if roll else ():
        at = (D.va[tc] == t).to_numpy()
        prior = (D.va[tc] + td < t).to_numpy()
        pv[at] = fit_pred([("tr", D.tr, None), ("va", D.va, prior)], X["va"][at])
    if roll:
        res["val:roll"] = pv
    if write_test:
        base = [("tr", D.tr, None), ("va", D.va, None)]
        if D.dense > 1:
            base += [("dtr", D.dtr, None), ("dva", D.dva, None)]
        res["test:static"] = fit_pred(base, X["te"])
        pt = np.full(len(D.te), np.nan)
        for t in np.sort(D.te[tc].unique()) if roll else ():
            at = (D.te[tc] == t).to_numpy()
            prior = (D.ext[tc] + td < t).to_numpy()
            pt[at] = fit_pred(base + [("gap", D.gap, None), ("ext", D.ext, prior)], X["te"][at])
        if roll:
            res["test:roll"] = pt
    assert all(np.isfinite(v).all() for v in res.values())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--cand", default="c16_r3")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--iter", default="11")
    ap.add_argument("--leakcheck", action="store_true")
    ap.add_argument("--write-test", action="store_true")
    ap.add_argument("--no-roll", action="store_true", help="skip the per-seed-time arm (TabPFN: 55 fits per seed)")
    ap.add_argument("--dense", type=int, default=1, help="seeds per timedelta in the fitting rows")
    ap.add_argument("--pfn-seeds", default="", help="seeds for TabPFN members (default: --seeds)")
    ap.add_argument("--tag", default="", help="suffix of the preds/r5-<arm><tag> directories")
    args = ap.parse_args()

    mod = load_cand(args.cand)
    name, t0 = args.task, time.time()
    assert args.dense == 1 or args.no_roll, "--dense is defined for the static arm only"
    D = Data(mod, name, args.dense)
    task = D.task
    print(f"[r5] {name} train={len(D.tr)} val={len(D.va)} gap={len(D.gap)} ext={len(D.ext)} test={len(D.te)} "
          f"feats={D.X['tr'].shape[1]} ({time.time() - t0:.0f}s)", flush=True)
    if args.leakcheck:
        return 1 if leakcheck(D) else 0

    groups = [[mod.variant_cfg(name, sv) for sv in g.split("|")] for g in args.variant.split("~")]
    seeds = [int(s) for s in args.seeds.split(",")]
    pfn_seeds = [int(s) for s in args.pfn_seeds.split(",")] if args.pfn_seeds else seeds
    assert set(pfn_seeds) <= set(seeds)

    def member_seed(c, seed):   # a TabPFN member outside its seed list reuses one of its own seeds
        return seed if c.get("learner") != "pfn" or seed in pfn_seeds else pfn_seeds[seed % len(pfn_seeds)]
    val_arms = [a for a in VAL_ARMS if a != "roll" or not args.no_roll]
    test_arms = [a for a in TEST_ARMS if a != "roll" or not args.no_roll]
    keys = [f"val:{a}" for a in val_arms] + ([f"test:{a}" for a in test_arms] if args.write_test else [])
    P = {k: {s: [] for s in seeds} for k in keys}   # arm -> seed -> per group prediction
    for seed in seeds:
        for cfgs in groups:
            rs = [run_cfg(D, c, member_seed(c, seed), args.write_test, not args.no_roll) for c in cfgs]
            for k in keys:
                P[k][seed].append(np.mean([r[k] for r in rs], axis=0))
        print(f"[r5] {name} seed={seed} " + " ".join(
            f"{a}={score(task, D.va, combine(P[f'val:{a}'][seed])):.4f}" for a in val_arms)
            + f" ({time.time() - t0:.0f}s)", flush=True)

    def pooled(k):
        return combine([np.mean([P[k][s][g] for s in seeds], axis=0) for g in range(len(groups))])

    ref = board_rows(name)
    brow = [f"{ref.get(r, float('nan')):.4f}" for r in ("LightGBM", "GNN", "Kapso (Leeroo Team)")]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    secs = f"{time.time() - t0:.0f}"
    note = (f"round 5 full-db (harness/fulldb.py) cand={args.cand} variant={args.variant}"
            + (f" dense={args.dense} n_fit={len(D.tr) + len(D.dtr)}" if args.dense > 1 else "")
            + (f" pfn-seeds={args.pfn_seeds}" if args.pfn_seeds else ""))
    dtag = f"+dense{args.dense}" if args.dense > 1 else ""
    for a in val_arms:
        for s in seeds:
            append([now, args.iter, f"r5:{a}{dtag}", name, "val", "roc_auc",
                    f"{score(task, D.va, combine(P[f'val:{a}'][s])):.4f}", *brow, secs, "", f"{note} seed={s}"])
        v = score(task, D.va, pooled(f"val:{a}"))
        append([now, args.iter, f"r5:{a}{dtag}", name, "val", "roc_auc", f"{v:.4f}", *brow, secs, "",
                f"{note} {len(seeds)}-seed pooled prediction"])
        print(f"[r5] {name} {a}: val pooled over {len(seeds)} seeds = {v:.4f}")
    if args.write_test:
        from relbench.submit import write_prediction_table
        for a in test_arms:
            pred = pooled(f"test:{a}")
            assert np.all((pred >= 0) & (pred <= 1)) and len(pred) == len(D.te)
            d = ROOT / "preds" / f"r5-{a}{args.tag}"
            d.mkdir(parents=True, exist_ok=True)
            write_prediction_table(task, pred, d / f"{name.replace('/', '__')}.csv")
        print(f"[r5] {name} test CSVs written under preds/r5-<arm>{args.tag}/ (not scored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
