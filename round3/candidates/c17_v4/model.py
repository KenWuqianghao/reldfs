"""c17_v4: c16_r3 + the v4 families (round3/README.md, "v4"). features.py and domain.py are
byte-identical to c16_r3's. Added here:
  bd.*  rel-stack: badge state, account anniversary, post-score thresholds (badgefeat.py)
  ab.*  rel-stack/user-badge: length of users.AboutMe (harness/fetch_stack_text.py)
  cd.*  rel-hm and rel-amazon churn tasks: activity cadence in units of the timedelta (cadence.py)
  ax.*  numeric summaries of static string columns the slim mirror drops (harness/fetch_aux.py)
Frames over CHUNK rows are built in pieces; RELDFS_MAX_TRAIN overrides the 1M train cap.

(c16_r3 docstring follows.) c16_r3: c15_dom + round-3 options (third submission round, "round 3"; protocol: round3/README.md, step 9b).

New variant tokens (variant_cfg), all relative to the task's c14_fam3 system ("base"):
  * "era"   the f1 era-normalised family era.* (domain.py), its own prefix;
  * "pfn"   learner = TabPFN v2 (Prior Labs, weights Prior-Labs/TabPFN-v2-clf, Prior Labs
            License 1.1; docs/licensing.md) on the variant's columns: the top PFN_K columns
            by LightGBM gain importance (a fixed PFN_IMP_ROUNDS-round LightGBM on the fitting
            rows only; no early stopping, so no held-out label is read), a context of at most
            PFN_CTX fitting rows (sampled with the seed, probability proportional to the
            recency weight when one is set), PFN_EST ensemble members, CPU, 1 torch thread.
            TabPFN is pretrained (on synthetic data) and is used in context: no gradient step
            on the target DB.
  * "hl<N>y" / "win<N>y"  time-decayed weights (halflife N years) / only the last N years of
            cutoffs of the fitting set; "hlq<k>" / "winq<k>" the same with halflife / window =
            the fitting set's cutoff span / k (dataset-agnostic).
harness/rollbt.py "a~b" = rank-average blend (per scored split, ranks scaled to [0, 1]).

(c15_dom docstring follows.) c15_dom: c14_fam3 + round-2 switches (second submission round, "round 2").

Adds, per task (TASK_FAMILIES / TASK_CFG, chosen on val and train-internal backtests only,
round3/README.md, step 8b):
  * "dom." hand-written per-dataset features (domain.py), a family like gc./pe./ft.;
  * drop_cols: identifier-like own columns (entity string columns with nunique >= half the
    entity rows, e.g. driver names), which only let the model memorise individuals;
  * params: a LightGBM parameter override;
  * seeds: the number of seeds whose predictions are averaged (bagging).
variant_cfg(name, v) gives the grid that harness/rollbt.py scores.

(c14_fam3 docstring follows.) c14_fam3: H13b per-task family switches over {gc, pe, ft} (EXPERIMENTS.md, round 7).

features.py is byte-identical to c13_fam's. TASK_FAMILIES names, per task, which of gc.* (H9),
pe.* (H7) and ft.* (H10) are switched on; the base is c9_tie (all three dropped). The switches
come from harness/famselect.py (unchanged) on the backtests only (bt1_refit and bt2_refit,
seeds 0 and 1; rule pre-registered in EXPERIMENTS.md, step 7a), never on val.

(c13_fam docstring follows.) c13_fam: H13 per-task family switches over c12_fine's model (EXPERIMENTS.md, round 6).

features.py computes every family (c9_tie + gc.* H9 + pe.* H7 + ft.* H10). The model drops the
families a task does not use: BASE_FT says whether ft.* (H10) is in the base, and TASK_FAMILIES
names, per task, which of gc.* / pe.* are switched on. Those switches were chosen on the
backtests only (bt1_refit and bt2_refit, seeds 0 and 1; rule fixed in EXPERIMENTS.md 6b before any
H13 number), never on val. Variants b0* (base c9_tie) and b1* (base c12_fine) are the grid
harness/backtest.py scores; "gc" / "pe" / "gp" = + H9 / + H7 / + both.

(c12_fine docstring follows.) c12_fine: c9_tie model; features.py adds H10 (ft.*). Variant "noft" drops ft.* (= c9_tie).

(c9_tie docstring follows.) c9_tie: c7_guard model + the gating train cap (EXPERIMENTS.md, round 3); features.py = c7_guard + plast tie fix.

Gating cap (a cap introduced in the round-1 protocol, EXPERIMENTS.md "Protocol", step 2a): with RELDFS_TRAIN_CAP=N (e.g. 250000) the
train split is capped at N rows, sampled STRATIFIED by label with a fixed RandomState(0)
(deterministic; the same rows for every model seed, so the feature cache is shared). Without
it the old 1M random cap (MAX_TRAIN) applies, reserved for final or test runs.

(c7_guard docstring follows.) c7_guard: c4_det model unchanged (refit on train+val); features.py adds the scale-up guard.

Two knobs, both schema/task generic:
  * recency weights: w = 0.5 ** (age_days / halflife_days), age measured from the latest
    cutoff in the fitting set (halflife None = uniform, as in c0);
  * refit: the val predictions always come from the model trained on train and early-stopped
    on val. With refit=True the TEST predictions come from a second model fit on train+val
    (weights recomputed from the new latest cutoff) for round(best_iter * scale) rounds,
    scale = 1 + n_val / n_train. This choice cannot be checked on val (val is in its
    training set), so it is validated with harness/backtest.py (last train period held out).

CFG is chosen on val (halflife) and on the backtest (refit); see EXPERIMENTS.md, step 1a.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from relbench.base import TaskType

import features as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "harness"))
from common import cached_build  # noqa: E402

MAX_TRAIN = int(os.environ.get("RELDFS_MAX_TRAIN", "1000000"))
CHUNK = 250_000
# LightGBM on 1 OpenMP thread: on the shared 4-core box (load > 6) 2 threads ran 18x slower
# (61.6 s vs 3.4 s for 100 rounds on f1 dnf) from OpenMP spin-waiting. duckdb keeps 2 threads.
LGB_THREADS = 1
PARAMS = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=20, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbosity=-1)
CFG = dict(halflife_days=None, refit=True)   # EXPERIMENTS.md 1a: refit kept on the backtest, recency weights rejected
# grid explored by harness/backtest.py (val + backtest); names are recorded in results.tsv
VARIANTS = {
    "uniform": dict(halflife_days=None),
    "hl2y": dict(halflife_days=730),
    "hl4y": dict(halflife_days=1461),
    "hl8y": dict(halflife_days=2922),
    "hl16y": dict(halflife_days=5844),
    "last8y": dict(window_days=2922),
    "noft": dict(halflife_days=None, drop_prefix="ft."),   # H10 ablation in the same process (= c9_tie)
}
# H13 grid (drop_prefix may be a tuple; str.startswith accepts it)
for _b, _ft in (("b0", ("ft.",)), ("b1", ())):
    for _s, _keep in (("", ()), ("gc", ("gc.",)), ("pe", ("pe.",)), ("gp", ("gc.", "pe."))):
        VARIANTS[_b + _s] = dict(halflife_days=None,
                                 drop_prefix=tuple(p for p in ("gc.", "pe.") if p not in _keep) + _ft)
BASE_FT = False         # unused in c14 (kept for the c13 VARIANTS grid); the base is c9_tie
# task -> subset of ("gc.", "pe.", "ft."), from harness/famselect.py (unchanged since round 1) on
# the backtests only (EXPERIMENTS.md 7b; ../results/grid/h13b.select.txt). Tasks not listed keep the base
# (c9_tie columns): rel-f1/driver-dnf, rel-trial/study-outcome, rel-event/user-repeat,
# rel-stack/user-engagement.
TASK_FAMILIES = {
    "rel-f1/driver-top3": ("ft.",),                       # b1
    "rel-event/user-ignore": ("gc.",),                    # b0gc
    "rel-avito/user-visits": ("pe.", "ft."),              # b1pe
    "rel-avito/user-clicks": ("gc.", "pe.", "ft."),       # b1gp
    "rel-hm/user-churn": ("ft.",),                        # b1
    "rel-stack/user-badge": ("gc.", "ft."),               # b1gc
    "rel-amazon/user-churn": ("pe.", "ft."),              # b1pe
    "rel-amazon/item-churn": ("ft.",),                    # b1
}


FAMS = ("gc.", "pe.", "ft.", "dom.", "txt.", "era.", "rate.", "bd.", "ab.", "cd.", "ax.")
TASK_CFG: dict = {}      # task -> extra cfg (drop_cols "id", params, ...); filled by the round-2 rule
ROBUST = dict(learning_rate=0.02, min_data_in_leaf=50, feature_fraction=0.5, lambda_l2=5.0)


def task_drop(name: str, fams=None) -> tuple:
    keep = TASK_FAMILIES.get(name, ()) if fams is None else fams
    return tuple(p for p in FAMS if p not in keep)


def variant_cfg(name: str, v: str) -> dict:
    """Round-2 grid, relative to the task's c14_fam3 system ("base")."""
    fams = tuple(p for p in TASK_FAMILIES.get(name, ()) if p != "dom.")
    c = dict(CFG)
    parts = v.split("+")
    assert parts[0] == "base", v
    for p in parts[1:]:
        if p == "dom":
            fams = fams + ("dom.",)
        elif p == "noid":
            c["drop_cols"] = "id"
        elif p == "txt":       # 8i: point-in-time text score (rel-trial)
            fams = fams + ("txt.",)
        elif p == "domonly":   # 8d: only dom.* and the entity's own columns
            fams = fams + ("dom.",) if "dom." not in fams else fams
            c["keep_prefix"] = ("dom.", "own.")
        elif p == "robust":
            c["params"] = dict(ROBUST)
        elif p == "era":       # 9b: f1 era-normalised family
            fams = fams + ("era.",)
            if c.get("keep_prefix"):
                c["keep_prefix"] = tuple(c["keep_prefix"]) + ("era.",)
        elif p in ("bd", "ab", "cd", "ax"):   # v4 families
            fams = fams + (p + ".",)
        elif p == "rate":      # 9g: avito group-level historical CTRs
            fams = fams + ("rate.",)
        elif p == "pfn":       # 9b: TabPFN v2 learner on the variant's columns
            c["learner"] = "pfn"
        elif re.fullmatch(r"pfnk\d+", p):   # 9e: TabPFN with K columns instead of PFN_K
            c["learner"] = "pfn"
            c["pfn_k"] = int(p[4:])
        elif re.fullmatch(r"hl\d+y", p):
            c["halflife_days"] = 365.25 * int(p[2:-1])
        elif re.fullmatch(r"win\d+y", p):
            c["window_days"] = 365.25 * int(p[3:-1])
        elif re.fullmatch(r"hlq\d+", p):
            c["halflife_frac"] = 1.0 / int(p[3:])
        elif re.fullmatch(r"winq\d+", p):
            c["window_frac"] = 1.0 / int(p[4:])
        else:
            raise KeyError(v)
    c["drop_prefix"] = task_drop(name, fams)
    return c


def id_cols(db, task) -> list:
    """Identifier-like entity columns: strings with nunique >= half the entity rows (schema-generic)."""
    df = db.table_dict[task.entity_table].df
    out = []
    for c in df.columns:
        s = df[c]
        if c == db.table_dict[task.entity_table].pkey_col or _is_num_like(s):
            continue
        try:
            n = s.nunique(dropna=True)
        except TypeError:   # list-valued column (rel-amazon product): not an identifier
            continue
        if n >= 0.5 * len(df):
            out.append("own." + str(c))
    return out


def _is_num_like(s) -> bool:
    return (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
            or pd.api.types.is_datetime64_any_dtype(s))


def weights(t: pd.Series, cfg: dict) -> np.ndarray:
    t = pd.to_datetime(t)
    age = (t.max() - t).dt.days.to_numpy().astype("float64")
    span = max(1.0, float(age.max())) if len(age) else 1.0   # the fitting set's cutoff span (days)
    w = np.ones(len(t))
    hl = cfg.get("halflife_days") or (cfg.get("halflife_frac") and cfg["halflife_frac"] * span)
    if hl:
        w = 0.5 ** (age / hl)
    win = cfg.get("window_days") or (cfg.get("window_frac") and cfg["window_frac"] * span)
    if win:
        w = np.where(age <= win, w, 0.0)
    return w


ID_COLS: list = []   # set by features() from the task's db (id_cols)


def select_cols(cols, cfg) -> list:
    dp = cfg.get("drop_prefix") or ()
    dc = set(ID_COLS) if cfg.get("drop_cols") == "id" else set(cfg.get("drop_cols") or ())
    kp = cfg.get("keep_prefix")
    return [c for c in cols if not (dp and c.startswith(dp)) and c not in dc and (not kp or c.startswith(kp))]


def fit_predict(task, Xtr, tr, Xva, va, Xouts, cfg, seed=0, threads=2):
    """Fit on (Xtr, tr) with early stopping on (Xva, va); predict every X in Xouts.

    Returns (preds_noref, preds_ref or None, best_iter). preds_ref comes from the refit on
    train+val and is only computed when cfg['refit'].
    """
    if cfg.get("learner") == "pfn":
        return pfn_fit_predict(task, Xtr, tr, Xva, va, Xouts, cfg, seed)
    cls = task.task_type == TaskType.BINARY_CLASSIFICATION
    y = task.target_col
    keep_c = select_cols(Xtr.columns, cfg)
    if len(keep_c) < Xtr.shape[1]:   # ablation: drop feature families / columns (order kept)
        Xtr, Xva, Xouts = Xtr[keep_c], Xva[keep_c], [X[keep_c] for X in Xouts]
    p = dict(PARAMS, **(cfg.get("params") or {}))
    p.update(objective="binary" if cls else "regression_l1",
             metric="auc" if cls else "l1", num_threads=LGB_THREADS, seed=seed)
    w = weights(tr[task.time_col], cfg)
    keep = w > 0
    dtr = lgb.Dataset(Xtr[keep], tr[y].to_numpy()[keep], weight=w[keep])
    dva = lgb.Dataset(Xva, va[y].to_numpy(), reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    it = max(1, m.best_iteration)
    out = [m.predict(X, num_iteration=it) for X in Xouts]
    ref = None
    if cfg.get("refit"):
        Xall = pd.concat([Xtr, Xva], ignore_index=True)
        lab = pd.concat([tr[[task.time_col, y]], va[[task.time_col, y]]], ignore_index=True)
        w2 = weights(lab[task.time_col], cfg)
        k2 = w2 > 0
        scale = cfg.get("refit_scale") or (1.0 + len(va) / max(1, int(keep.sum())))
        m2 = lgb.train(p, lgb.Dataset(Xall[k2], lab[y].to_numpy()[k2], weight=w2[k2]),
                       num_boost_round=max(1, int(round(it * scale))))
        ref = [m2.predict(X) for X in Xouts]
    return out, ref, it, m


PFN_K = int(os.environ.get("RELDFS_PFN_K", "48"))          # columns given to TabPFN
PFN_CTX = int(os.environ.get("RELDFS_PFN_CTX", "3000"))     # max context (fitting) rows
PFN_EST = int(os.environ.get("RELDFS_PFN_EST", "4"))        # TabPFN ensemble members
PFN_IMP_ROUNDS = 300
DATA_DIR = Path(os.environ.get("RELDFS_DATA") or HERE.parents[1] / "data")   # local mirrors and caches
PFN_DIR = DATA_DIR / "tabpfn"


def _pfn_one(task, X, lab, Xouts, cfg, seed):
    """One TabPFN v2 fit on the fitting rows (X, lab); returns predictions for Xouts."""
    import torch
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(PFN_DIR))
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    torch.set_num_threads(1)
    y = lab[task.target_col].to_numpy().astype(int)
    w = weights(lab[task.time_col], cfg)
    keep = np.flatnonzero(w > 0)
    # feature selection: gain importance of a fixed-round LightGBM on the fitting rows only
    p = dict(PARAMS, **(cfg.get("params") or {}))
    p.update(objective="binary", num_threads=LGB_THREADS, seed=seed, verbosity=-1)
    m = lgb.train(p, lgb.Dataset(X.iloc[keep], y[keep], weight=w[keep]), num_boost_round=PFN_IMP_ROUNDS)
    imp = m.feature_importance("gain")
    order = np.argsort(-imp, kind="mergesort")
    K = cfg.get("pfn_k") or PFN_K
    cols = [X.columns[i] for i in order[:K] if imp[i] > 0] or [X.columns[order[0]]]
    rs = np.random.RandomState(seed)
    if len(keep) > PFN_CTX:
        pw = w[keep] / w[keep].sum()
        keep = np.sort(rs.choice(keep, size=PFN_CTX, replace=False, p=pw))
    def arr(D):
        a = D[cols].to_numpy(dtype="float32", copy=True)
        a[~np.isfinite(a)] = np.nan
        return a
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, n_estimators=PFN_EST, device="cpu", ignore_pretraining_limits=True,
        random_state=seed, n_preprocessing_jobs=1)
    clf.fit(arr(X.iloc[keep]), y[keep])
    out = []
    for D in Xouts:
        a = arr(D)
        out.append(np.concatenate([clf.predict_proba(a[i:i + 2000])[:, 1] for i in range(0, len(a), 2000)]))
    return out


def pfn_fit_predict(task, Xtr, tr, Xva, va, Xouts, cfg, seed=0):
    """Same contract as fit_predict. Without refit: fit on train, predict Xouts. With refit:
    fit on train+val (the refit's data) and predict Xouts; the no-refit predictions are then
    not computed (rollbt reads only the refit ones on its folds)."""
    keep_c = select_cols(Xtr.columns, cfg)
    Xtr, Xva, Xouts = Xtr[keep_c], Xva[keep_c], [X[keep_c] for X in Xouts]
    if cfg.get("refit"):
        Xall = pd.concat([Xtr, Xva], ignore_index=True)
        lab = pd.concat([tr[[task.time_col, task.target_col]], va[[task.time_col, task.target_col]]],
                        ignore_index=True)
        ref = _pfn_one(task, Xall, lab, Xouts, cfg, seed)
        return ref, ref, 0, None
    out = _pfn_one(task, Xtr, tr.reset_index(drop=True), Xouts, cfg, seed)
    return out, None, 0, None


TEXT_FILE = DATA_DIR / "rel-trial-text" / "studies_text.parquet"
TXT_MIN_FIT = 200


def txt_scores(ctx, frames: dict) -> dict:
    """8i: point-in-time stacked text score for rel-trial (round3/README.md, step 8i).

    A row at cutoff s is scored by a TF-IDF + logistic-regression model fit only on labelled
    rows whose label was known by s (cutoff s' with s' + timedelta <= s): train rows on train
    rows, val rows on train rows, test rows on train + val rows (the refit's data). Fits are
    shared between rows with the same fit set. Returns {split: float32 array} (NaN when the
    fit set has fewer than TXT_MIN_FIT rows)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    task = ctx.task
    if task.entity_table != "studies" or not TEXT_FILE.exists():
        return {}
    T = pd.read_parquet(TEXT_FILE)
    T = T.set_index("nct_id")
    txt = (T["brief_title"].fillna("") + " . " + T["official_title"].fillna("") + " . "
           + T["brief_summaries"].fillna("")).str.lower()
    ec, tc, y, td = task.entity_col, task.time_col, task.target_col, task.timedelta
    lab = {"train": frames["train"], "val": pd.concat([frames["train"], frames["val"]], ignore_index=True)}
    src = {"train": "train", "val": "train", "test": "val"}   # fit pool per scored split
    fits = {}
    out = {}
    for split, fr in frames.items():
        if split not in src:
            continue
        pool = lab[src[split]]
        sc = np.full(len(fr), np.nan, dtype="float64")
        for s in np.unique(fr[tc].to_numpy()):
            fit = pool[pool[tc] + td <= s]
            assert (fit[tc] + td <= s).all()   # labels known by s only
            if len(fit) < TXT_MIN_FIT or fit[y].nunique() < 2:
                continue
            key = (src[split], pd.Timestamp(fit[tc].max()), len(fit))
            if key not in fits:
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, max_features=200_000,
                                      dtype=np.float32)
                Xf = vec.fit_transform(txt.reindex(fit[ec].to_numpy()).fillna("").to_numpy())
                lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000)
                lr.fit(Xf, fit[y].to_numpy().astype(int))
                fits[key] = (vec, lr)
            vec, lr = fits[key]
            m = (fr[tc] == s).to_numpy()
            sc[m] = lr.predict_proba(vec.transform(txt.reindex(fr.loc[m, ec].to_numpy()).fillna("").to_numpy()))[:, 1]
        out[split] = sc.astype("float32")
    return out


def features(ctx, splits=("train", "val", "test")):
    task = ctx.task
    plan = F.make_plan(ctx.db, task)
    tr = ctx.train
    cap = int(os.environ.get("RELDFS_TRAIN_CAP", "0") or 0)
    if cap and len(tr) > cap:   # gating cap: stratified by label, fixed sample
        y = tr[task.target_col].to_numpy()
        rs = np.random.RandomState(0)
        idx = []
        for v in np.unique(y):
            iv = np.flatnonzero(y == v)
            idx.append(rs.choice(iv, size=int(round(cap * len(iv) / len(y))), replace=False))
        tr = tr.iloc[np.sort(np.concatenate(idx))].reset_index(drop=True)
    elif len(tr) > MAX_TRAIN:
        tr = tr.iloc[np.sort(np.random.RandomState(0).permutation(len(tr))[:MAX_TRAIN])].reset_index(drop=True)
    frames = {"train": tr, "val": ctx.val, "test": ctx.test}
    ID_COLS[:] = id_cols(ctx.db, task)
    def build(lab, tag):
        # frames over CHUNK rows are built in pieces (features are row-wise functions of the entity,
        # the cutoff and the db; the engine's duckdb limits were set for 250k-row calls)
        if len(lab) <= CHUNK:
            return cached_build(F, HERE / "features.py", ctx.db, task, plan, lab, tag, ctx.workers,
                                extra=(HERE / "domain.py",))
        return pd.concat([cached_build(F, HERE / "features.py", ctx.db, task, plan,
                                       lab.iloc[i:i + CHUNK].reset_index(drop=True), f"{tag}|c{i}", ctx.workers,
                                       extra=(HERE / "domain.py",)) for i in range(0, len(lab), CHUNK)],
                         ignore_index=True)

    X = {s: build(frames[s], f"{ctx.name}|{s}") for s in splits}
    ts = txt_scores(ctx, {s: frames[s] for s in splits})   # 8i: appended last, prefix txt.
    if ctx.name.startswith("rel-stack/"):                    # v4: bd.* and ab.*
        import badgefeat

        con = badgefeat.connect(ctx.db)
        for s in splits:
            B = badgefeat.bd_features(ctx.db, task, frames[s].reset_index(drop=True), con)
            X[s] = pd.concat([X[s].reset_index(drop=True), B], axis=1)
        con.close()
        about = DATA_DIR / "rel-stack-text" / "users_about.parquet"
        if about.exists():   # the profile text is state as of the dump date, not of the seed
            ln = pd.read_parquet(about).set_index("Id")["about_len"]
            for s in splits:
                n = ln.reindex(frames[s][task.entity_col].to_numpy()).fillna(0).to_numpy().astype("float32")
                X[s]["ab.len"] = n
                X[s]["ab.open"] = ((n > 0) & (X[s]["bd.bn_autobiographer"].to_numpy() == 0)).astype("float32")
    import cadence
    if ctx.name in cadence.TASKS:                            # v4: cd.* and ax.*
        con = cadence.connect(ctx.db, ctx.name)
        for s in splits:
            C = cadence.cd_features(ctx.db, task, ctx.name, frames[s].reset_index(drop=True), con)
            X[s] = pd.concat([X[s].reset_index(drop=True), C], axis=1)
        con.close()
        aux = {"rel-hm/user-churn": "hm_customer", "rel-amazon/user-churn": "amazon_customer",
               "rel-amazon/item-churn": "amazon_product"}[ctx.name]
        f = DATA_DIR / "aux" / f"{aux}.parquet"
        if f.exists():                                       # static per entity
            A = pd.read_parquet(f).set_index(task.entity_col)
            for s in splits:
                R = A.reindex(frames[s][task.entity_col].to_numpy()).reset_index(drop=True).add_prefix("ax.")
                X[s] = pd.concat([X[s], R.astype("float32")], axis=1)
    for s in splits:
        if s in ts:
            X[s] = X[s].copy()
            X[s]["txt.score"] = ts[s]
    return X, frames


def run(ctx, cfg=None):
    cfg = dict(CFG, drop_prefix=task_drop(ctx.name), **TASK_CFG.get(ctx.name, {}), **(cfg or {}))
    t0 = time.time()
    X, fr = features(ctx)
    tf = time.time() - t0
    out, ref, it, m = fit_predict(ctx.task, X["train"], fr["train"], X["val"], fr["val"],
                                  [X["val"], X["test"]], cfg, ctx.seed, ctx.workers)
    val_pred = out[0]
    test_pred = ref[1] if ref is not None else out[1]
    cols = select_cols(X["train"].columns, cfg)
    imp = sorted(zip(m.feature_importance("gain"), cols), reverse=True)[:5]
    note = (f"cfg={cfg} n_train={len(fr['train'])} feats={len(cols)} build={tf:.0f}s iters={it} "
            f"top={'|'.join(c for _, c in imp)}")
    return val_pred, test_pred, {"note": note}
