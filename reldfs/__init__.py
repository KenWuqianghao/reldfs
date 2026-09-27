"""reldfs: time-aware relational feature synthesis + LightGBM for RelBench.

    from reldfs import load_task, run_task
    ctx = load_task("data", "rel-f1/driver-top3")
    res = run_task(ctx.db, ctx.task, ctx.name, ctx.train, ctx.val, ctx.test)["reldfs"]
    res["test_pred"]      # 2-seed mean of the refit models' test probabilities

Lower-level pieces: reldfs.features (make_plan, build), reldfs.families (per-task family
switches and the selection rule), reldfs.model (fit_refit), reldfs.leak (the leak checker).
"""

from .data import load_task
from .families import SYSTEMS, columns, select, task_drop
from .features import build, make_plan
from .model import fit_refit, sample_train
from .pipeline import run_task

__version__ = "1.0.0"

__all__ = ["load_task", "SYSTEMS", "columns", "select", "task_drop", "build", "make_plan",
           "fit_refit", "sample_train", "run_task", "__version__"]
