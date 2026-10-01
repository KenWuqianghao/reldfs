"""Round-2 per-task selection (rule pre-registered in round3/README.md, step 8b, before any round-2 number).

    python harness/rbselect.py <log> [<log> ...]

Reads harness/rollbt.py's [rb] lines. Per task and variant: fold scores = every *_refit key,
averaged over the seeds present; R = mean over those folds. A variant is eligible iff
  (i)   R(v) - R(base) >= +.005,
  (ii)  it beats base on at least 2/3 of the folds (fold-level seed means),
  (iii) its seed-mean val is not below base's.
The eligible variant with the largest R gain wins; none eligible -> base.
"""
import re
import sys
from collections import defaultdict

R = defaultdict(lambda: defaultdict(dict))   # (task, variant) -> key -> {seed: value}
for lg in sys.argv[1:]:
    for line in open(lg):
        m = re.match(r"\[rb\] (\S+):(\S+) (\S+) seed=(\d+) (.*?)( \(|$)", line)
        if not m:
            continue
        for kv in m.group(5).split():
            k, v = kv.split("=")
            d = R[(m.group(3), m.group(2))][k]
            s = int(m.group(4))
            if s in d and abs(d[s] - float(v)) > 1e-9:   # a re-run must reproduce (determinism check)
                print(f"WARNING not reproduced: {m.group(3)} {m.group(2)} seed={s} {k} {d[s]} vs {v}")
            d[s] = float(v)


def mean(x):
    x = list(x.values()) if isinstance(x, dict) else x
    return sum(x) / len(x)


tasks = sorted({t for t, _ in R})
for t in tasks:
    base = R[(t, "base")]
    folds = sorted(k for k in base if k.endswith("_refit"))
    rb = mean([mean(base[k]) for k in folds])
    vb = mean(base["val"])
    print(f"{t}: base R={rb:.5f} val={vb:.5f} folds={folds} seeds={len(base['val'])}")
    best, bestg = "base", None
    for v in sorted(v for (tt, v) in R if tt == t and v != "base"):
        d = R[(t, v)]
        if any(len(d[k]) != len(base[k]) for k in folds + ["val"]):
            print(f"  {v}: incomplete")
            continue
        fd = [mean(d[k]) - mean(base[k]) for k in folds]
        g = mean(fd)
        wins = sum(x > 0 for x in fd)
        dv = mean(d["val"]) - vb
        ok = g >= 0.005 and wins >= 2 * len(folds) / 3 and dv >= 0
        print(f"  {v:28s} R gain {g:+.5f} folds " + " ".join(f"{x:+.4f}" for x in fd)
              + f" wins {wins}/{len(folds)} val {mean(d['val']):.5f} ({dv:+.5f}) {'ELIGIBLE' if ok else '-'}")
        if ok and (bestg is None or g > bestg):
            best, bestg = v, g
    print(f"  -> {t}: {best}\n")
