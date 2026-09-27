"""Per-task family selection from backtest logs (the rule of reldfs.families.select).

    python harness/famselect.py <base variant> log1 [log2 ...]

    # the recorded selections:
    python harness/famselect.py c13_fam:b0 results/grid/h13_*.bt                    # H13
    python harness/famselect.py c13_fam:b0 results/grid/h13_*.bt results/grid/h13b_*.bt \
        results/grid/h13b-hm-rerun.bt                                                # H13b

Reads the "[bt] <variant> <task> seed=<s> key=value ..." lines that harness/backtest.py prints
(later lines override earlier ones for the same key). Rule, fixed before the first grid ran:
for every non-base configuration take the 4 deltas against the base on bt1_refit and
bt2_refit, seeds 0 and 1; a configuration is eligible only if all 4 are > 0; the eligible one
with the largest mean delta wins, otherwise the task keeps the base. The selection reads only
bt*_refit keys. Val is printed afterwards, for the chosen configuration only.

The logic is the script used for the recorded selections, moved into reldfs.families.select;
tests/test_families.py checks that this script reproduces the recorded outputs
(results/grid/*.select.txt) byte for byte.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reldfs.families import select  # noqa: E402


def parse(logs) -> dict:
    R = {}
    for lg in logs:
        for line in open(lg):
            m = re.match(r"\[bt\] (\S+) (\S+) seed=(\d) (.*)", line)
            if not m:
                continue
            for kv in m.group(4).split():
                k, v = kv.split("=")
                R[(m.group(1), m.group(2), int(m.group(3)), k)] = float(v)
    return R


def main(argv) -> str:
    base, logs = argv[0], argv[1:]
    cand = base.split(":")[0]
    R = {k: v for k, v in parse(logs).items() if k[0].startswith(cand + ":")}
    tasks = sorted({k[1] for k in R})
    out = ["SELECTION (backtests only)"]
    choice = {}
    for t in tasks:
        best, rows = select(R, base, t)
        for v, d, mean, ok in rows:
            if d is None:
                out.append(f"  {t} {v}: incomplete")
                continue
            out.append(f"  {t:30s} {v:18s} bt deltas " + " ".join(f"{x:+.5f}" for x in d)
                       + f"  mean {mean:+.5f}  {'eligible' if ok else '-'}")
        choice[t] = best
        out.append(f"  -> {t}: {best}")
    out.append("\nGATE INPUT (val of the chosen configuration vs base; read after the choice)")
    tot = [0.0, 0.0]
    for t in tasks:
        c = choice[t]
        d = [R[(c, t, s, "val")] - R[(base, t, s, "val")] for s in (0, 1)]
        vb = sum(R[(base, t, s, "val")] for s in (0, 1)) / 2
        vc = sum(R[(c, t, s, "val")] for s in (0, 1)) / 2
        tot = [tot[0] + d[0], tot[1] + d[1]]
        out.append(f"  {t:30s} {c:18s} base {vb:.4f} chosen {vc:.4f}  dval s0 {d[0]:+.5f} s1 {d[1]:+.5f}")
    out.append(f"  12-task mean val gain (tasks not listed = 0): s0 {tot[0] / 12:+.5f}  s1 {tot[1] / 12:+.5f}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.write(main(sys.argv[1:]))
