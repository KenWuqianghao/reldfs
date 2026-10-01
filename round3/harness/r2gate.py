"""Round-2 combined gate (round3/README.md, step 8b): the round-1 keep rule on the combined system.

    python harness/r2gate.py --choice task=variant ... -- <rollbt logs>

Per seed (0 and 1): the 12-task val mean gain of the chosen variants over base (tasks at base,
or not listed, contribute 0), and the worst single-task drop. Val numbers come from the
rollbt [rb] lines (early-stopped on val, 250k gating cap where it applies), the same numbers
the selection read as its condition (iii).
"""
import re
import sys

args = sys.argv[1:]
i = args.index("--")
choice = dict(a.split("=", 1) for a in args[:i] if a != "--choice")
logs = args[i + 1:]
V = {}
for lg in logs:
    for line in open(lg):
        m = re.match(r"\[rb\] (\S+):(\S+) (\S+) seed=(\d+) val=(\S+)", line)
        if m:
            V[(m.group(3), m.group(2), int(m.group(4)))] = float(m.group(5))
tot, worst = {0: 0.0, 1: 0.0}, {0: 0.0, 1: 0.0}
for t, v in sorted(choice.items()):
    d = [V[(t, v, s)] - V[(t, "base", s)] for s in (0, 1)]
    print(f"{t:30s} {v:45s} base {V[(t, 'base', 0)]:.5f} {V[(t, 'base', 1)]:.5f}  "
          f"chosen {V[(t, v, 0)]:.5f} {V[(t, v, 1)]:.5f}  d s0 {d[0]:+.5f} s1 {d[1]:+.5f}")
    for s in (0, 1):
        tot[s] += d[s]
        worst[s] = min(worst[s], d[s])
for s in (0, 1):
    print(f"seed {s}: 12-task val mean gain {tot[s] / 12:+.5f} (need >= +.005); worst task {worst[s]:+.5f} (need > -.01)")
ok = all(tot[s] / 12 >= 0.005 and worst[s] > -0.01 for s in (0, 1))
print("GATE", "PASS" if ok else "FAIL")
