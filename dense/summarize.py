"""
Summarise dense/results/bench256_*.jsonl as a markdown table.

For every run three numbers matter:
  * "stopped": the model a user would get. ac2f/ac2f8 use their own plateau rule; all other
    methods get the paper's sparsity criterion applied after the fact (the step before the
    fraction of non-zero pixels first exceeds 3%; the last step if it never does).
  * "best": the best step in hindsight (needs the truth; shows the method's potential).
  * "final": the last step run (60 unless stopped earlier).
Similarity is invariant to global phase, translation and the twin image; "solved" = > 0.99.

    python dense/summarize.py dense/results/bench256_*.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

SPARSITY = 0.03
ORDER = ["l1", "elastic0.1", "elastic0.5", "group2", "group4", "c2f2", "c2f", "ac2f", "ac2f8"]
LABEL = {
    "l1": "l1 (H-FISTA)",
    "elastic0.1": "elastic net β=0.1",
    "elastic0.5": "elastic net β=0.5",
    "group2": "blocks 2×2",
    "group4": "blocks 4×4",
    "c2f2": "fixed 2×2 → 1",
    "c2f": "fixed 4 → 2 → 1",
    "ac2f": "adaptive 4 → 2 → 1",
    "ac2f8": "adaptive 8 → 4 → 2 → 1",
}
REGIME = {"U12": "uniform 12.5%", "U25": "uniform 25%", "L25": "log-uniform 25%", "L50": "log-uniform 50%"}


def stopped_step(r, n_pix):
    steps = r["steps"]
    if r["method"].startswith("ac2f"):
        return steps[-1]
    for i, s in enumerate(steps):
        if s["n"] / n_pix > SPARSITY:
            return steps[max(i - 1, 0)]
    return steps[-1]


def main(files):
    runs = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get("error") or not r["steps"]:
                continue
            runs[(r["regime"], r["seed"], r["method"])] = r  # last record wins
    table = defaultdict(lambda: defaultdict(list))
    for (regime, seed, method), r in runs.items():
        n_pix = 256 * 256
        st = stopped_step(r, n_pix)
        table[regime][method].append(dict(seed=seed, stopped=st["sim"], best=r["best"]["sim"], final=r["last"]["sim"],
                                          step=st["step"], wall=r["wall"], resid=st["resid"]))
    for regime in ["U12", "U25", "L25", "L50"]:
        if regime not in table:
            continue
        seeds = sorted({x["seed"] for m in table[regime].values() for x in m})
        print(f"\n### {REGIME[regime]} ({len(seeds)} seeds: {seeds})\n")
        print("| method | runs | solved (stopped) | mean similarity, stopped | mean best | mean final | per-seed stopped |")
        print("|---|---|---|---|---|---|---|")
        for m in ORDER:
            if m not in table[regime]:
                continue
            v = sorted(table[regime][m], key=lambda x: x["seed"])
            st = np.array([x["stopped"] for x in v])
            print(f"| {LABEL[m]} | {len(v)} | {int((st > 0.99).sum())} | {st.mean():.3f} | {np.mean([x['best'] for x in v]):.3f} | "
                  f"{np.mean([x['final'] for x in v]):.3f} | {' '.join(f'{x:.2f}' for x in st)} |")
        # combination needing no truth: run l1 and ac2f, keep the stopped model with the lower data residual
        if "l1" in table[regime] and "ac2f" in table[regime]:
            a = {x["seed"]: x for x in table[regime]["l1"]}
            b = {x["seed"]: x for x in table[regime]["ac2f"]}
            common = sorted(set(a) & set(b))
            pick = np.array([(a[k] if a[k]["resid"] <= b[k]["resid"] else b[k])["stopped"] for k in common])
            print(f"| **l1 + adaptive, lower residual** | {len(common)} | {int((pick > 0.99).sum())} | {pick.mean():.3f} | – | – | "
                  f"{' '.join(f'{x:.2f}' for x in pick)} |")


if __name__ == "__main__":
    main(sys.argv[1:])
