"""
Compare greedy pursuit with H-FISTA (l1) on the 256² benchmark, seed by seed.

"stopped" for greedy (no truth needed): the first round whose relative residual is below
1e-6 (the noise-free data are fitted); otherwise the last round (the support reached 3% of
the pixels, the paper's sparsity criterion). A residual-plateau rule does not work: the
residual barely moves in the first rounds. Noisy data would need a noise-level rule. "stopped" for l1: the paper's 3% sparsity criterion, as in
dense/summarize.py. "solved" = similarity > 0.99.

    python greedy/summarize.py greedy/results/bench256*.jsonl -- dense/results/bench256_*.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, ".")
from dense.summarize import stopped_step, REGIME  # noqa: E402


def greedy_stopped(steps, solved=1e-6):
    for s in steps:
        if s["resid"] < solved:
            return s
    return steps[-1]


def load(files):
    runs = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r.get("error") or not r["steps"]:
                continue
            runs[(r["regime"], r["seed"], r["method"])] = r
    return runs


def main(argv):
    i = argv.index("--") if "--" in argv else len(argv)
    g, d = load(argv[:i]), load(argv[i + 1:])
    table = defaultdict(dict)
    for (reg, seed, m), r in g.items():
        if m == "l1":
            continue
        st = greedy_stopped(r["steps"])
        table[(reg, m)][seed] = (st["sim"], st.get("fft", 0), r["wall"], st["step"])
    for (reg, seed, m), r in d.items():
        if m == "l1":
            st = stopped_step(r, 256 * 256)
            table[(reg, "l1")][seed] = (st["sim"], None, r["wall"], st["step"])
    l1fft = {k[:2]: r["steps"][-1]["fft"] for k, r in g.items() if k[2] == "l1"}
    methods = ["l1"] + sorted({m for (_, m) in table if m != "l1"})
    for reg in ["U12", "U25", "L25", "L50"]:
        print(f"\n### {REGIME[reg]}\n")
        print("| method | solved | mean similarity, stopped | per-seed | mean wall [s] | mean FFTs |")
        print("|---|---|---|---|---|---|")
        for m in methods:
            v = table.get((reg, m))
            if not v:
                continue
            seeds = sorted(v)
            sims = np.array([v[s][0] for s in seeds])
            ffts = [v[s][1] for s in seeds if v[s][1]] or [l1fft[(reg, s)] for s in seeds if (reg, s) in l1fft]
            print(f"| {m} | {int((sims > 0.99).sum())}/{len(seeds)} | {sims.mean():.3f} | {' '.join(f'{x:.2f}' for x in sims)} | "
                  f"{np.mean([v[s][2] for s in seeds]):.0f} | {np.mean(ffts):.0f} |" if ffts else
                  f"| {m} | {int((sims > 0.99).sum())}/{len(seeds)} | {sims.mean():.3f} | {' '.join(f'{x:.2f}' for x in sims)} | "
                  f"{np.mean([v[s][2] for s in seeds]):.0f} | – |")


if __name__ == "__main__":
    main(sys.argv[1:])
