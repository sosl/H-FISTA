"""
Benchmark greedy pursuit on the same 256² noise-free simulations as dense/benchmark.py
(same regimes and seeds, so the l1 records in dense/results/bench256_*.jsonl are the
reference). Every round records similarity to the truth, relative residual, number of
components and the cumulative FFT count.

    python greedy/benchmark.py --methods greedy --seeds 0 1 2 3 4 5 --out greedy/results/bench256.jsonl
    python greedy/benchmark.py --methods l1 --seeds 0 1 --out greedy/results/l1_ffts.jsonl   # FFT counts for l1
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import logger  # noqa: E402,F401

logging.disable(logging.INFO)
import fista as F  # noqa: E402

F.PROX_IMPLEMENTATION = "fast"
from simulate import simulate_regions, wavefield_similarity, relative_residual  # noqa: E402
from greedy.pursuit import greedy_pursuit, restarted_pursuit, FFTCounter  # noqa: E402
from dense.benchmark import REGIMES  # noqa: E402
from dense.variants import run_variant  # noqa: E402

METHODS = {
    "greedy": dict(growth=0.15, prune=True),
    "greedy_noprune": dict(growth=0.15, prune=False),
    "greedy_slow": dict(growth=0.05, prune=True),
    "greedy_fast": dict(growth=0.4, prune=True),
    "greedy_r8": dict(growth=0.15, prune=True, restarts=8, jitter=0.3),
    "greedy_r8j1": dict(growth=0.15, prune=True, restarts=8, jitter=1.0),
    "l1": None,  # H-FISTA through dense.variants (bitwise identical to the published loop), for FFT counts
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["greedy"])
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    FFTCounter.install()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["regime"], r["seed"], r["method"]))
    for regime in a.regimes:
        density, decades = REGIMES[regime]
        for seed in a.seeds:
            sim = simulate_regions(N=a.N, M=a.N, density=density, decades=decades, seed=seed)
            for name in a.methods:
                if (regime, seed, name) in done:
                    continue
                trace = []
                FFTCounter.n = 0
                t0 = time.time()

                def rec(m):
                    return dict(sim=wavefield_similarity(m, sim.wavefield), resid=relative_residual(sim.data, m),
                                n=int(np.count_nonzero(m)), fft=FFTCounter.n, t=time.time() - t0)

                if name == "l1":
                    def cb(io, s):
                        trace.append(dict(step=s, **rec(io["models"][s])))

                    run_variant(sim.data, desired_components=15, max_steps=60, step_callback=cb)
                elif "restarts" in METHODS[name]:
                    models, chosen = restarted_pursuit(sim.data, **METHODS[name])
                    for r in range(1, max(models) + 1):
                        trace.append(dict(step=r, restart=chosen[r], **rec(models[r])))
                else:
                    def cb(models, r):
                        trace.append(dict(step=r, **rec(models[r])))

                    greedy_pursuit(sim.data, round_callback=cb, **METHODS[name])
                wall = time.time() - t0
                out = dict(regime=regime, seed=seed, method=name, wall=wall, steps=trace)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(out) + "\n")
                last = trace[-1]
                best = max(trace, key=lambda x: x["sim"])
                print(f"{regime} seed {seed} {name:>15}: last sim {last['sim']:.4f} resid {last['resid']:.2e} n {last['n']} "
                      f"| best {best['sim']:.4f} @ {best['step']} | {last['fft']} FFT {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
