"""
Benchmark H-FISTA variants on noise-free simulations where the true wavefield is known.

    python dense/benchmark.py --methods l1 group2 c2f --regimes U12 U25 L50 L25 --seeds 0 1 --out results.jsonl

Each run records, for every λ step, the ambiguity-invariant similarity to the truth
(1 = perfect up to global phase, translation and twin), the relative data residual
and the number of components. Noise-free data: a correct recovery reaches
similarity ~1 and residual ~0.
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
from simulate import simulate_regions, wavefield_similarity, relative_residual, support_scores  # noqa: E402
from dense.variants import run_variant  # noqa: E402

REGIMES = {
    # name: (density, decades)
    "U12": (0.125, 0.0),
    "U25": (0.25, 0.0),
    "L25": (0.25, 4.0),
    "L50": (0.5, 4.0),
}

METHODS = {
    "l1": dict(method="l1"),
    "elastic0.1": dict(method="elastic", beta=0.1),
    "elastic0.5": dict(method="elastic", beta=0.5),
    "group2": dict(method="group", block_sizes=[2]),
    "group4": dict(method="group", block_sizes=[4]),
    "c2f": dict(method="group", block_sizes=[4] * 8 + [2] * 8 + [1]),
    "c2f2": dict(method="group", block_sizes=[2] * 15 + [1]),
    "ac2f": dict(method="group", levels=[4, 2, 1], stop_on_final_plateau=True),
    "ac2f8": dict(method="group", levels=[8, 4, 2, 1], stop_on_final_plateau=True),
    # non-convex / reweighted penalties and the amplitude loss
    "mcp3": dict(method="mcp", gamma=3.0),
    "rw1": dict(method="l1", reweight_passes=2, reweight_c=1.0),
    "rw4": dict(method="l1", reweight_passes=2, reweight_c=4.0),
    "amp": dict(method="l1", loss="amplitude"),
    "amp_rw1": dict(method="l1", loss="amplitude", reweight_passes=2, reweight_c=1.0),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--n0", type=int, default=15)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    done = set()
    if os.path.exists(a.out):  # resumable: skip runs already recorded
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["regime"], r["seed"], r["method"]))
            except ValueError:
                pass
    for regime in a.regimes:
        density, decades = REGIMES[regime]
        for seed in a.seeds:
            sim = simulate_regions(N=a.N, M=a.N, density=density, decades=decades, seed=seed)
            for name in a.methods:
                if (regime, seed, name) in done:
                    continue
                per_step = []

                def cb(io, s):
                    m = io["models"][s]
                    per_step.append(dict(step=s, sim=wavefield_similarity(m, sim.wavefield),
                                         resid=relative_residual(sim.data, m), n=int(np.count_nonzero(m)),
                                         block=int(io["block"][s])))

                t0 = time.time()
                try:
                    io, _ = run_variant(sim.data, desired_components=a.n0, max_steps=a.steps, step_callback=cb, **METHODS[name])
                    err = None
                except Exception as e:  # record failures (e.g. prox emptied the model)
                    err = f"{type(e).__name__}: {e}"
                wall = time.time() - t0
                best = max(per_step, key=lambda r: r["sim"]) if per_step else None
                last = per_step[-1] if per_step else None
                sc = support_scores(io["models"][last["step"]], sim.wavefield) if per_step and err is None else None
                rec = dict(regime=regime, seed=seed, method=name, n_true=sim.params["n_components"], wall=wall, error=err,
                           last=last, best=best, support=sc, steps=per_step,
                           plateaus=io.get("plateaus", []) if err is None else [])
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"{regime} seed {seed} {name:>11}: last sim {last['sim'] if last else float('nan'):.4f} resid {last['resid'] if last else float('nan'):.2e} "
                      f"n {last['n'] if last else -1}/{sim.params['n_components']} | best sim {best['sim'] if best else float('nan'):.4f} @ {best['step'] if best else -1} "
                      f"| {wall:.0f}s {err or ''}", flush=True)


if __name__ == "__main__":
    main()
