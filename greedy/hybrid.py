"""
Greedy pursuit as a cheap first pass, then H-FISTA from its model.

Greedy runs until its support reaches `target` components (an early stop: a small,
high-confidence support, to limit the risk of letting twin-image components in, which is
what H-FISTA's slow λ schedule guards against). H-FISTA then starts from that model with
its pixels approved (dense.variants.run_variant(h_init=...)), choosing its initial λ so
that N0 new pixels pass the threshold, and runs its usual hierarchy and stopping rules.

Benchmark on the 256² simulations (same regimes/seeds as dense/benchmark.py; "stopped" =
the paper's 3% sparsity criterion, applied by dense/summarize.py):

    python greedy/hybrid.py --targets 60 250 1000 --seeds 0 1 2 3 4 5 --out greedy/results/hybrid256.jsonl
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
from greedy.pursuit import greedy_pursuit, FFTCounter  # noqa: E402
from dense.benchmark import REGIMES  # noqa: E402
from dense.variants import run_variant  # noqa: E402


def greedy_first_pass(data, target, mask=None, **kw):
    """greedy rounds until the support has >= target components; returns that model and its round"""
    models, masks, chosen = greedy_pursuit(data, mask=mask, stop=lambda m, ms, r: np.count_nonzero(m[r]) >= target, **kw)
    last = max(models)
    return models[last], last


def hybrid(data, target, mask=None, desired_components=15, max_steps=60, step_callback=None, clean_RFI=False, stop=None):
    h0, rounds = greedy_first_pass(data, target, mask=mask, clean_RFI=clean_RFI)
    io, chosen = run_variant(data, mask=mask, h_init=h0, desired_components=desired_components, max_steps=max_steps,
                             step_callback=step_callback, clean_RFI=clean_RFI, stop=stop)
    return io, chosen, h0, rounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", type=int, default=[60, 250, 1000])
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
            for target in a.targets:
                name = f"hyb{target}"
                if (regime, seed, name) in done:
                    continue
                per_step = []
                FFTCounter.n = 0
                t0 = time.time()

                def cb(io, s):
                    m = io["models"][s]
                    per_step.append(dict(step=s, sim=wavefield_similarity(m, sim.wavefield), resid=relative_residual(sim.data, m),
                                         n=int(np.count_nonzero(m)), block=1, fft=FFTCounter.n))

                io, _, h0, rounds = hybrid(sim.data, target, step_callback=cb)
                first = dict(rounds=rounds, n=int(np.count_nonzero(h0)), sim=wavefield_similarity(h0, sim.wavefield),
                             resid=relative_residual(sim.data, h0))
                rec = dict(regime=regime, seed=seed, method=name, wall=time.time() - t0, error=None, first_pass=first,
                           steps=per_step, last=per_step[-1], best=max(per_step, key=lambda r: r["sim"]), plateaus=[])
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"{regime} seed {seed} {name}: greedy {first['n']} comp. sim {first['sim']:.3f} -> "
                      f"last {rec['last']['sim']:.4f} n {rec['last']['n']} | best {rec['best']['sim']:.4f} @ {rec['best']['step']} "
                      f"| {per_step[-1]['fft']} FFT {rec['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
