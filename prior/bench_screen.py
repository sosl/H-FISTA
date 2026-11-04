"""
Benchmark on thin-screen simulations (prior/screen.py): the wavefield is a physical,
generally dense field, not a set of on-grid point components.

Regimes (r_F = 15 px, 20% bandwidth, anisotropy 4, 256x256, 1% noise): screen rms phase
  S3  (σ_φ = 3 rad):  90% of the power in ~0.1% of pixels, origin ~30% of the power
  S10 (σ_φ = 10 rad): ~3%, origin ~2%
  S30 (σ_φ = 30 rad): ~12%, origin ~0.2%  (dense)

Methods, each followed by the paper's densification (1000 unregularised FISTA iterations
on the allowed half plane):
  l1:     H-FISTA with the paper's stopping criteria (N0 = 15)
  greedy: greedy pursuit, noise-level (χ²) stop, else the paper's criteria, else the 3% cap

Similarity to the true wavefield is the ambiguity-invariant one (simulate.wavefield_similarity).

    python prior/bench_screen.py --regimes S3 S10 S30 --seeds 0 1 2 3 --out prior/results/screen256.jsonl
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
from simulate import wavefield_similarity, relative_residual  # noqa: E402
from stopping import check_stopping  # noqa: E402
from densify import get_dense_solution  # noqa: E402
from dense.variants import run_variant  # noqa: E402
from prior.screen import simulate_screen, twin_split  # noqa: E402

REGIMES = {"S3": 3.0, "S10": 10.0, "S30": 30.0}


def run_l1(D, n0=15):
    io, chosen = run_variant(D, desired_components=n0, max_steps=60, stop=check_stopping)
    s = chosen if chosen > 0 else max(io["models"])
    return io["models"][s], dict(step=s)


def run_greedy(D):
    from prior.greedy_local import greedy_pursuit, noise_variance  # copied from claude/greedy-pursuit

    sigma2 = noise_variance(D)
    state = {}

    def stop(models, masks, rnd):
        res = np.abs(np.fft.ifft2(models[rnd])) ** 2 - D
        chi2 = np.sum(res ** 2) / (sigma2 * D.size)
        if chi2 <= 1.0:
            state["rule"] = "chi2"
            return True
        if rnd > 1 and check_stopping({"models": models}, rnd):
            state["rule"] = "paper"
            return True
        return False

    models, masks, chosen = greedy_pursuit(D, stop=stop)
    r = chosen if chosen >= 0 else max(models)
    if state.get("rule") == "chi2":
        r = max(models)  # the χ² rule keeps the round that reached the noise level
    return models[r], dict(round=r, rule=state.get("rule", "cap"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--methods", nargs="+", default=["l1", "greedy"])
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["regime"], r["seed"], r["method"]))
    for regime in a.regimes:
        for seed in a.seeds:
            sim = simulate_screen(n_t=a.N, n_f=a.N, sigma_phi=REGIMES[regime], r_f=15.0, bandwidth=0.2, seed=seed, noise=a.noise)
            D, T = sim.data, sim.wavefield
            for m in a.methods:
                if (regime, seed, m) in done:
                    continue
                t0 = time.time()
                h, info = run_l1(D) if m == "l1" else run_greedy(D)
                t1 = time.time()
                d = get_dense_solution(h, D, np.ones_like(D), iterations=1000)
                rec = dict(regime=regime, seed=seed, method=m, params=sim.params, info=info, n=int(np.count_nonzero(h)),
                           sparse_sim=float(wavefield_similarity(h, T)), dense_sim=float(wavefield_similarity(d, T)),
                           sparse_resid=relative_residual(D, h), dense_resid=relative_residual(D, d),
                           wall_solver=t1 - t0, wall=time.time() - t0,
                           sparse_split=twin_split(h, T), dense_split=twin_split(d, T))
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"{regime} seed {seed} {m:>6}: n {rec['n']} sparse {rec['sparse_sim']:.3f} dense {rec['dense_sim']:.3f} "
                      f"resid {rec['sparse_resid']:.3f}/{rec['dense_resid']:.1e} true/twin {rec['dense_split']['true']:.2f}/"
                      f"{rec['dense_split']['twin']:.2f} {info} {rec['wall_solver']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
