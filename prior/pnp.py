"""
Plug-and-play phase retrieval with the learned wavefield prior (prior/denoiser.py).

FISTA on H-FISTA's demerit with the proximal step replaced by the learned denoiser:
    x = A ⊙ Denoise(y - ∇f(y)/L, σ_k)
(A zeroes forbidden delays), with B3 backtracking as in H-FISTA. The denoising strength
σ_k (relative to the rms of the wavefield) decreases geometrically over stages, the role
H-FISTA's λ schedule plays: strong prior first, data later. Optionally finished with the
paper's densification (unregularised FISTA on the allowed half plane).

Compared on the thin-screen benchmark (prior/bench_screen.py regimes) against H-FISTA
and greedy pursuit, each followed by densification.

    python prior/pnp.py --model prior/results/denoiser128.pt --regimes S3 S10 S30 --seeds 0 1 2 3 --N 128 --out prior/results/pnp128.jsonl
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
from fista import construct_lambda_matrix  # noqa: E402
from lib import Residual  # noqa: E402
from simulate import wavefield_similarity, relative_residual  # noqa: E402
from densify import get_dense_solution  # noqa: E402
from dense.variants import fista_generic  # noqa: E402
from prior.denoiser import Prior  # noqa: E402
from prior.screen import simulate_screen, twin_split  # noqa: E402
from prior.bench_screen import REGIMES  # noqa: E402


def pnp(data, prior, mask=None, sigma0=1.0, sigma1=0.01, stages=20, iters=20, h_init=None, delay_for_inf=-4, callback=None):
    N, M = data.shape
    mask = np.ones_like(data) if mask is None else mask
    allowed = np.isfinite(construct_lambda_matrix((N, M), 1.0, delay_for_inf))
    if h_init is None:
        h = np.zeros((N, M), complex)
        h[0, 0] = np.sqrt(np.mean(data * mask)) * N * M
    else:
        h = h_init.astype(complex).copy()
    sigmas = np.geomspace(sigma0, sigma1, stages)
    for k, s in enumerate(sigmas):
        r = Residual(data, h, None, mask)
        h, _, _ = fista_generic(r, h, lambda z, L: prior(z, s) * allowed, iters, r.get_Lipschitz_constant_grad(), fix_phase=None)
        if callback is not None:
            callback(h, k, s)
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--noise", type=float, default=0.01)
    p.add_argument("--sigma0", type=float, default=1.0)
    p.add_argument("--sigma1", type=float, default=0.01)
    p.add_argument("--stages", type=int, default=20)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    prior = Prior(a.model)
    for regime in a.regimes:
        for seed in a.seeds:
            sim = simulate_screen(n_t=a.N, n_f=a.N, sigma_phi=REGIMES[regime], r_f=15.0, bandwidth=0.2, seed=seed, noise=a.noise)
            D, T = sim.data, sim.wavefield
            t0 = time.time()
            trace = []
            h = pnp(D, prior, sigma0=a.sigma0, sigma1=a.sigma1, stages=a.stages, iters=a.iters,
                    callback=lambda h, k, s: trace.append(dict(stage=k, sigma=float(s), sim=float(wavefield_similarity(h, T)),
                                                               resid=relative_residual(D, h))))
            t1 = time.time()
            d = get_dense_solution(h, D, np.ones_like(D), iterations=1000)
            rec = dict(regime=regime, seed=seed, method="pnp", N=a.N, sigma0=a.sigma0, sigma1=a.sigma1, stages=a.stages,
                       iters=a.iters, sim=float(wavefield_similarity(h, T)), dense_sim=float(wavefield_similarity(d, T)),
                       resid=relative_residual(D, h), dense_resid=relative_residual(D, d), wall_solver=t1 - t0, trace=trace,
                       split=twin_split(h, T), dense_split=twin_split(d, T))
            with open(a.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{regime} seed {seed} pnp: sim {rec['sim']:.3f} dense {rec['dense_sim']:.3f} resid {rec['resid']:.3f}/"
                  f"{rec['dense_resid']:.1e} true/twin {rec['dense_split']['true']:.2f}/{rec['dense_split']['twin']:.2f} "
                  f"{rec['wall_solver']:.0f}s | trace " +
                  " ".join(f"{x['sim']:.2f}" for x in trace[::4]), flush=True)


if __name__ == "__main__":
    main()
