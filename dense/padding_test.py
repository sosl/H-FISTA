"""
Does padding the time axis with masked rows change the retrieved wavefield?

H-FISTA's forward model is a cyclic 2-D DFT, so on the original grid the observation is
implicitly periodic in time: the last sub-integration wraps round to the first. Real
observations are not periodic. Here the truth is simulated over a longer span (256 rows)
and only the first W rows are observed (noise added). H-FISTA is run on the W-row grid
and on grids padded with masked rows; each model's field ifft2(h) over the observed rows
is compared with the true field (|<E, E_true>| / norms, 1 = same up to a global phase).
Uses the paper's stopping criteria and densification.

    python dense/padding_test.py --seeds 0 1 2 3 4 5 6 7 --out dense/results/padding_test.jsonl
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
from simulate import simulate_regions  # noqa: E402
from dense.variants import run_variant  # noqa: E402
from dense.compare_padding import field_similarity  # noqa: E402
from stopping import check_stopping  # noqa: E402
from densify import get_dense_solution  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--W", type=int, default=121, help="observed rows (prime, like J1939's 487)")
    p.add_argument("--pads", nargs="+", type=int, default=[0, 128, 242], help="0 = original grid")
    p.add_argument("--density", type=float, default=0.125)
    p.add_argument("--noise", type=float, default=0.02, help="noise std relative to the data std")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rng = np.random.default_rng(12345)
    for seed in a.seeds:
        sim = simulate_regions(N=256, M=256, density=a.density, decades=4.0, seed=seed)
        E_true = np.fft.ifft2(sim.wavefield)[: a.W]
        D = np.abs(E_true) ** 2
        D = D + rng.normal(0, a.noise * D.std(), D.shape)
        m = np.ones_like(D)
        rec = dict(seed=seed, W=a.W, density=a.density, noise=a.noise, runs={})
        for pad in a.pads:
            rows = pad or a.W
            Dp, mp = np.pad(D, ((0, rows - a.W), (0, 0))), np.pad(m, ((0, rows - a.W), (0, 0)))
            t0 = time.time()
            io, c = run_variant(Dp, mask=mp, desired_components=15, max_steps=60, stop=check_stopping)
            wall = time.time() - t0
            s = c if c > 0 else max(io["models"])
            h = io["models"][s]
            d = get_dense_solution(h, Dp, io["masks"][s], iterations=1000)
            fs = lambda hh: field_similarity(np.fft.ifft2(hh)[: a.W], E_true, m > 0)  # noqa: E731
            rec["runs"][rows] = dict(step=s, n=int(np.count_nonzero(h)), sparse=fs(h), dense=fs(d), wall=wall)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(seed, {k: (v["step"], v["n"], round(v["sparse"], 3), round(v["dense"], 3), round(v["wall"])) for k, v in rec["runs"].items()},
              flush=True)


if __name__ == "__main__":
    main()
