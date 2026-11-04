"""
Training pairs for a twin-aware refiner: (H-FISTA densified model, true wavefield) on
thin-screen simulations from the broad family (σ_φ log-uniform [2, 40] rad, anisotropy
[1.5, 12], r_F [6, 20], 1% noise), seeds 100_000 + i (disjoint from all test seeds).

    python prior/pairs.py --start 0 --n 500 --out prior/results/pairs128_0.npz
"""
import argparse, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prior.bench_screen import run_l1  # noqa: E402
from prior.screen import simulate_screen  # noqa: E402
from densify import get_dense_solution  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    X, Y, P = [], [], []
    t = time.time()
    for i in range(a.start, a.start + a.n):
        rng = np.random.default_rng(100_000 + i)
        s = float(np.exp(rng.uniform(np.log(2), np.log(40))))
        ar = float(np.exp(rng.uniform(np.log(1.5), np.log(12))))
        rf = float(np.exp(rng.uniform(np.log(6), np.log(20))))
        sim = simulate_screen(n_t=a.N, n_f=a.N, sigma_phi=s, ar=ar, r_f=rf, bandwidth=0.2, seed=100_000 + i, noise=0.01)
        h, _ = run_l1(sim.data)
        d = get_dense_solution(h, sim.data, np.ones_like(sim.data), iterations=1000)
        X.append(d.astype(np.complex64)); Y.append(sim.wavefield.astype(np.complex64)); P.append([s, ar, rf])
        if (i - a.start) % 25 == 24:
            np.savez(a.out, x=np.array(X), y=np.array(Y), p=np.array(P), data_seed0=100_000 + a.start)
            print(i - a.start + 1, f"{time.time() - t:.0f}s", flush=True)
    np.savez(a.out, x=np.array(X), y=np.array(Y), p=np.array(P), data_seed0=100_000 + a.start)


if __name__ == "__main__":
    main()
