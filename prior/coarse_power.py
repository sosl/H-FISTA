"""
Does H-FISTA get the coarse (block-averaged) power distribution of a locally dense wavefield
right, even when the fine structure is wrong? On thin-screen simulations (128²), compare the
k×k block power of the sparse and densified models with that of the truth (and its twin):
normalised correlation of the block-power maps (non-negative), best over integer shifts.
Also: how concentrated the model's power is within the occupied blocks of the truth
(fraction of block power in the brightest pixel, model vs truth) — the "all power into a
single component" effect.

    python prior/coarse_power.py --out prior/results/coarse_power128.jsonl
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prior.bench_screen import REGIMES, run_l1, run_group, run_greedy  # noqa: E402
from prior.screen import simulate_screen  # noqa: E402
from densify import get_dense_solution  # noqa: E402
from simulate import twin  # noqa: E402


def block_power(h, k):
    p = np.abs(h) ** 2
    p[0, 0] = 0  # origin excluded
    N, M = p.shape
    return p.reshape(N // k, k, M // k, k).sum(axis=(1, 3))


def corr_best_shift(a, b):
    c = np.real(np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))))
    return float(c.max() / (np.linalg.norm(a) * np.linalg.norm(b)))


def peak_fraction(h, truth, k):
    """power-weighted mean, over the truth's brightest blocks holding 90% of its (non-origin)
    power, of (max pixel power / block power)"""
    pt = block_power(truth, k)
    order = np.argsort(pt.ravel())[::-1]
    sel = order[: np.searchsorted(np.cumsum(pt.ravel()[order]), 0.9 * pt.sum()) + 1]
    p = np.abs(h) ** 2
    p[0, 0] = 0
    N, M = p.shape
    blk = p.reshape(N // k, k, M // k, k).transpose(0, 2, 1, 3).reshape(-1, k * k)
    tot = blk.sum(1)[sel]
    mx = blk.max(1)[sel]
    w = tot
    return float(np.sum(np.where(tot > 0, mx / np.maximum(tot, 1e-300), 0) * w) / max(w.sum(), 1e-300))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--methods", nargs="+", default=["l1", "group2", "greedy"])
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for regime in a.regimes:
        for seed in a.seeds:
            sim = simulate_screen(n_t=a.N, n_f=a.N, sigma_phi=REGIMES[regime], r_f=15.0, bandwidth=0.2, seed=seed, noise=0.01)
            D, T = sim.data, sim.wavefield
            for m in a.methods:
                h, _ = run_l1(D) if m == "l1" else run_group(D, m) if m != "greedy" else run_greedy(D)
                d = get_dense_solution(h, D, np.ones_like(D), iterations=1000)
                rec = dict(regime=regime, seed=seed, method=m)
                for k in (1, 2, 4, 8):
                    bt, btw = block_power(T, k), block_power(twin(T), k)
                    for name, x in (("sparse", h), ("dense", d)):
                        bx = block_power(x, k)
                        rec[f"{name}_k{k}"] = corr_best_shift(bx, bt)
                        rec[f"{name}_k{k}_twin"] = corr_best_shift(bx, btw)
                for k in (2, 4):
                    rec[f"peak_truth_k{k}"] = peak_fraction(T, T, k)
                    rec[f"peak_sparse_k{k}"] = peak_fraction(h, T, k)
                    rec[f"peak_dense_k{k}"] = peak_fraction(d, T, k)
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(regime, seed, m, {k: round(v, 2) for k, v in rec.items() if k not in ("regime", "seed", "method")}, flush=True)


if __name__ == "__main__":
    main()
