"""
Use the coarse solution to extend H-FISTA's support, then refit without the l1 penalty.

Start from the H-FISTA (l1) model and refit, with 1000 FISTA iterations, on a support S
(pixels outside S, or at forbidden delays, are set to 0), either unpenalised or with a ridge
penalty μ‖h‖² (μ = c·L₀; spreads power instead of concentrating it). Supports:

  l1        H-FISTA's own support (no extension)
  dense90   l1 support ∪ the k×k blocks holding 90% of the (non-origin) power of the
            densified H-FISTA model, i.e. the coarse power estimate
  group     l1 support ∪ the k×k blocks touched by the 2×2 group-lasso model
  oracle    l1 support ∪ the k×k blocks holding 90% of the true power (upper bound: is the
            idea worth pursuing if the coarse support were known?)
  full      the whole allowed half plane (= the paper's densification)

    python prior/coarse_support.py --out prior/results/coarse_support128.jsonl
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prior.bench_screen import REGIMES, run_l1, run_group  # noqa: E402
from prior.screen import simulate_screen, twin_split  # noqa: E402
from prior.coarse_power import block_power  # noqa: E402
from fista import construct_lambda_matrix  # noqa: E402
from lib import Residual  # noqa: E402
from simulate import wavefield_similarity, relative_residual  # noqa: E402
from densify import get_dense_solution  # noqa: E402
from dense.variants import fista_generic  # noqa: E402


def blocks_to_pixels(bmask, k, shape):
    return np.kron(bmask, np.ones((k, k), bool))[: shape[0], : shape[1]]


def top_blocks(h, k, frac=0.9):
    b = block_power(h, k)
    order = np.argsort(b.ravel())[::-1]
    n = np.searchsorted(np.cumsum(b.ravel()[order]), frac * b.sum()) + 1
    m = np.zeros(b.size, bool)
    m[order[:n]] = True
    return m.reshape(b.shape)


def refit(D, h0, support, iters=1000, ridge=0.0):
    r = Residual(D, h0, None, np.ones_like(D))
    L0 = r.get_Lipschitz_constant_grad()
    mu = ridge * L0

    def prox(z, L):
        return z * support / (1.0 + 2.0 * mu / L)

    h, _, _ = fista_generic(r, h0 * support, prox, iters, L0, fix_phase=None)
    return h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--ridges", nargs="+", type=float, default=[0.0, 0.01, 0.1])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    k = a.k
    for regime in a.regimes:
        for seed in a.seeds:
            sim = simulate_screen(n_t=a.N, n_f=a.N, sigma_phi=REGIMES[regime], r_f=15.0, bandwidth=0.2, seed=seed, noise=0.01)
            D, T = sim.data, sim.wavefield
            allowed = np.isfinite(construct_lambda_matrix(D.shape, 1.0, -4))
            h, _ = run_l1(D)
            d_full = get_dense_solution(h, D, np.ones_like(D), iterations=1000)
            g, _ = run_group(D, "group2")
            base = h != 0
            supports = {
                "l1": base,
                "dense90": base | blocks_to_pixels(top_blocks(d_full, k), k, D.shape),
                "group": base | blocks_to_pixels(block_power(g, k) > 0, k, D.shape),
                "oracle": base | blocks_to_pixels(top_blocks(T, k), k, D.shape),
                "full": allowed,
            }
            for name, S in supports.items():
                S = S & allowed
                S[0, 0] = True
                for c in a.ridges:
                    if name == "full" and c == 0.0:
                        x = d_full
                    else:
                        x = refit(D, h, S, ridge=c)
                    rec = dict(regime=regime, seed=seed, support=name, ridge=c, k=k, frac=float(S.sum() / allowed.sum()),
                               sim=float(wavefield_similarity(x, T)), resid=relative_residual(D, x), split=twin_split(x, T),
                               l1_sim=float(wavefield_similarity(h, T)))
                    with open(a.out, "a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(f"{regime} {seed} {name:>7} ridge {c:<5} frac {rec['frac']:.3f} sim {rec['sim']:.3f} resid {rec['resid']:.1e} "
                          f"true/twin {rec['split']['true']:.2f}/{rec['split']['twin']:.2f}", flush=True)


if __name__ == "__main__":
    main()
