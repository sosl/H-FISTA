"""
Greedy sparse phase retrieval for pulsar dynamic spectra: a fast CLEAN.

CLEAN (and the CLEAN-like method of Walker & Stinebring 2005, Walker et al. 2008) adds
the component that best explains the residual, one at a time. The sparse phase-retrieval
literature has greedy methods of the same family that are reported to beat l1 at higher
densities (GESPAR, Shechtman, Beck & Eldar 2014; CoSaMP / subspace pursuit adapted to
phase retrieval). Their speed comes from the same trick as Clark / Cotton-Schwab CLEAN:
pick *many* components per cycle (minor cycle) and only then refit everything against
the data with FFTs (major cycle).

One round of `greedy_pursuit`:

  1. g = ∂f/∂h* at the current model (f = Σ mask (|ifft2 h|^2 - D)^2, as in H-FISTA)
  2. add the 2k allowed pixels with the largest |g| outside the support S
     (k = max(k_min, growth·|S|)); they start at the gradient step -g/L
  3. refit all values on S ∪ new (FISTA restricted to the support, i.e. H-FISTA's debias)
  4. prune: keep the |S| + k largest |h| (the origin pixel is always kept) and refit

Step 4 lets the method undo wrong picks, which l1 with H-FISTA's approval cannot do: an
approved pixel stays approved. No λ, no threshold schedule.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

import fista as F
from fista import construct_lambda_matrix
from lib import Residual
from dense.variants import fista_generic

log = logging.getLogger("H-FISTA.greedy")


class FFTCounter:
    """Count FFTs done by lib.Residual (one inverse per state change, one forward per gradient)."""

    n = 0

    @classmethod
    def install(cls):
        if getattr(Residual, "_counted", False):
            return
        set_state, set_der = Residual._set_state, Residual.set_derivative

        def _set_state(self, w):
            cls.n += 1
            return set_state(self, w)

        def set_derivative(self):
            cls.n += 1
            return set_der(self)

        Residual._set_state, Residual.set_derivative, Residual._counted = _set_state, set_derivative, True


def refit(data, h, support_mask, mask, niter, eta=1.1):
    """FISTA on the fixed support (no penalty): H-FISTA's debiasing step."""
    Lam = np.where(support_mask, 0.0, np.inf)
    r = Residual(data, h, None, mask)
    s = r.get_Lipschitz_constant_grad()
    x, dem, L = fista_generic(r, h, lambda z, LL: F.complex_prox_l1(z, Lam, LL), niter, s, eta)
    return x, L


@dataclass
class Trace:
    rounds: List[dict] = field(default_factory=list)


def greedy_pursuit(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    delay_for_inf: int = -4,
    growth: float = 0.15,
    k_min: int = 4,
    oversample: float = 2.0,
    refit_iters: int = 30,
    max_fraction: float = 0.03,
    max_rounds: int = 200,
    prune: bool = True,
    round_callback: Optional[Callable] = None,
    jitter: float = 0.0,
    seed: int = 0,
    h_init: Optional[np.ndarray] = None,
):
    """Returns (models per round, trace). Stops when the support would exceed max_fraction of
    the pixels (the paper's sparsity criterion: the last model below it is the answer).
    jitter > 0 multiplies the selection scores by exp(jitter * N(0, 1)): a randomised
    selection for restarts (GESPAR also restarts from random supports)."""
    rng = np.random.default_rng(seed)
    N, M = data.shape
    mask = np.ones_like(data) if mask is None else mask
    allowed = np.isfinite(construct_lambda_matrix((N, M), 1.0, delay_for_inf))
    if h_init is None:
        h = np.zeros((N, M), dtype=complex)
        h[0, 0] = np.sqrt(np.mean(data * mask)) * N * M
    else:  # e.g. an early H-FISTA model: start from its support
        h = h_init.copy()
    S = h != 0
    S[0, 0] = True
    models = {0: h.copy()}
    cap = max_fraction * N * M
    for rnd in range(1, max_rounds + 1):
        n = int(S.sum())
        k = max(k_min, int(np.ceil(growth * n)))
        if n + k > cap:
            break
        r = Residual(data, h, None, mask)
        g = r.get_derivative()
        L = r.get_Lipschitz_constant_grad()
        score = np.where(allowed & ~S, np.abs(g), -1.0)
        if jitter > 0:
            score = np.where(score > 0, score * np.exp(jitter * rng.standard_normal(score.shape)), score)
        n_add = int(np.ceil(oversample * k)) if prune else k
        cand = np.argpartition(score.ravel(), -n_add)[-n_add:]
        cand = cand[score.ravel()[cand] > 0]
        h = h.copy()
        h.ravel()[cand] = -g.ravel()[cand] / L
        S2 = S.copy()
        S2.ravel()[cand] = True
        h, L = refit(data, h, S2, mask, refit_iters)
        if prune and S2.sum() > n + k:
            a = np.abs(h).ravel().copy()
            a[0] = np.inf  # keep the origin
            keep = np.argpartition(a, -(n + k))[-(n + k):]
            S = np.zeros_like(S)
            S.ravel()[keep] = True
            h = np.where(S, h, 0)
            h, L = refit(data, h, S, mask, refit_iters)
        else:
            S = S2
        S &= h != 0
        models[rnd] = h
        if round_callback is not None:
            round_callback(models, rnd)
    return models


def restarted_pursuit(data, restarts=8, jitter=0.3, mask=None, **kw):
    """Run greedy_pursuit `restarts` times (the first without jitter) and, at every round,
    keep the model with the lowest data residual over the restarts (no truth needed).
    Returns models per round and the index of the restart chosen at each round."""
    from simulate import relative_residual

    runs = [greedy_pursuit(data, mask=mask, jitter=0.0 if i == 0 else jitter, seed=i, **kw) for i in range(restarts)]
    rounds = min(max(r) for r in runs)
    models, chosen = {}, {}
    for k in range(rounds + 1):
        res = [relative_residual(data, r[k], mask) for r in runs]
        j = int(np.argmin(res))
        models[k], chosen[k] = runs[j][k], j
    return models, chosen
