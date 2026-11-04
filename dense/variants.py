"""
H-FISTA with pluggable regularisers, for wavefields that are locally dense.

The hierarchy is the published one (regularised FISTA, debiasing on a fixed support,
hard thresholding, approval, geometric λ schedule). What changes is the *unit* that
the penalty, hard threshold and approval act on, and the penalty itself:

* "l1"      : per-pixel l1 (published H-FISTA; reproduces pipeline.run_hfista exactly)
* "elastic" : per-pixel l1 + l2 (elastic net, paper §5.4); prox = soft threshold
              followed by shrinkage 1/(1 + κ), κ = beta · λ_s/λ_init
* "group"   : k x k blocks in (Doppler, delay) with penalty λ k Σ_b ||h_b||_2 (group
              lasso). Sparsity is enforced between blocks, not within them: an approved
              block approves *all* its pixels, which is the "estimate at lower resolution,
              then populate all corresponding higher-resolution pixels" idea.
* "mcp"     : per-pixel minimax concave penalty (non-convex); prox = firm threshold:
              zero below λ/L, unshrunk above γ λ/L, linear in between. Less bias on strong
              components than l1, so the selection is less driven by shrinkage.
* reweighting (reweight_passes > 0, any per-pixel method): after the regularised FISTA
              of a λ step, re-solve R times with Λ_ij scaled by ε/(|h_ij| + ε),
              ε = c λ/L (majorise-minimise for the log penalty; Candès, Wakin & Boyd 2008).
              Pixels already found are penalised less, zero pixels keep λ.
* loss="amplitude": fit |ifft2 h| to sqrt(D) instead of |ifft2 h|^2 to D (amplitude flow).
* coarse-to-fine: a block size per stage (e.g. 4 → 2 → 1) moves from the coarse to
              the fine description as λ decreases; approval carries over between levels.
"""
import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.fft import fftn, fftshift, ifftshift, ifftn
from scipy.linalg import norm

import fista as F
from fista import complex_phase_fix, construct_lambda_matrix
from lib import Residual
from lambda_loop import get_initial_lambda
from auxiliary import generate_narrowband_RFI_mask, update_RFI_mask

log = logging.getLogger("H-FISTA.dense")


# ------------------------------------------------------------------------------------
# block helpers (blocks are aligned on the fftshifted grid, so they are contiguous in
# signed Doppler/delay)
# ------------------------------------------------------------------------------------
def _blocks(n, k):
    return -(-n // k)  # ceil(n / k): a dimension that is not a multiple of k gets a partial last block


def block_reduce(a: np.ndarray, k: int) -> np.ndarray:
    """sum of a over k x k blocks of the fftshifted array -> (ceil(N/k), ceil(M/k))"""
    s = fftshift(a)
    N, M = s.shape
    nb, mb = _blocks(N, k), _blocks(M, k)
    if nb * k != N or mb * k != M:
        s = np.pad(s, ((0, nb * k - N), (0, mb * k - M)))
    return s.reshape(nb, k, mb, k).sum(axis=(1, 3))


def block_expand(b: np.ndarray, k: int, shape=None) -> np.ndarray:
    """inverse layout of block_reduce: blocks -> unshifted array of `shape`, constant on blocks"""
    full = np.kron(b, np.ones((k, k), dtype=b.dtype))
    if shape is not None:
        full = full[: shape[0], : shape[1]]
    return ifftshift(full)


def block_ids(shape, k):
    N, M = shape
    ids = np.arange(_blocks(N, k) * _blocks(M, k)).reshape(_blocks(N, k), _blocks(M, k))
    return block_expand(ids, k, shape)


# ------------------------------------------------------------------------------------
# amplitude loss: f(h) = Σ mask (|ifft2 h| - sqrt(D))^2
# ------------------------------------------------------------------------------------
class AmplitudeResidual(Residual):
    """Residual with the amplitude loss. Same interface as lib.Residual (derivative is the
    Wirtinger derivative ∂f/∂h*, so the B3 backtracking test is unchanged)."""

    def __init__(self, data, wavefield, l1_penalty, RFI_mask, **kw):
        self.sqrt_data = np.sqrt(np.maximum(data, 0.0))
        super().__init__(data, wavefield, l1_penalty, RFI_mask, **kw)

    def set_residual(self):
        self.abs_H = np.sqrt(self.prediction)
        self.residual = self.abs_H - self.sqrt_data

    def set_derivative(self):
        prod = np.prod(self.data.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ph = np.where(self.abs_H > 0, self.H / self.abs_H, 0.0)
        self._derivative = 1.0 / prod * fftn(self.residual * self.RFI_mask * ph, workers=self.workers)

    def set_Lipschitz_constant_grad(self):
        # f is ~ ||ifft2(h) - b||^2 near the solution: L = 2/(NM) in the Wirtinger convention
        self.Lipschitz_constant = 2.0 / np.prod(self.data.shape)


def make_residual(loss):
    if loss == "intensity":
        return Residual
    if loss == "amplitude":
        return AmplitudeResidual
    raise ValueError(loss)


def firm_threshold(z, Lam, L, gamma):
    """complex firm threshold (prox of the MCP with parameter gamma > 1); Λ = inf -> 0, Λ = 0 -> z"""
    a = np.abs(z)
    t = Lam / L
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mid = gamma * (a - t) / ((gamma - 1.0) * a)
        scale = np.where(a <= t, 0.0, np.where(a > gamma * t, 1.0, mid))
    scale = np.where(np.isinf(Lam), 0.0, np.where(Lam == 0, 1.0, scale))
    out = z * scale
    if np.count_nonzero(out) == 0:
        raise ValueError("firm_threshold: λ/L too large, no elements left")
    return out


# ------------------------------------------------------------------------------------
# regularisers: prox(z, Lam, forbid, L) and unit handling
# ------------------------------------------------------------------------------------
@dataclass
class Reg:
    kind: str = "l1"
    k: int = 1  # block size (group)
    beta: float = 0.0  # elastic-net strength relative to λ
    lam_ratio: float = 1.0  # λ_s / λ_init, set per stage
    gamma: float = 3.0  # MCP concavity


    def prox(self, z: np.ndarray, Lam: np.ndarray, L: float) -> np.ndarray:
        if self.kind == "mcp":
            return firm_threshold(z, Lam, L, self.gamma)
        if self.kind in ("l1", "elastic") or (self.kind == "group" and self.k == 1):
            x = F.complex_prox_l1(z, Lam, L)
            if self.kind == "elastic" and self.beta > 0:
                kappa = self.beta * self.lam_ratio
                shrink = np.where((Lam > 0) & np.isfinite(Lam), 1.0 / (1.0 + kappa), 1.0)
                x = x * shrink
            return x
        # group lasso: forbidden pixels (Λ = inf) are removed first; blocks with any approved
        # pixel (Λ = 0) are unpenalised; others are shrunk as a block with threshold λ k / L
        k = self.k
        z = np.where(np.isinf(Lam), 0.0, z)
        bn = np.sqrt(block_reduce(np.abs(z) ** 2, k))
        lam_b = block_reduce(np.where(np.isfinite(Lam), Lam, 0.0), k) / (k * k)  # mean finite λ in block
        approved_b = block_reduce((Lam == 0).astype(float), k) > 0
        thr_b = np.where(approved_b, 0.0, lam_b * k / L)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale_b = np.where(bn > thr_b, 1.0 - thr_b / bn, 0.0)
        x = z * block_expand(scale_b, k, z.shape)
        if np.count_nonzero(x) == 0:
            raise ValueError("group prox: λ/L too large, no elements left")
        return x

    def units_of(self, x: np.ndarray, allowed: np.ndarray) -> np.ndarray:
        """support used for debiasing/approval: non-zero pixels, or all allowed pixels of non-zero blocks"""
        if self.kind != "group" or self.k == 1:
            return np.transpose(np.nonzero(x))
        nzb = block_reduce((x != 0).astype(float), self.k) > 0
        pix = block_expand(nzb, self.k, x.shape) & allowed
        return np.transpose(np.nonzero(pix))

    def hard_threshold(self, x, support, threshold):
        """zero units below threshold; returns x, remaining support, deleted pixel coords"""
        if self.kind != "group" or self.k == 1:
            vals = np.abs(x[support[:, 0], support[:, 1]])
            dele = support[vals < threshold]
            x[dele[:, 0], dele[:, 1]] = 0
            keep = vals >= threshold
            return x, support[keep], dele
        k = self.k
        bn = np.sqrt(block_reduce(np.abs(x) ** 2, k))
        ids = block_ids(x.shape, k)
        sup_ids = ids[support[:, 0], support[:, 1]]
        bn_flat = bn.ravel()
        drop = bn_flat[sup_ids] < threshold * k
        dele = support[drop]
        x[dele[:, 0], dele[:, 1]] = 0
        return x, support[~drop], dele


# ------------------------------------------------------------------------------------
# FISTA with backtracking and a pluggable prox (same arithmetic as fista.fista)
# ------------------------------------------------------------------------------------
def fista_generic(resid: Residual, x0, prox: Callable, niter: int, s: float, eta: float = 1.1, fix_phase=(0, 0)):
    def P(z, L):
        x = prox(z, 1.0 / (1.0 / L))  # fista passes alpha = 1/L, then uses 1/alpha
        if fix_phase is not None:
            x = complex_phase_fix(x, [list(fix_phase)], 0.0)
        return x

    x = y = x0
    t = 1.0
    L = s
    demerits = []
    for _ in range(niter):
        f_y, g = resid.get_func_val_and_derivative(y)
        i = 0
        L = 1.0 / (1.0 / L)  # fista keeps alpha = 1/L between iterations and backtrack_B3 starts from 1/alpha
        Li = L
        xt = P(y - 1.0 / Li * g, Li)
        ft = resid.get_func_val(xt)
        while ft > f_y + 2.0 * np.real(np.vdot(g, (xt - y))) + Li / 2.0 * np.power(norm((xt - y)), 2):
            i += 1
            Li = L * np.power(eta, i)
            xt = P(y - 1.0 / Li * g, Li)
            ft = resid.get_func_val(xt)
        L = Li
        t_new = (1 + np.sqrt(1 + 4 * np.power(t, 2))) / 2.0
        y = xt + (t - 1.0) / t_new * (xt - x)
        x, t = xt, t_new
        demerits.append(ft)
    return x, np.array(demerits), L


# ------------------------------------------------------------------------------------
# hierarchical driver
# ------------------------------------------------------------------------------------
def run_variant(
    data: np.ndarray,
    method: str = "l1",
    block_sizes: Optional[Sequence[int]] = None,
    beta: float = 0.0,
    mask: Optional[np.ndarray] = None,
    desired_components: int = 60,
    lambda_scale: float = 1.15,
    FISTA_niter: int = 80,
    delay_for_inf: int = -4,
    threshold_factor: float = 1.0,
    max_steps: int = 60,
    clean_RFI: bool = False,
    stop: Optional[Callable] = None,
    step_callback: Optional[Callable] = None,
    eta: float = 1.1,
    levels: Optional[Sequence[int]] = None,
    plateau_tol: float = 0.05,
    min_resid_drop: float = 0.3,
    stop_on_final_plateau: bool = False,
    gamma: float = 3.0,
    reweight_passes: int = 0,
    reweight_c: float = 1.0,
    loss: str = "intensity",
):
    """method: "l1", "elastic" (with beta) or "group" (with block_sizes: one per stage, the last
    value repeating; e.g. [4]*10 + [2]*10 + [1]).

    levels (group only), e.g. [4, 2, 1]: adaptive coarse-to-fine. Stay at a block size until the
    number of non-zero pixels plateaus (grows by <= plateau_tol between λ steps, after having
    grown at this level), then move to the next, finer block size. Leaving the coarsest level
    also requires the relative data residual to have dropped by at least min_resid_drop
    during it (the coarse structure has been found, not just a slow start). With stop_on_final_plateau,
    a plateau at the last level ends the run (chosen = that step).
    method "mcp" (with gamma), reweight_passes/reweight_c and loss: see the module docstring.
    Returns io, chosen step."""
    Res = make_residual(loss)
    N, M = data.shape
    if mask is None:
        mask = np.ones_like(data)
    h = np.zeros((N, M), dtype=complex)
    h[0, 0] = np.sqrt(np.mean(data * mask)) * N * M + 0.0j
    resid = Res(data, h, None, mask)
    lam_init = get_initial_lambda(desired_components, delay_for_inf, resid) * lambda_scale
    allowed = np.isfinite(construct_lambda_matrix((N, M), 1.0, delay_for_inf))

    io = {"models": {0: h}, "lambdas": {0: lam_init}, "masks": {0: mask}, "Ls": {}, "demerits": {}, "block": {}}
    approved = np.array([[0, 0]])
    chosen = -1
    level, n_level_start, n_prev, r_level_start = 0, None, None, None
    io["plateaus"] = []
    for step in range(1, max_steps + 1):
        lam = io["lambdas"][step - 1] / lambda_scale
        k = 1
        if method == "group" and levels:
            k = levels[level]
        elif method == "group":
            bs = list(block_sizes or [2])
            k = bs[min(step - 1, len(bs) - 1)]
        reg = Reg(kind=method, k=k, beta=beta, lam_ratio=lam / lam_init, gamma=gamma)
        s_guess = Res(data, io["models"][step - 1], None, io["masks"][step - 1]).get_Lipschitz_constant_grad()

        Lam = construct_lambda_matrix((N, M), lam, delay_for_inf, zero_penalty_coords=approved)
        r = Res(data, io["models"][step - 1], None, io["masks"][step - 1])
        fit, dem, L = fista_generic(r, io["models"][step - 1], lambda z, LL: reg.prox(z, Lam, LL), FISTA_niter, s_guess, eta)
        rw_dems = []
        for _ in range(reweight_passes):
            eps = reweight_c * lam / L
            Lam_rw = Lam * (eps / (np.abs(fit) + eps))  # inf stays inf, 0 stays 0
            r = Res(data, fit, None, io["masks"][step - 1])
            fit, d_rw, L = fista_generic(r, fit, lambda z, LL: reg.prox(z, Lam_rw, LL), FISTA_niter, s_guess, eta)
            rw_dems.append(d_rw)
        if rw_dems:
            dem = np.concatenate([dem] + rw_dems)
        mask_s = io["masks"][step - 1]
        if clean_RFI:
            mask_s = update_RFI_mask(mask_s, generate_narrowband_RFI_mask(r.residual, 0, 5.0, 21))
        support = reg.units_of(fit, allowed)
        dems = [dem]
        prev = None
        while True:
            Lam_fix = construct_lambda_matrix((N, M), 0.0, fix_support=support)
            r = Res(data, fit, None, mask_s)
            fit, dem, L = fista_generic(r, fit, lambda z, LL: F.complex_prox_l1(z, Lam_fix, LL), FISTA_niter, s_guess, eta)
            dems.append(dem)
            if clean_RFI:
                mask_s = update_RFI_mask(mask_s, generate_narrowband_RFI_mask(r.residual, 0, 5.0, 21))
            nz = np.transpose(np.nonzero(fit))
            fit, _, dele = reg.hard_threshold(fit, nz, threshold_factor * lam / L)
            if len(dele):
                dset = set(map(tuple, dele.tolist()))
                support = np.array([c for c in support.tolist() if tuple(c) not in dset]).reshape(-1, 2)
            if len(dele) == 0:
                break
            key = sorted(map(tuple, dele.tolist()))
            if key == prev:
                break
            prev = key
        approved = support
        io["models"][step] = fit
        io["lambdas"][step] = lam
        io["masks"][step] = mask_s
        io["Ls"][step] = L
        io["demerits"][step] = np.concatenate(dems)
        io["block"][step] = k
        if step_callback is not None:
            step_callback(io, step)
        if method == "group" and levels:
            n = int(np.count_nonzero(fit))
            pred = np.abs(ifftn(fit)) ** 2
            rel_res = np.linalg.norm((pred - data) * mask_s) / np.linalg.norm(data * mask_s)
            if n_level_start is None:
                n_level_start, r_level_start = n, rel_res
            elif (n_prev is not None and n > n_level_start and n <= n_prev * (1 + plateau_tol)
                  and (level > 0 or rel_res <= (1 - min_resid_drop) * r_level_start)):
                io["plateaus"].append((step, k))
                if level < len(levels) - 1:
                    level += 1
                    n_level_start = None
                elif stop_on_final_plateau:
                    chosen = step
                    break
            n_prev = n
        if stop is not None and step > 1 and stop(io, step):
            chosen = step - 1
            break
    return io, chosen
