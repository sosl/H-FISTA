"""
Synthetic wavefields and dynamic spectra for testing H-FISTA.

The Zenodo dataset used in the paper is not always available (e.g. offline
sandboxes), so this module generates data with the same structure:

* the wavefield h lives in (Doppler, delay) space, in the *unshifted* FFT layout
  used throughout the code base: axis 0 is Doppler (conjugate to time), axis 1 is
  delay (conjugate to frequency);
* power lies on/near a scintillation arc, delay ~= eta * doppler**2, at positive
  delays only, plus a strong unscattered component at the origin;
* within a band of half-width ``arc_width`` delay pixels around the arc each
  pixel is populated independently with probability ``density`` (the "local
  density" of the paper);
* the dynamic spectrum is |ifft2(h)|**2 (scipy normalisation), optionally with
  additive white Gaussian noise.

This is an approximation of the paper's simulations, not a reproduction of the
exact files on Zenodo.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.fft import ifftn


@dataclass
class Simulation:
    data: np.ndarray
    wavefield: np.ndarray
    noise_free: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def support(self) -> np.ndarray:
        return np.transpose(np.nonzero(self.wavefield))


def simulate(
    N: int = 256,
    M: int = 256,
    density: float = 0.125,
    arc_width: int = 2,
    max_doppler_frac: float = 0.35,
    max_delay_frac: float = 0.4,
    envelope_frac: float = 0.5,
    origin_power: float = 0.5,
    noise: float = 0.0,
    seed: Optional[int] = 0,
) -> Simulation:
    """Simulate a thin-screen-like wavefield and its dynamic spectrum.

    Args:
        N, M: data shape (time, frequency) == wavefield shape (doppler, delay).
        density: probability that a pixel inside the arc band is non-zero.
        arc_width: half-width of the arc band, in delay pixels.
        max_doppler_frac: arc extends to +-max_doppler_frac * N/2 Doppler pixels.
        max_delay_frac: arc reaches max_delay_frac * M/2 delay pixels at its ends.
        envelope_frac: Gaussian amplitude envelope width, as a fraction of the arc extent.
        origin_power: fraction of the total power in the origin component.
        noise: standard deviation of additive noise, relative to the mean of the data.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    k_max = int(max_doppler_frac * N / 2)
    tau_max = max_delay_frac * M / 2
    eta = tau_max / k_max**2

    h = np.zeros((N, M), dtype=complex)
    for k in range(-k_max, k_max + 1):
        centre = eta * k * k
        lo = max(int(np.floor(centre - arc_width)), 1)
        hi = min(int(np.ceil(centre + arc_width)), M // 2 - 1)
        for j in range(lo, hi + 1):
            if rng.random() < density:
                env = np.exp(-0.5 * (k / (envelope_frac * k_max)) ** 2)
                h[k % N, j] = env * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)

    scattered_power = np.sum(np.abs(h) ** 2)
    if scattered_power == 0:
        raise ValueError("No scattered components generated; increase density")
    h[0, 0] = np.sqrt(origin_power / (1 - origin_power) * scattered_power)

    # normalise so that the mean of the dynamic spectrum is 1 (scipy ifft carries 1/(N M))
    noise_free = np.abs(ifftn(h)) ** 2
    h *= 1.0 / np.sqrt(np.mean(noise_free))
    noise_free = np.abs(ifftn(h)) ** 2

    data = noise_free.copy()
    if noise > 0:
        data = data + rng.standard_normal(data.shape) * noise * np.mean(noise_free)

    params = dict(
        N=N,
        M=M,
        density=density,
        arc_width=arc_width,
        max_doppler_frac=max_doppler_frac,
        max_delay_frac=max_delay_frac,
        envelope_frac=envelope_frac,
        origin_power=origin_power,
        noise=noise,
        seed=seed,
    )
    return Simulation(data=data, wavefield=h, noise_free=noise_free, params=params)


def region_mask(N: int, M: int, seed: Optional[int] = 0, n_blobs: int = 3, arc: bool = True) -> np.ndarray:
    """Random mask of separate areas at positive delays, in the unshifted FFT layout.

    Modelled on the paper's simulations (four areas: a crescent-shaped arc through the
    origin region and three compact blobs at larger delays). Sizes scale with N, M.
    """
    rng = np.random.default_rng(seed)
    dop = np.fft.fftfreq(N) * N  # signed Doppler index
    dly = np.fft.fftfreq(M) * M  # signed delay index
    D, T = np.meshgrid(dop, dly, indexing="ij")
    mask = np.zeros((N, M), dtype=bool)
    if arc:
        k_max = rng.uniform(0.35, 0.6) * N / 2
        k0 = rng.uniform(-0.15, 0.15) * N / 2
        tau_end = rng.uniform(0.04, 0.1) * M / 2
        width = max(1.0, rng.uniform(0.01, 0.02) * M / 2)
        curve = tau_end * ((D - k0) / k_max) ** 2
        mask |= (np.abs(D - k0) <= k_max) & (np.abs(T - curve) <= width)
    for _ in range(n_blobs):
        cx = rng.uniform(-0.5, 0.5) * N / 2
        cy = rng.uniform(0.25, 0.65) * M / 2
        rx = rng.uniform(0.03, 0.07) * N / 2
        ry = rng.uniform(0.06, 0.12) * M / 2
        mask |= ((D - cx) / rx) ** 2 + ((T - cy) / ry) ** 2 <= 1.0
    mask &= T >= 0
    return mask


def simulate_regions(
    N: int = 128,
    M: int = 128,
    density: float = 0.125,
    mask: Optional[np.ndarray] = None,
    n_blobs: int = 3,
    decades: float = 4.0,
    origin_amplitude: float = 1.0,
    noise: float = 0.0,
    wavefield_noise: float = 0.0,
    seed: Optional[int] = 0,
) -> Simulation:
    """Paper-style simulation: components drawn at random, with probability `density`,
    inside a mask of separate positive-delay areas, with random modulus and phase.

    Args:
        mask: region mask (unshifted layout); default `region_mask(N, M, seed, n_blobs)`.
        decades: moduli are log-uniform over this many decades below 1 (the paper's
            noise-free 12.5% wavefield spans ~4 decades); 0 gives uniform moduli in (0, 1).
        origin_amplitude: modulus of the component at the origin (the strongest
            components are ~1); 0 for none. The paper's simulation has one ~as strong as
            the largest scattered component.
        noise: additive white noise on the dynamic spectrum (relative to its mean).
        wavefield_noise: complex Gaussian noise added to every wavefield pixel, relative
            to the rms of the non-zero components (the paper's noisy case adds noise to the
            wavefield).
    """
    rng = np.random.default_rng(None if seed is None else seed + 7919)
    if mask is None:
        mask = region_mask(N, M, seed, n_blobs)
    pick = mask & (rng.random((N, M)) < density)
    h = np.zeros((N, M), dtype=complex)
    n = int(pick.sum())
    if n == 0:
        raise ValueError("no components drawn; increase density")
    modulus = 10.0 ** rng.uniform(-decades, 0.0, n) if decades > 0 else rng.uniform(0.0, 1.0, n)
    h[pick] = modulus * np.exp(2j * np.pi * rng.random(n))
    if origin_amplitude > 0:
        h[0, 0] = origin_amplitude
        pick[0, 0] = True
        n = int(pick.sum())
    clean = h.copy()
    if wavefield_noise > 0:
        rms = np.sqrt(np.mean(np.abs(h[pick]) ** 2))
        h = h + wavefield_noise * rms * (rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))) / np.sqrt(2)
    scale = 1.0 / np.sqrt(np.mean(np.abs(ifftn(h)) ** 2))
    h *= scale
    clean *= scale
    noise_free = np.abs(ifftn(h)) ** 2
    data = noise_free.copy()
    if noise > 0:
        data = data + rng.standard_normal(data.shape) * noise * np.mean(noise_free)
    params = dict(N=N, M=M, density=density, n_blobs=n_blobs, decades=decades, origin_amplitude=origin_amplitude, noise=noise,
                  wavefield_noise=wavefield_noise, seed=seed, n_components=n, mask_pixels=int(mask.sum()))
    # `wavefield` is the noise-free truth; `noise_free` is the spectrum of the (possibly noisy) wavefield
    return Simulation(data=data, wavefield=clean, noise_free=noise_free, params=params)


def twin(h: np.ndarray) -> np.ndarray:
    """Conjugate-reflected wavefield h*(-k): it produces the same dynamic spectrum as h."""
    return np.conj(np.roll(h[::-1, ::-1], 1, axis=(0, 1)))


def align(estimate: np.ndarray, truth: np.ndarray) -> dict:
    """Best match of truth to estimate over the ambiguities of |ifft2(h)|^2: global phase,
    translation (any (Doppler, delay) shift) and the twin image.

    Returns dict(similarity, shift, twin) where similarity = max |<est, T truth>| / (|est||truth|).
    """
    den = np.linalg.norm(estimate) * np.linalg.norm(truth)
    if den == 0:
        return dict(similarity=0.0, shift=(0, 0), twin=False)
    best = None
    E = np.fft.fft2(estimate)
    for is_twin, t in ((False, truth), (True, twin(truth))):
        # c[s] = sum_k est[k] conj(t[k - s])
        c = np.abs(np.fft.ifft2(E * np.conj(np.fft.fft2(t))))
        idx = np.unravel_index(np.argmax(c), c.shape)
        val = c[idx] / den
        if best is None or val > best["similarity"]:
            best = dict(similarity=float(val), shift=(int(idx[0]), int(idx[1])), twin=is_twin)
    return best


def aligned_truth(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    a = align(estimate, truth)
    t = twin(truth) if a["twin"] else truth
    return np.roll(t, a["shift"], axis=(0, 1))


def wavefield_similarity(estimate: np.ndarray, truth: np.ndarray, invariant: bool = True) -> float:
    """1 for a perfect match. With invariant=True (default) the match is up to a global phase,
    a translation and the twin image (see `align`); otherwise only up to a global phase."""
    if invariant:
        return align(estimate, truth)["similarity"]
    num = np.abs(np.vdot(estimate, truth))
    den = np.linalg.norm(estimate) * np.linalg.norm(truth)
    return float(num / den) if den > 0 else 0.0


def support_scores(estimate: np.ndarray, truth: np.ndarray, invariant: bool = True) -> dict:
    if invariant:
        truth = aligned_truth(estimate, truth)
    est = estimate != 0
    tru = truth != 0
    tp = np.count_nonzero(est & tru)
    precision = tp / max(np.count_nonzero(est), 1)
    recall = tp / max(np.count_nonzero(tru), 1)
    return dict(precision=precision, recall=recall, n_est=int(est.sum()), n_true=int(tru.sum()))


def relative_residual(data: np.ndarray, model_wavefield: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """||mask (|ifft2 h|^2 - data)|| / ||mask data||"""
    if mask is None:
        mask = np.ones_like(data)
    pred = np.abs(ifftn(model_wavefield)) ** 2
    return float(np.linalg.norm((pred - data) * mask) / np.linalg.norm(data * mask))
