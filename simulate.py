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


def wavefield_similarity(estimate: np.ndarray, truth: np.ndarray) -> float:
    """|<est, truth>| / (|est| |truth|): 1 for a perfect match up to a global phase."""
    num = np.abs(np.vdot(estimate, truth))
    den = np.linalg.norm(estimate) * np.linalg.norm(truth)
    return float(num / den) if den > 0 else 0.0


def support_scores(estimate: np.ndarray, truth: np.ndarray) -> dict:
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
