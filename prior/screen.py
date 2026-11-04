"""
Thin phase-screen simulation of pulsar dynamic spectra (no fitted parameters of any real
pulsar; a generic generator for realistic test and training wavefields).

  * phase screen: Gaussian random field with power spectrum
        P(q) ∝ (q_x² + (q_y / ar)²)^(-β/2) exp(-(q / q_inner)²),   β = 11/3 (Kolmogorov),
    ar > 1 elongates the scattering along x (anisotropic screens give arcs with arclets)
    and rms phase sigma_phi at the reference frequency (weak scattering: ~1 rad, few
    images, sparse wavefield; strong: >> 1 rad, many images, dense wavefield);
  * frequency ν = f ν0: screen phase φ / f, Fresnel scale r_F² ∝ 1 / f;
  * observer field: E_f = ifft2(fft2(exp(i φ / f)) exp(-i q² r_F² / (2 f))), sampled along
    the line y = 0 as the screen moves past (x = t), over the first n_t of n_x pixels
    (not periodic in time, like a real observation);
  * D(t, ν) = |E(t, ν)|² and the true wavefield h = fft2(E) on the same grid as H-FISTA.

The sign convention is chosen so that the scattered power lies at positive delays, as
H-FISTA assumes (E and conj(E) give the same D: the twin).
"""
from dataclasses import dataclass

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq

from fista import construct_lambda_matrix


@dataclass
class ScreenSim:
    data: np.ndarray  # (n_t, n_f) dynamic spectrum
    field: np.ndarray  # E(t, ν)
    wavefield: np.ndarray  # h = fft2(E)
    params: dict


def phase_screen(nx, ny, sigma_phi, ar=1.0, beta=11 / 3, inner=4.0, rng=None):
    rng = np.random.default_rng(rng)
    qx = fftfreq(nx)[:, None]
    qy = fftfreq(ny)[None, :]
    q2 = qx ** 2 + (qy / ar) ** 2
    q2[0, 0] = np.inf
    P = q2 ** (-beta / 2) * np.exp(-(qx ** 2 + qy ** 2) * inner ** 2)
    noise = rng.standard_normal((nx, ny)) + 1j * rng.standard_normal((nx, ny))
    phi = np.real(ifft2(noise * np.sqrt(P)))
    phi -= phi.mean()
    return phi * sigma_phi / phi.std()


def simulate_screen(n_t=256, n_f=256, sigma_phi=3.0, ar=4.0, r_f=6.0, bandwidth=0.1, nx_factor=2, ny=128, inner=2.0,
                    seed=0, noise=0.0):
    """r_f: Fresnel scale in screen pixels (one pixel = the distance moved per time sample)."""
    rng = np.random.default_rng(seed)
    nx = nx_factor * n_t
    phi = phase_screen(nx, ny, sigma_phi, ar=ar, inner=inner, rng=rng)
    qx = 2 * np.pi * fftfreq(nx)[:, None]
    qy = 2 * np.pi * fftfreq(ny)[None, :]
    q2 = qx ** 2 + qy ** 2
    fs = 1 + bandwidth * (np.arange(n_f) / n_f - 0.5)
    E = np.empty((n_t, n_f), complex)
    for j, f in enumerate(fs):
        u = fft2(np.exp(1j * phi / f))
        e = ifft2(u * np.exp(-1j * q2 * r_f ** 2 / (2 * f)))
        E[:, j] = e[:n_t, 0]
    h = fft2(E)
    allowed = np.isfinite(construct_lambda_matrix((n_t, n_f), 1.0, -4))
    p = np.abs(h) ** 2
    if p[~allowed].sum() > p[allowed].sum():  # put the scattered power at positive delays
        E = np.conj(E)
        h = fft2(E)
    D = np.abs(E) ** 2
    if noise > 0:
        D = D + rng.normal(0, noise * D.std(), D.shape)
    p = np.abs(h) ** 2
    srt = np.sort(p.ravel())[::-1]
    frac90 = float(np.searchsorted(np.cumsum(srt), 0.9 * srt.sum()) / srt.size)
    params = dict(n_t=n_t, n_f=n_f, sigma_phi=sigma_phi, ar=ar, r_f=r_f, bandwidth=bandwidth, seed=seed, noise=noise,
                  origin_power=float(p[0, 0] / p.sum()), frac_pixels_90pct_power=frac90,
                  power_outside_allowed=float(p[~allowed].sum() / p.sum()))
    return ScreenSim(D, E, h, params)
