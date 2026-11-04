"""Chromatic (NuT-style) rescaling test: share of pixels holding 90% of the wavefield power, with and without
rescaling time per channel by (nu/nu0)^P (P set below)."""
P = 1
import sys, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prior.screen import simulate_screen

def nut(E, fs, ref=1.0):
    """resample each channel in time: E'(t', nu) = E(t' * ref / f, nu) by Fourier interpolation (periodic approx)"""
    n_t = E.shape[0]
    out = np.zeros_like(E); mask = np.zeros(E.shape, bool)
    t = np.arange(n_t)
    k = np.fft.fftfreq(n_t)
    for j, f in enumerate(fs):
        src = t * (f / ref) ** P
        ok = src <= n_t - 1
        X = np.fft.fft(E[:, j])
        out[ok, j] = (np.exp(2j*np.pi*np.outer(src[ok], k)) @ X) / n_t
        mask[:, j] = ok
    return out, mask

def frac90(h):
    p = np.sort(np.abs(h.ravel())**2)[::-1]
    return np.searchsorted(np.cumsum(p), 0.9*p.sum())/p.size

N = 128
for bw in (0.2, 0.05):
    for s in (3, 10, 30):
        sim = simulate_screen(n_t=N, n_f=N, sigma_phi=s, r_f=15.0, bandwidth=bw, seed=0)
        fs = 1 + bw*(np.arange(N)/N - 0.5)
        for ref in (fs.min(), fs.max()):
            En, m = nut(sim.field, fs, ref)
            print(f"bw {bw} S{s} ref {ref:.2f}: frac90 plain {frac90(sim.wavefield):.4f}  NuT {frac90(np.fft.fft2(En*m)):.4f}  rows kept {m.all(1).sum()}")
