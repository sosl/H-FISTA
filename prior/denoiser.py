"""
A learned prior for wavefields: a small convolutional denoiser trained on thin-screen
simulations (prior/screen.py), for plug-and-play phase retrieval (prior/pnp.py).

Input: a noisy wavefield on the fftshifted (Doppler, delay) grid, normalised by its rms,
as 2 channels (real, imaginary) plus a constant channel with the noise level σ (relative to
the rms). Output: the estimated noise (residual learning, as DnCNN). Pixels at forbidden
delays are zeroed after denoising by the caller.

Training data: screens with σ_φ drawn log-uniformly in [2, 40] rad, r_F = 15, 20% bandwidth,
anisotropy 4, at 128²; seeds 10_000+ (the test benchmarks use seeds < 100). Noise σ drawn
log-uniformly in [5e-3, 1] × rms (of the non-origin pixels) per sample. Loss: mean over
samples of log(output MSE / noise power), so every noise level counts equally in dB.

    python prior/denoiser.py --n_train 1500 --epochs 20 --out prior/results/denoiser128.pt
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prior.screen import simulate_screen  # noqa: E402


class Denoiser(torch.nn.Module):
    def __init__(self, ch=32, depth=8):
        super().__init__()
        layers = [torch.nn.Conv2d(3, ch, 3, padding=1, padding_mode="circular"), torch.nn.ReLU()]
        for _ in range(depth - 2):
            layers += [torch.nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular"), torch.nn.ReLU()]
        layers += [torch.nn.Conv2d(ch, 2, 3, padding=1, padding_mode="circular")]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x, sigma):
        # the network sees a compressed input x / (1 + |x|) (values span ~4 decades; uncompressed, training
        # does not converge) and predicts the noise in units of sigma
        sv = sigma.view(-1, 1, 1, 1)
        s = sv.expand(-1, 1, x.shape[2], x.shape[3])
        xc = x / (1 + torch.sqrt((x ** 2).sum(1, keepdim=True)))
        return x - self.net(torch.cat([xc, s], 1)) * sv


def to_input(h):
    """complex (N, M) unshifted wavefield -> (2, N, M) float, shifted, normalised by the rms of the
    pixels other than the origin; the origin pixel is removed (it holds up to ~30% of the power and
    would swamp the network) and returned separately. Returns x, scale, origin value."""
    h = h.copy()
    h00 = h[0, 0]
    h[0, 0] = 0
    scale = np.sqrt(np.sum(np.abs(h) ** 2) / (h.size - 1)) + 1e-300
    hs = np.fft.fftshift(h) / scale
    return np.stack([hs.real, hs.imag]).astype(np.float32), scale, h00


def from_output(x, scale, h00):
    h = np.fft.ifftshift((x[0] + 1j * x[1]).astype(complex)) * scale
    h[0, 0] = h00
    return h


class Prior:
    """callable wrapper: denoise(h, sigma_rel) for numpy wavefields"""

    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model = Denoiser(**ck["arch"])
        self.model.load_state_dict(ck["state"])
        self.model.eval()

    def __call__(self, h, sigma_rel):
        """sigma_rel: noise level relative to the rms of the non-origin pixels; the origin passes unchanged"""
        x, scale, h00 = to_input(h)
        with torch.no_grad():
            y = self.model(torch.from_numpy(x)[None], torch.tensor([sigma_rel], dtype=torch.float32))[0].numpy()
        return from_output(y, scale, h00)


def make_data(n, N, seed0, rng):
    xs = []
    for i in range(n):
        s = float(np.exp(rng.uniform(np.log(2), np.log(40))))
        sim = simulate_screen(n_t=N, n_f=N, sigma_phi=s, r_f=15.0, bandwidth=0.2, seed=seed0 + i)
        x, _, _ = to_input(sim.wavefield)
        xs.append(x)
    return np.stack(xs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=1500)
    p.add_argument("--n_val", type=int, default=100)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)
    cache = a.out.replace(".pt", "_data.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        Xtr, Xva = z["tr"], z["va"]
        if z.get("version") is None:  # cache from the first version (origin kept, total-rms scaling): convert
            def conv(X):
                X = X.copy()
                X[:, :, X.shape[2] // 2, X.shape[3] // 2] = 0  # origin sits at the centre after fftshift
                rms = np.sqrt((X ** 2).sum(axis=(1, 2, 3), keepdims=True) / (X.shape[2] * X.shape[3] - 1))
                return (X / rms).astype(np.float32)
            Xtr, Xva = conv(Xtr), conv(Xva)
            np.savez(cache, tr=Xtr, va=Xva, version=2)
    else:
        t = time.time()
        Xtr = make_data(a.n_train, a.N, 10_000, rng)
        Xva = make_data(a.n_val, a.N, 50_000, rng)
        np.savez(cache, tr=Xtr, va=Xva, version=2)
        print(f"data {time.time() - t:.0f}s", flush=True)
    arch = dict(ch=32, depth=8)
    model = Denoiser(**arch)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    Xtr_t, Xva_t = torch.from_numpy(Xtr), torch.from_numpy(Xva)
    gen = torch.Generator().manual_seed(0)

    def noisy(x):
        sig = torch.exp(torch.empty(x.shape[0]).uniform_(np.log(5e-3), np.log(1.0), generator=gen))
        n = torch.randn(x.shape, generator=gen) * sig.view(-1, 1, 1, 1) / np.sqrt(2)  # complex noise, rms sig
        return x + n, sig

    log = []
    for ep in range(a.epochs):
        t = time.time()
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            x = Xtr_t[perm[i:i + a.batch]]
            xn, sig = noisy(x)
            # scale-free loss: mean over samples of log(output error / noise power), i.e. every sample counts
            # equally in dB (a 1/σ²-weighted MSE lets the smallest-σ samples, where identity is optimal, dominate)
            loss = torch.log(((model(xn, sig) - x) ** 2).mean(dim=(1, 2, 3)) / (sig ** 2 / 2 * 2) + 1e-12).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(x)
        sched.step()
        model.eval()
        with torch.no_grad():
            xn, sig = noisy(Xva_t)
            out = model(xn, sig)
            gain = (((xn - Xva_t) ** 2).mean(dim=(1, 2, 3)) / ((out - Xva_t) ** 2).mean(dim=(1, 2, 3))).log10().mul(10)
        bins = {f"gain_db_sigma<{b:g}": float(gain[(sig < b) & (sig >= b / 10)].median()) for b in (0.05, 0.3, 1.0)}
        rec = dict(epoch=ep, train=tot / len(perm), val_gain_db_median=float(gain.median()), **bins, t=time.time() - t)
        log.append(rec)
        print(rec, flush=True)
        torch.save(dict(arch=arch, state=model.state_dict(), log=log, N=a.N), a.out)
    json.dump(log, open(a.out.replace(".pt", "_log.json"), "w"))


if __name__ == "__main__":
    main()
