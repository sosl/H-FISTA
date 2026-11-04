"""
A twin-aware refiner: a CNN trained to map H-FISTA's densified model to the true wavefield
(pairs from prior/pairs.py). Unlike the plug-and-play denoiser, which learns to remove
Gaussian noise, this learns to remove the artefacts H-FISTA actually leaves: twin mixtures
and over-concentrated power.

Targets are aligned to the model (global phase, integer shift, truth or twin: whichever
matches best), since the data cannot fix these. Input and target are fftshifted, the origin
removed, and normalised by the input's rms (as prior/denoiser.py). Dilated convolutions give
a ~100-pixel receptive field.

Evaluation on the thin-screen test cases (seeds < 100): similarity of the refined model, and
of the refined model after the paper's densification restores data consistency.

    python prior/refiner.py train --pairs prior/results/pairs128_*.npz --out prior/results/refiner128.pt
    python prior/refiner.py eval --model prior/results/refiner128.pt --out prior/results/refiner128_eval.jsonl
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from prior.denoiser import to_input, from_output  # noqa: E402


class Refiner(torch.nn.Module):
    def __init__(self, ch=48, dil=(1, 2, 4, 8, 16, 8, 4, 2, 1)):
        super().__init__()
        layers = [torch.nn.Conv2d(2, ch, 3, padding=1, padding_mode="circular"), torch.nn.ReLU()]
        for d in dil:
            layers += [torch.nn.Conv2d(ch, ch, 3, padding=d, dilation=d, padding_mode="circular"), torch.nn.ReLU()]
        layers += [torch.nn.Conv2d(ch, 2, 3, padding=1, padding_mode="circular")]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        xc = x / (1 + torch.sqrt((x ** 2).sum(1, keepdim=True)))
        return x + self.net(xc)


def align(target, est):
    """target (or its twin), shifted and phase-rotated to best match est"""
    from simulate import twin

    E = np.fft.fft2(est)
    best = None
    for t in (target, twin(target)):
        c = np.fft.ifft2(E * np.conj(np.fft.fft2(t)))
        i = np.unravel_index(np.argmax(np.abs(c)), c.shape)
        if best is None or np.abs(c[i]) > best[0]:
            best = (np.abs(c[i]), np.roll(t, i, axis=(0, 1)) * np.exp(1j * np.angle(c[i])))
    return best[1]


def prepare(x, y):
    xi, scale, h00 = to_input(x)
    ya = align(y, x)
    ya = ya * (np.vdot(ya, x) / abs(np.vdot(ya, x)))  # keep the phase convention of x
    ya[0, 0] = 0
    yi = np.fft.fftshift(ya) / scale
    return xi, np.stack([yi.real, yi.imag]).astype(np.float32)


def train(a):
    torch.set_num_threads(a.threads)
    xs, ys = [], []
    for f in sorted(sum([glob.glob(p) for p in a.pairs], [])):
        z = np.load(f)
        for x, y in zip(z["x"], z["y"]):
            xi, yi = prepare(x.astype(complex), y.astype(complex))
            xs.append(xi); ys.append(yi)
    X, Y = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
    n_val = min(max(50, len(X) // 10), len(X) // 2)
    Xtr, Ytr, Xva, Yva = X[n_val:], Y[n_val:], X[:n_val], Y[:n_val]
    print(f"{len(Xtr)} train, {n_val} val", flush=True)
    model = Refiner(ch=a.ch)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    gen = torch.Generator().manual_seed(0)

    def sim(o, y):  # normalised |<o, y>| per sample (aligned, origin excluded)
        num = (o * y).sum(dim=(1, 2, 3))  # real part of <o, y>
        return num / (o.flatten(1).norm(dim=1) * y.flatten(1).norm(dim=1) + 1e-12)

    log = []
    for ep in range(a.epochs):
        t = time.time()
        model.train()
        perm = torch.randperm(len(Xtr), generator=gen)
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            x, y = Xtr[perm[i:i + a.batch]], Ytr[perm[i:i + a.batch]]
            loss = (1 - sim(model(x), y)).mean()  # scale-free: maximise similarity to the aligned truth
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(x)
        sched.step()
        model.eval()
        with torch.no_grad():
            s_in, s_out = sim(Xva, Yva).mean(), sim(model(Xva), Yva).mean()
        rec = dict(epoch=ep, train=tot / len(perm), val_sim_in=float(s_in), val_sim_out=float(s_out), t=time.time() - t)
        log.append(rec); print(rec, flush=True)
        torch.save(dict(arch=dict(ch=a.ch), state=model.state_dict(), log=log), a.out)


def evaluate(a):
    from prior.bench_screen import REGIMES, run_l1
    from prior.screen import simulate_screen, twin_split
    from simulate import wavefield_similarity, relative_residual
    from densify import get_dense_solution

    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = Refiner(**ck["arch"]); model.load_state_dict(ck["state"]); model.eval()
    settings = [(4.0, 15.0), (1.0, 15.0), (25.0, 15.0), (4.0, 30.0)]
    for ar, rf in settings:
        for regime in a.regimes:
            for seed in a.seeds:
                sim = simulate_screen(n_t=128, n_f=128, sigma_phi=REGIMES[regime], ar=ar, r_f=rf, bandwidth=0.2, seed=seed, noise=0.01)
                D, T = sim.data, sim.wavefield
                h, _ = run_l1(D)
                d = get_dense_solution(h, D, np.ones_like(D), iterations=1000)
                xi, scale, h00 = to_input(d)
                with torch.no_grad():
                    out = model(torch.from_numpy(xi)[None])[0].numpy()
                r = from_output(out, scale, h00)
                rd = get_dense_solution(r, D, np.ones_like(D), iterations=a.refit)
                rec = dict(ar=ar, r_f=rf, regime=regime, seed=seed, hfista=float(wavefield_similarity(d, T)),
                           refined=float(wavefield_similarity(r, T)), refined_refit=float(wavefield_similarity(rd, T)),
                           resid_refined=relative_residual(D, r), resid_refit=relative_residual(D, rd),
                           split_hfista=twin_split(d, T), split_refit=twin_split(rd, T))
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(ar, rf, regime, seed, f"H-FISTA {rec['hfista']:.3f} refined {rec['refined']:.3f} (resid {rec['resid_refined']:.2f}) "
                      f"refit {rec['refined_refit']:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--pairs", nargs="+", default=["prior/results/pairs128_*.npz"])
    p.add_argument("--model")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ch", type=int, default=48)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--regimes", nargs="+", default=["S3", "S10", "S30"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--refit", type=int, default=200)
    a = p.parse_args()
    train(a) if a.cmd == "train" else evaluate(a)


if __name__ == "__main__":
    main()
