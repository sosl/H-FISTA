"""
Run a method on an observed dynamic spectrum (configured as in the paper's notebooks),
densify the result as in the paper (§2.5) and report diagnostics that need no truth:

  * relative residual of the sparse and dense models (masked),
  * fraction of the dense model's power at negative delays (beyond the allowed buffer):
    the twin image shows up there, a good model is noise-like (paper §5.1),
  * the model images (log10 |h|^2, fftshifted).

    python dense/real_data.py --dataset J1939 --method l1 --out dense/results/J1939_l1
    python dense/real_data.py --dataset J1939 --method ac2f --out dense/results/J1939_ac2f
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "review"))
import logger  # noqa: E402,F401

logging.disable(logging.INFO)
import fista as F  # noqa: E402

F.PROX_IMPLEMENTATION = "fast"
from regression import load  # noqa: E402
from stopping import check_stopping  # noqa: E402
from densify import get_dense_solution  # noqa: E402
from dense.variants import run_variant  # noqa: E402

METHODS = {
    "l1": dict(method="l1"),
    "ac2f": dict(method="group", levels=[4, 2, 1], stop_on_final_plateau=True),
    "mcp3": dict(method="mcp", gamma=3.0),
    "mcp10": dict(method="mcp", gamma=10.0),
}


def negative_delay_power(h, buffer=4):
    M = h.shape[1]
    dly = np.fft.fftfreq(M) * M
    neg = dly < -buffer
    p = np.abs(h) ** 2
    return float(p[:, neg].sum() / p.sum())


def rel_resid(data, h, mask):
    pred = np.abs(np.fft.ifft2(h)) ** 2
    return float(np.linalg.norm((pred - data) * mask) / np.linalg.norm(data * mask))


def save_image(h, fn, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.fft import fftshift

    p = np.abs(fftshift(h)) ** 2
    lp = np.log10(p + p.max() * 1e-12)
    N, M = h.shape
    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
    vmin = np.percentile(lp, 60)
    ax.imshow(lp.T, origin="lower", aspect="auto", cmap="gray_r", vmin=vmin, extent=(-N / 2, N / 2, -M / 2, M / 2))
    ax.set_xlabel("Doppler [pixels]")
    ax.set_ylabel("delay [pixels]")
    ax.set_ylim(-M / 8, M / 2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fn)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="J1939")
    p.add_argument("--method", choices=list(METHODS), default="l1")
    p.add_argument("--max_steps", type=int, default=None, help="default: as in the paper's notebook for the dataset")
    p.add_argument("--pad_rows", type=int, default=0,
                   help="pad axis 0 (time) to this many rows with masked (mask = 0) rows, e.g. 512 for J1939's 487: "
                        "FFTs of a prime length are ~3x slower. Changes the Doppler grid (finer), so the model is not the paper's")
    p.add_argument("--truth", default=None, help="npz (shape, idx, val) of the true wavefield: adds similarity per step")
    p.add_argument("--dense_iters", type=int, default=1000)
    p.add_argument("--out", required=True, help="output prefix")
    a = p.parse_args()
    data, mask, cfg = load(ROOT, a.dataset)
    max_steps = a.max_steps or cfg["max_steps"]
    n_orig = data.shape[0]
    if a.pad_rows:
        extra = a.pad_rows - n_orig
        assert extra >= 0
        data = np.pad(data, ((0, extra), (0, 0)))
        mask = np.pad(mask, ((0, extra), (0, 0)))  # zeros: the padded rows are treated as missing data
    truth = None
    if a.truth:
        from simulate import wavefield_similarity

        z = np.load(a.truth)
        truth = np.zeros(int(np.prod(z["shape"])), complex)
        truth[z["idx"]] = z["val"]
        truth = truth.reshape(z["shape"])
    t0 = time.time()
    log = []

    def cb(io, s):
        m = io["models"][s]
        rec = dict(step=s, n=int(np.count_nonzero(m)), resid=rel_resid(data, m, io["masks"][s]), t=time.time() - t0,
                   block=int(io["block"][s]))
        if truth is not None:
            rec["sim"] = float(wavefield_similarity(m, truth))
        log.append(rec)
        print(rec, flush=True)

    # the paper's stopping criteria; ac2f also stops at its own plateau
    io, chosen = run_variant(data, mask=mask, desired_components=60, max_steps=max_steps, clean_RFI=cfg["clean_RFI"],
                             stop=check_stopping, step_callback=cb, **METHODS[a.method])
    sel = chosen if chosen > 0 else max(io["models"])
    sparse, m = io["models"][sel], io["masks"][sel]
    dense = get_dense_solution(sparse, data, m, iterations=a.dense_iters)
    res = dict(dataset=a.dataset, method=a.method, pad_rows=a.pad_rows, n_orig_rows=n_orig, chosen=chosen, selected=sel, steps=max(io["models"]),
               n=int(np.count_nonzero(sparse)), sparse_resid=rel_resid(data, sparse, m), dense_resid=rel_resid(data, dense, m),
               sparse_neg_power=negative_delay_power(sparse), dense_neg_power=negative_delay_power(dense),
               wall=time.time() - t0, plateaus=io.get("plateaus", []), log=log)
    if truth is not None:
        res["sparse_sim"] = float(wavefield_similarity(sparse, truth))
        res["dense_sim"] = float(wavefield_similarity(dense, truth))
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
    with open(a.out + ".json", "w") as fh:
        json.dump(res, fh)
    idx = np.flatnonzero(sparse)
    np.savez_compressed(a.out + "_sparse.npz", shape=sparse.shape, idx=idx, val=sparse.ravel()[idx])
    np.save(a.out + "_dense.npy", dense.astype(np.complex64))
    np.save(a.out + "_mask.npy", m > 0)
    save_image(sparse, a.out + "_sparse.png", f"{a.dataset} {a.method}: sparse model (step {sel}, {res['n']} comp.)")
    save_image(dense, a.out + "_dense.png", f"{a.dataset} {a.method}: dense model (negative-delay power {res['dense_neg_power']:.3f})")


if __name__ == "__main__":
    main()
