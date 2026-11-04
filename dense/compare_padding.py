"""
Compare a run on the original grid with a run whose time axis was padded with masked rows
(real_data.py --pad_rows), e.g. J1939: 487 -> 512 rows.

The two wavefields live on different Doppler grids, so they are compared through what
the data constrain: the field E(t, ν) = ifft2(h) at the observed times (the first
n_orig rows of the padded field). Reported, for the sparse and the dense model:

  * field similarity |<E_a, E_b>| / (||E_a|| ||E_b||) over the pixels valid in both masks
    (1 = the same field up to a global phase);
  * wavefield similarity (invariant to global phase, translation and twin; simulate.align)
    between h_a and the padded model restricted to the observation window,
    fft2(E_b[:n_orig]);
  * relative residual of each model against the original data and mask;
  * stopping step, components, time per λ step, negative-delay power.

    python dense/compare_padding.py dense/results/paper/J1939_l1_rerun dense/results/paper/J1939_l1_pad512
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "review"))
from simulate import wavefield_similarity  # noqa: E402
from regression import load  # noqa: E402


def sparse(prefix):
    z = np.load(prefix + "_sparse.npz")
    h = np.zeros(int(np.prod(z["shape"])), complex)
    h[z["idx"]] = z["val"]
    return h.reshape(z["shape"])


def field_similarity(a, b, valid):
    a, b = a[valid], b[valid]
    return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rel_resid(data, E, mask):
    return float(np.linalg.norm((np.abs(E) ** 2 - data) * mask) / np.linalg.norm(data * mask))


def main(a, b):
    ra, rb = json.load(open(a + ".json")), json.load(open(b + ".json"))
    n = ra["n_orig_rows"] if "n_orig_rows" in ra else sparse(a).shape[0]
    data, mask0, _ = load(ROOT, ra["dataset"])
    ma, mb = np.load(a + "_mask.npy"), np.load(b + "_mask.npy")[:n]
    valid = ma & mb
    out = {}
    for kind in ("sparse", "dense"):
        ha = sparse(a) if kind == "sparse" else np.load(a + "_dense.npy").astype(complex)
        hb = sparse(b) if kind == "sparse" else np.load(b + "_dense.npy").astype(complex)
        # same normalisation as the forward model on each grid: E = ifft2(h)
        Ea, Eb = np.fft.ifft2(ha), np.fft.ifft2(hb)[:n]
        hb_win = np.fft.fft2(Eb)  # padded model restricted to the observation window, on the original grid
        out[kind] = dict(field_similarity=field_similarity(Ea, Eb, valid),
                         wavefield_similarity=float(wavefield_similarity(hb_win, ha)),
                         resid_a=rel_resid(data, Ea, valid), resid_b=rel_resid(data, Eb, valid))
    t_a = np.diff([0] + [x["t"] for x in ra["log"]])
    t_b = np.diff([0] + [x["t"] for x in rb["log"]])
    out["runs"] = {k: dict(chosen=r["chosen"], n=r["n"], shape_rows=r.get("pad_rows") or n,
                           sec_per_step_median=float(np.median(t)), wall=r["wall"],
                           sparse_neg_power=r["sparse_neg_power"], dense_neg_power=r["dense_neg_power"])
                   for k, r, t in (("original", ra, t_a), ("padded", rb, t_b))}
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = main(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        json.dump(res, open(sys.argv[3], "w"), indent=1)
