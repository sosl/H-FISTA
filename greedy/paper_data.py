"""
Greedy pursuit on the paper's datasets (configured as in the notebooks: masks, RFI
cleaning), with the stopping rules for noisy data evaluated along one run:

  * "paper": the paper's criteria (3% sparsity, KS test on the Doppler distribution of the
    components new in a round); the answer is the round before the first trigger
  * "chi2":  the masked squared residual reaches the noise power estimated from the outer
    secondary spectrum (greedy.pursuit.noise_variance); the answer is that round
  * "cap":   the last round below 3% (sparsity alone)

The run continues to the cap so all three can be compared. Per round: components,
relative residual, chi2, and the similarity to the truth where it is known.

    python greedy/paper_data.py --dataset sim12.5 --prune_rule magnitude --out greedy/results/paper/sim12.5_greedy
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
from simulate import wavefield_similarity  # noqa: E402
from greedy.pursuit import greedy_pursuit, noise_variance, FFTCounter  # noqa: E402
from dense.real_data import negative_delay_power, rel_resid, save_image  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--prune_rule", default="magnitude", choices=["magnitude", "mcp"])
    p.add_argument("--truth", default=None)
    p.add_argument("--dense_iters", type=int, default=1000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    FFTCounter.install()
    data, mask0, cfg = load(ROOT, a.dataset)
    truth = None
    if a.truth:
        z = np.load(a.truth)
        truth = np.zeros(int(np.prod(z["shape"])), complex)
        truth[z["idx"]] = z["val"]
        truth = truth.reshape(z["shape"])
    t0 = time.time()
    log, keep = [], {}
    state = dict(prev=None, paper=None, chi2=None)

    masks = {}

    def cb(models, rnd):
        m = models[rnd]
        mask = masks[rnd]
        res = np.abs(np.fft.ifft2(m)) ** 2 - data
        s2 = noise_variance(data, mask)
        chi2 = float(np.sum((res * mask) ** 2) / (s2 * np.sum(mask)))
        rec = dict(round=rnd, n=int(np.count_nonzero(m)), resid=rel_resid(data, m, mask), chi2=chi2, fft=FFTCounter.n,
                   t=time.time() - t0)
        if truth is not None:
            rec["sim"] = float(wavefield_similarity(m, truth))
        if state["paper"] is None and rnd > 1 and check_stopping({"models": {rnd - 1: state["prev"], rnd: m}}, rnd):
            state["paper"] = rnd - 1
            keep["paper"] = (state["prev"], state["prev_mask"])
        if state["chi2"] is None and (chi2 <= 1.0 or rec["resid"] < 1e-6):
            state["chi2"] = rnd
            keep["chi2"] = (m, mask)
        keep["cap"] = (m, mask)
        state["prev"], state["prev_mask"] = m, mask
        log.append(rec)
        print(rec, flush=True)
        for d in (models, masks):  # keep memory bounded (J1939 is 2M pixels)
            for r in [r for r in d if r < rnd]:
                del d[r]

    greedy_pursuit(data, mask=mask0, prune_rule=a.prune_rule, clean_RFI=cfg["clean_RFI"], round_callback=cb, masks_out=masks)
    out = dict(dataset=a.dataset, prune_rule=a.prune_rule, wall=time.time() - t0, rounds=len(log),
               stopped={"paper": state["paper"], "chi2": state["chi2"], "cap": log[-1]["round"]}, log=log, results={})
    for rule, (m, mask) in keep.items():
        dense = get_dense_solution(m, data, mask, iterations=a.dense_iters) if a.dense_iters else m
        r = dict(n=int(np.count_nonzero(m)), sparse_resid=rel_resid(data, m, mask), dense_resid=rel_resid(data, dense, mask),
                 sparse_neg_power=negative_delay_power(m), dense_neg_power=negative_delay_power(dense))
        if truth is not None:
            r["sparse_sim"] = float(wavefield_similarity(m, truth))
            r["dense_sim"] = float(wavefield_similarity(dense, truth))
        out["results"][rule] = r
        save_rule = next(r_ for r_ in ("paper", "chi2", "cap") if r_ in keep)  # first rule that fired; the cap always exists
        if rule == save_rule:
            idx = np.flatnonzero(m)
            np.savez_compressed(a.out + "_sparse.npz", shape=m.shape, idx=idx, val=m.ravel()[idx])
            save_image(m, a.out + "_sparse.png", f"{a.dataset} greedy ({a.prune_rule}, {rule} stop): sparse model ({r['n']} comp.)")
            save_image(dense, a.out + "_dense.png",
                       f"{a.dataset} greedy ({a.prune_rule}): dense model (negative-delay power {r['dense_neg_power']:.3f})")
    print(json.dumps({k: v for k, v in out.items() if k != "log"}), flush=True)
    with open(a.out + ".json", "w") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    main()
