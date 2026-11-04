"""
Run H-FISTA on the paper's datasets (as configured in the example notebooks) and
store a compact record of every λ step, so that two versions of the code can be
compared step by step.

    python review/regression.py run  --code-dir /path/to/code --dataset J0837 --out base_J0837.npz [--steps K]
    python review/regression.py compare base_J0837.npz new_J0837.npz

Datasets are read from <code-dir>/data (the Zenodo files, see README).
"""
import argparse
import importlib
import json
import logging
import os
import sys
import time

import numpy as np

DATASETS = {
    # name: (file, transpose, clean_RFI, stop, max_steps, normalise, init_mask)
    "sim12.5NF": ("simulated/simulated_12.5percent_noisefree.pkl", False, False, True, 66, False, False),
    "sim25NF": ("simulated/simulated_25percent_noisefree.fits", True, False, True, 66, False, False),
    "sim12.5": ("simulated/simulated_12.5percent_wnoise.pkl", False, True, True, 100, False, False),
    "J0837": ("J0837+0610.fits", False, True, True, 100, True, False),
    "J1939": ("J1939+2134.fits", False, True, True, 100, False, True),
}


def load(code_dir, name):
    fn, transpose, clean_RFI, stop, max_steps, normalise, init_mask = DATASETS[name]
    from helpers import get_data
    from auxiliary import generate_narrowband_RFI_mask

    data = get_data(os.path.join(code_dir, "data", fn))
    data = np.asarray(data, dtype=np.float64)
    if transpose:
        data = data.T
    if normalise:
        # as in J0837+0610.ipynb
        timestructure = np.average(np.transpose(data), axis=0)
        data = np.divide(data, np.reshape(timestructure, [len(timestructure), 1]))
    mask = generate_narrowband_RFI_mask(data, 1, 5, window=51) if init_mask else np.ones_like(data)
    return data, mask, dict(clean_RFI=clean_RFI, stop=stop, max_steps=max_steps)


def run(args):
    code_dir = os.path.abspath(args.code_dir)
    sys.path.insert(0, code_dir)
    import logger  # noqa: F401

    logging.getLogger("H-FISTA").setLevel(logging.WARNING)
    from pipeline import run_hfista

    if args.prox:
        import fista

        fista.PROX_IMPLEMENTATION = args.prox

    data, mask, cfg = load(code_dir, args.dataset)
    if args.steps:
        cfg["max_steps"] = min(cfg["max_steps"], args.steps)
    step_times = []
    last = [time.time()]

    def cb(io, step):
        now = time.time()
        step_times.append(now - last[0])
        last[0] = now
        n = np.count_nonzero(io["models"][step])
        print(f"{args.dataset} step {step}: {n} components, {step_times[-1]:.1f}s", flush=True)

    io, chosen, wall = run_hfista(data, mask=mask, step_callback=cb, **cfg)
    rec = {}
    for s in sorted(io["models"]):
        m = io["models"][s]
        idx = np.flatnonzero(m)
        rec[f"m{s}_idx"] = idx
        rec[f"m{s}_val"] = m.ravel()[idx]
        if s > 0:
            rec[f"dem{s}"] = np.asarray(io["demerits"][s], dtype=float)
            rec[f"L{s}"] = np.asarray(io["Ls"][s], dtype=float)
            rec[f"mask{s}"] = np.flatnonzero(io["masks"][s] == 0)
    rec["lambdas"] = np.array([io["lambdas"][s] for s in sorted(io["lambdas"])])
    rec["shape"] = np.array(data.shape)
    rec["meta"] = json.dumps(dict(dataset=args.dataset, chosen=chosen, wall=wall, step_times=step_times, steps=max(io["models"]), code_dir=code_dir))
    np.savez_compressed(args.out, **rec)
    print(f"{args.dataset}: chosen step {chosen}, {max(io['models'])} steps, wall {wall:.1f}s -> {args.out}")


def unpack(fn):
    z = np.load(fn, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    shape = tuple(z["shape"])
    models = {}
    for k in z.files:
        if k.startswith("m") and k.endswith("_idx"):
            s = int(k[1:-4])
            m = np.zeros(int(np.prod(shape)), complex)
            m[z[k]] = z[f"m{s}_val"]
            models[s] = m.reshape(shape)
    return z, meta, models


def compare(args):
    za, ma, A = unpack(args.a)
    zb, mb, B = unpack(args.b)
    print(f"A: {args.a}: chosen {ma['chosen']}, steps {ma['steps']}, wall {ma['wall']:.1f}s")
    print(f"B: {args.b}: chosen {mb['chosen']}, steps {mb['steps']}, wall {mb['wall']:.1f}s  (speed-up x{ma['wall']/mb['wall']:.2f})")
    worst = 0.0
    for s in sorted(set(A) & set(B)):
        a, b = A[s], B[s]
        same_support = np.array_equal(a != 0, b != 0)
        rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)
        dem = ""
        if s > 0:
            da, db = za[f"dem{s}"], zb[f"dem{s}"]
            dem = f" demerits equal: {np.array_equal(da, db)} (max rel diff {np.max(np.abs(da - db) / np.abs(da)) if len(da) == len(db) else 'len differs'})"
        worst = max(worst, rel)
        if args.verbose or not same_support or rel > 0:
            print(f"  step {s}: same support {same_support}, |a-b|/|a| = {rel:.2e}{dem}")
    print(f"steps compared: {len(set(A) & set(B))}, worst relative model difference: {worst:.2e}, bitwise identical: {worst == 0.0}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--code-dir", required=True)
    r.add_argument("--dataset", required=True, choices=list(DATASETS))
    r.add_argument("--out", required=True)
    r.add_argument("--steps", type=int, default=0)
    r.add_argument("--prox", choices=["reference", "fast"], default=None, help="fista.PROX_IMPLEMENTATION (new code only)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    run(args) if args.cmd == "run" else compare(args)
