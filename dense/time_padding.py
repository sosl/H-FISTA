"""Isolated timing of the first K λ steps of l1 on J1939, original grid vs padded with masked rows.
Run with nothing else on the machine.   python dense/time_padding.py 3 512"""
import json, logging, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "review")]
import logger  # noqa: E402,F401
logging.disable(logging.INFO)
import fista as F  # noqa: E402
F.PROX_IMPLEMENTATION = "fast"
from regression import load  # noqa: E402
from dense.variants import run_variant  # noqa: E402
from dense.greedy_fft_counter import FFTCounter  # noqa: E402

K, pad = int(sys.argv[1]), int(sys.argv[2])
D, m, cfg = load(ROOT, "J1939")
FFTCounter.install()
for rows in (D.shape[0], pad):
    Dp, mp = np.pad(D, ((0, rows - D.shape[0]), (0, 0))), np.pad(m, ((0, rows - D.shape[0]), (0, 0)))
    FFTCounter.n = 0
    t = time.time()
    run_variant(Dp, mask=mp, desired_components=60, max_steps=K, clean_RFI=True)
    w = time.time() - t
    print(json.dumps(dict(rows=rows, steps=K, wall=w, fft=FFTCounter.n, ms_per_fft_incl_overhead=1000 * w / FFTCounter.n)), flush=True)
