"""FFT count and wall time of H-FISTA (l1, published loop via dense.variants) on the paper's data,
as configured in the notebooks, for comparison with greedy/paper_data.py.
    python greedy/l1_paper_ffts.py sim12.5 sim12.5NF J0837 sim25NF > greedy/results/paper/l1_ffts.jsonl"""
import json, logging, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "review")]
import logger  # noqa: E402,F401
logging.disable(logging.INFO)
import fista as F  # noqa: E402
F.PROX_IMPLEMENTATION = "fast"
from regression import load  # noqa: E402
from stopping import check_stopping  # noqa: E402
from dense.variants import run_variant  # noqa: E402
from greedy.pursuit import FFTCounter  # noqa: E402

FFTCounter.install()
for ds in sys.argv[1:]:
    data, mask, cfg = load(ROOT, ds)
    FFTCounter.n = 0
    t0 = time.time()
    io, chosen = run_variant(data, mask=mask, desired_components=60, max_steps=cfg["max_steps"], clean_RFI=cfg["clean_RFI"],
                             stop=check_stopping)
    sel = chosen if chosen > 0 else max(io["models"])
    print(json.dumps(dict(dataset=ds, chosen=chosen, selected=sel, n=int(np.count_nonzero(io["models"][sel])),
                          fft=FFTCounter.n, wall=time.time() - t0)), flush=True)
