"""
Reproductions for the issues listed in review/REVIEW.md.

Each check builds the smallest scenario that triggers the issue and reports
whether the behaviour is present (CONFIRMED) or not (NOT REPRODUCED). Run from the
repository root:

    python review/repro_issues.py            # all checks
    python review/repro_issues.py B1 B4      # selected checks

Checks marked "needs data" use the Zenodo files under ./data (see README).
"""
import io as _io
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import logger  # noqa: E402

logging.getLogger("H-FISTA").setLevel(logging.ERROR)

from scipy.fft import ifftn  # noqa: E402

RESULTS = {}


def check(code, title):
    def deco(fn):
        fn.code = code
        fn.title = title
        RESULTS[code] = fn
        return fn

    return deco


def small_problem(N=64, M=64, seed=0):
    from simulate import simulate

    sim = simulate(N=N, M=M, density=0.125, seed=seed)
    h0 = np.zeros((N, M), dtype=complex)
    h0[0, 0] = np.sqrt(np.mean(sim.data)) * N * M
    return sim, h0


# --------------------------------------------------------------------------------------
# Bugs
# --------------------------------------------------------------------------------------


@check("B1", "CLI crashes on the first λ step: unexpected keyword 'force_more'")
def b1():
    from simulate import simulate
    import pickle

    sim = simulate(N=64, M=64, density=0.125, seed=0)
    with tempfile.TemporaryDirectory() as d:
        fn = os.path.join(d, "sim.pkl")
        with open(fn, "wb") as fh:
            pickle.dump(sim.data, fh)
        p = subprocess.run(
            [sys.executable, os.path.join(ROOT, "HFISTA.py"), "--data", fn, "--max_iterations", "2", "--prefix", os.path.join(d, "out")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
    err = p.stderr.strip().splitlines()
    msg = err[-1] if err else ""
    return "force_more" in msg, msg


@check("B2", "free_mem_from_dict raises NameError ('key' used before assignment) -> --partial-frequency crashes")
def b2():
    from auxiliary import free_mem_from_dict

    io = {"models": {0: 1, 1: 2, 2: 3}, "masks": {0: 1, 1: 2, 2: 3}}
    try:
        free_mem_from_dict(io)
    except NameError as e:
        return True, repr(e)
    except UnboundLocalError as e:
        return True, repr(e)
    return False, "no exception"


@check("B3", "argparse: type=bool / untyped thresholds -> '--sparse False' is True; '--spatial_threshold 1e-5' is a str")
def b3():
    sys.argv = ["HFISTA.py"]
    import HFISTA

    parser = HFISTA.setup_argparse()
    a = parser.parse_args(["--sparse", "False", "--spatial_stopping", "False", "--spatial_threshold", "1e-5"])
    problems = []
    if a.sparse is True:
        problems.append("--sparse False parsed as True")
    if bool(a.spatial_stopping):
        problems.append(f"--spatial_stopping False parsed as truthy {a.spatial_stopping!r}")
    if isinstance(a.spatial_threshold, str):
        try:
            0.5 > a.spatial_threshold
        except TypeError as e:
            problems.append(f"spatial_threshold is str -> {e}")
    return bool(problems), "; ".join(problems)


@check("B4", "fista(eps=...) raises IndexError on the first iteration (demerits[-2] with 1 element)")
def b4():
    from lib import Residual
    from fista import fista

    sim, h0 = small_problem()
    r = Residual(sim.data, h0, None, np.ones_like(sim.data))
    try:
        fista(h0, r, 5, 1e-3, eps=1e-12, backtrack=True, s=r.get_Lipschitz_constant_grad(), delay_for_inf=-4)
    except IndexError as e:
        return True, repr(e)
    return False, "no exception"


@check("B5", "fista(eps=...) / zero-solution messages say 'interrupting'/'stopping' but iterations continue")
def b5():
    import fista as F
    from lib import Residual

    sim, h0 = small_problem()
    r = Residual(sim.data, h0, None, np.ones_like(sim.data))
    # Avoid B4 by pre-seeding: call take_fista_step directly with a demerits history.
    calls = {"n": 0}
    orig = r.get_func_val

    def counting(w):
        calls["n"] += 1
        return orig(w)

    r.get_func_val = counting
    # eps huge -> "achieved precision" on every step after the first
    demerits = np.array([np.inf])
    x = y = h0
    t = 1.0
    for i in range(1, 6):
        x, y, L, t, _, demerits = F.take_fista_step(
            i, r, False, 1 / r.get_Lipschitz_constant_grad(), None, None, y, 1e-3, -4, np.array([[0, 0]]), 0.0, [0, 0], np.array([]), t, x, demerits, None, [], 1e300
        )
    return len(demerits) == 6, f"{len(demerits) - 1} steps taken although eps criterion met after step 1"


@check("B6", "very_verbose is ignored: second setLevel() overwrites the DEBUGV level")
def b6():
    from lib import Residual
    from fista import fista, log

    sim, h0 = small_problem()
    r = Residual(sim.data, h0, None, np.ones_like(sim.data))
    fista(h0, r, 1, 1e-3, backtrack=True, s=r.get_Lipschitz_constant_grad(), delay_for_inf=-4, very_verbose=True)
    lvl = log.level
    return lvl != logger.DEBUGV, f"logger level after very_verbose=True: {logging.getLevelName(lvl)}"


@check("B7", "complex_phase_fix fallback uses complex argmax (lexicographic) instead of the largest modulus")
def b7():
    from fista import complex_phase_fix

    x = np.zeros((4, 4), dtype=complex)
    x[1, 1] = 0.1 + 0.0j  # largest real part
    x[2, 2] = 0.0 + 5.0j  # largest modulus
    # coordinate [0, 0] has zero modulus -> fallback: rotate so that the largest component is real
    y = complex_phase_fix(x.copy(), [[0, 0]])
    return abs(y[2, 2].imag) > 1e-12, f"after the fallback the largest-modulus component is {complex(y[2, 2]):.3g} (expected real)"

@check("B8", "odd delay-axis length: the largest positive delay is treated as negative (λ=inf)")
def b8():
    from fista import construct_lambda_matrix

    M = 5  # fftfreq(5) = [0, 1, 2, -2, -1]
    lam = construct_lambda_matrix([3, M], 1.0, delay_for_inf=0)
    return np.isinf(lam[0, 2]), f"λ at delay index 2 (frequency +2) = {lam[0, 2]}"


@check("B9", "HFISTA.py: chosen_step > 1 discards a valid solution when stopping triggers at step 2")
def b9():
    src = open(os.path.join(ROOT, "HFISTA.py")).read()
    return "if chosen_step > 1:" in src, "stopping at step 2 sets chosen_step=1, which is then rejected"


@check("B10", "HFISTA.py helpers use globals (data, h_init) instead of their arguments")
def b10():
    import inspect

    sys.argv = ["HFISTA.py"]
    import HFISTA

    used = []
    for fn, name in ((HFISTA.populate_step_zero_io, "data"), (HFISTA.get_lambda_loop_config, "h_init")):
        if name in inspect.getsource(fn) and name not in inspect.signature(fn).parameters:
            used.append(f"{fn.__name__} reads global '{name}'")
    return bool(used), "; ".join(used) or "no globals used"

@check("B11", "Residual: l2_penalty enters the gradient but not get_func_val -> backtracking/demerit inconsistent")
def b11():
    from lib import Residual

    sim, h0 = small_problem()
    rng = np.random.default_rng(1)
    w = sim.wavefield * sim.data.size + 1e-3 * (rng.standard_normal(h0.shape) + 1j * rng.standard_normal(h0.shape))
    r0 = Residual(sim.data, w, None, np.ones_like(sim.data))
    r2 = Residual(sim.data, w, None, np.ones_like(sim.data), l2_penalty=1.0)
    same_f = np.isclose(r0.get_func_val(w), r2.get_func_val(w))
    diff_g = not np.allclose(r0.get_derivative(w), r2.get_derivative(w))
    return same_f and diff_g, "f(h) identical with and without l2 while ∇f differs"


@check("B12", "stored residual/prediction for a step are from before the final hard threshold")
def b12():
    from pipeline import run_hfista
    from lib import Residual

    sim, _ = small_problem(seed=3)
    io, _, _ = run_hfista(sim.data, max_steps=8, stop=False)
    mism = []
    for s in range(1, 9):
        r = Residual(sim.data, io["models"][s], None, io["masks"][s])
        if not np.allclose(r.residual, io["residuals"][s]):
            mism.append(s)
    return bool(mism), f"steps where io['residuals'] != residual(io['models']): {mism}"


@check("B13", "get_paper_figure(dense=True) (the default) crashes unless resid is passed")
def b13():
    import plotting  # noqa: F401
    import inspect

    sig = inspect.signature(plotting.get_paper_figure)
    return sig.parameters["dense"].default is True and sig.parameters["resid"].default is None, "dense=True uses resid.wavefield with resid=None"


@check("B14", "get_dense_wavefield_plot_from_sparse passes the wavefield untransposed (axes swapped vs get_paper_figure)")
def b14():
    import inspect
    import plotting

    src = inspect.getsource(plotting.get_dense_wavefield_plot_from_sparse)
    return "dense_wavefield,\n" in src and "dense_wavefield.T" not in src, "get_paper_figure passes .T, this helper does not"


@check("B15", "get_wavefield_plot: subint_time<=0 overwrites the Doppler label with 'Delay [aribtrary]'")
def b15():
    import inspect
    import plotting

    src = inspect.getsource(plotting.get_wavefield_plot)
    return 'ds_label = "Delay [aribtrary]"' in src, "label typo and wrong variable"


@check("B16", "read_psrflux: NameError if the header lacks 'MJD0:'")
def b16():
    from helpers import read_psrflux

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("# source: test\n")
        for t in range(2):
            for f in range(3):
                fh.write(f"{t} {f} {t*1.0} {100+f} 1.0 0.1\n")
        fn = fh.name
    try:
        read_psrflux(fn)
    except NameError as e:
        return True, repr(e)
    except UnboundLocalError as e:
        return True, repr(e)
    finally:
        os.unlink(fn)
    return False, "no exception"


@check("B17", "fista(niter=0) crashes (NameError on 'i', IndexError on demerits)")
def b17():
    from lib import Residual
    from fista import fista

    sim, h0 = small_problem()
    r = Residual(sim.data, h0, None, np.ones_like(sim.data))
    try:
        fista(h0, r, 0, 1e-3)
    except (NameError, IndexError, UnboundLocalError) as e:
        return True, repr(e)
    return False, "no exception"


# --------------------------------------------------------------------------------------
# Algorithmic / correctness concerns (behaviour differs from the paper or from theory)
# --------------------------------------------------------------------------------------


@check("A1", "gradient check: Residual.get_derivative is the Wirtinger derivative ∂f/∂h* of f=Σ mask R² (expected: correct)")
def a1():
    from lib import Residual

    sim, h0 = small_problem()
    rng = np.random.default_rng(0)
    w = sim.wavefield * sim.data.size * 0.9
    mask = (rng.random(sim.data.shape) > 0.1).astype(float)
    r = Residual(sim.data, w, None, mask)
    g = r.get_derivative(w).copy()
    f0 = r.get_func_val(w)
    errs = []
    for _ in range(5):
        i, j = rng.integers(0, 64, 2)
        eps = 1e-6 * np.abs(w).max()
        e = np.zeros_like(w)
        e[i, j] = eps
        dfx = (r.get_func_val(w + e) - r.get_func_val(w - e)) / (2 * eps)
        e[i, j] = 1j * eps
        dfy = (r.get_func_val(w + e) - r.get_func_val(w - e)) / (2 * eps)
        num = 0.5 * (dfx + 1j * dfy)  # ∂f/∂z* = (∂x + i ∂y)/2
        errs.append(abs(num - g[i, j]) / abs(g[i, j]))
    worst = max(errs)
    # This is a sanity check that is expected to pass, so "CONFIRMED" means the gradient is right.
    return worst < 1e-5, f"max relative error vs finite differences: {worst:.1e}"


@check("A2", "step uses y - ∇/L although the B3 test majorises with f(y)+2Re<∇,x-y>+L/2|x-y|² (optimal step 2/L)")
def a2():
    """For f = Σ R², the real gradient is 2∂f/∂h*; the quadratic upper bound used in
    backtracking (paper eq. 13) is minimised by x = y - (2/L) ∂f/∂h*. The code steps
    y - (1/L) ∂f/∂h*: a step half as long as the accepted L allows, and an l1 threshold
    λ/L that corresponds to penalty 2λ in that model. We compare plain gradient descent
    (λ = 0) on a fixed true support with both step sizes."""
    from lib import Residual
    import fista as F

    sim, _ = small_problem(N=128, M=128, seed=2)
    support = np.transpose(np.nonzero(sim.wavefield))
    rng = np.random.default_rng(0)
    start = sim.wavefield * sim.data.size * (1 + 0.3 * rng.standard_normal(sim.wavefield.shape))
    orig_apply = F.apply_prox_operators
    out = {}
    for factor in (1.0, 2.0):
        # backtrack_B3 computes y - (1/L) ∇ and passes alpha=1/L to the prox. Re-scale only
        # the step (y - (factor/L) ∇) and keep the B3 sufficient-decrease test unchanged.
        def backtrack(func, derivative, alpha, eta, y, _lambda=None, a=None, delay_for_inf=0, zero_penalty_coords=np.array([]), fix_phase_value=None, fix_phase_coords=None, fix_support=np.array([]), factor=factor):
            L = 1.0 / alpha
            g = derivative(y).copy()
            fy = func(y)
            i = 0
            while True:
                Li = L * eta**i
                x = orig_apply(_lambda, delay_for_inf, fix_phase_value, fix_phase_coords, fix_support, zero_penalty_coords, y - factor / Li * g, factor / Li)
                if func(x) <= fy + 2.0 * np.real(np.vdot(g, x - y)) + Li / 2.0 * np.linalg.norm(x - y) ** 2:
                    return Li, x
                i += 1

        F.backtrack_B3, saved = backtrack, F.backtrack_B3
        try:
            r = Residual(sim.data, start, None, np.ones_like(sim.data))
            _, dem, _, _, _ = F.fista(start, r, 200, None, backtrack=True, s=r.get_Lipschitz_constant_grad(), fix_support=support, fix_phase_value=0.0, fix_phase_coords=[0, 0])
        finally:
            F.backtrack_B3 = saved
        out[factor] = dem
    d1, d2 = out[1.0], out[2.0]
    it2 = int(np.argmax(d2 <= d1[-1])) if np.any(d2 <= d1[-1]) else len(d2)
    return d2[-1] < d1[-1], f"debiasing FISTA on the true support, 200 its: step 1/L -> {d1[-1]:.3e}, step 2/L -> {d2[-1]:.3e}; 2/L matches the 1/L final demerit after {it2} its"


@check("A3", "backtracking factor η defaults to 1.1 in code, the paper states 1.15")
def a3():
    import inspect
    import fista as F

    return inspect.signature(F.fista).parameters["eta"].default == 1.1, "fista(eta=1.1); take_lambda_step never passes eta"


# --------------------------------------------------------------------------------------
# Portability: current numpy / matplotlib
# --------------------------------------------------------------------------------------


@check("P1", "matplotlib>=3.9: 'from matplotlib.cm import get_cmap' fails -> lambda_loop (and everything importing it) fails")
def p1():
    import matplotlib

    try:
        import lambda_loop  # noqa: F401
    except ImportError as e:
        return True, f"matplotlib {matplotlib.__version__}: {e}"
    return False, f"matplotlib {matplotlib.__version__}: lambda_loop imports"

@check("P2", "numpy>=2: ndarray.newbyteorder removed -> get_data cannot load FITS files (needs data)")
def p2():
    from helpers import get_data

    try:
        d = get_data(os.path.join(ROOT, "data", "J0837+0610.fits"))
    except AttributeError as e:
        return True, f"numpy {np.__version__}: {e!r}"
    return False, f"numpy {np.__version__}: loaded {d.shape} {d.dtype}"

@check("P3", "numpy>=2: test_fista.py imports numpy.lib.function_base (removed)")
def p3():
    try:
        import test_fista  # noqa: F401
    except ImportError as e:
        return True, f"numpy {np.__version__}: {e}"
    return False, f"numpy {np.__version__}: test_fista imports"

@check("P4", "pyproject pins python <3.10 and numpy==1.22.4 (no wheels for current Python)")
def p4():
    src = open(os.path.join(ROOT, "pyproject.toml")).read()
    return 'python = ">=3.8,<3.10"' in src and 'numpy = "1.22.4"' in src, "python >=3.8,<3.10; numpy 1.22.4"


# --------------------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------------------


@check("F1", "redundant FFTs: count FFTs per backtracking FISTA iteration (minimum needed: 3)")
def f1():
    import lib
    import fista as F

    counts = {"fft": 0}
    o_ifftn, o_fftn = lib.ifftn, lib.fftn

    def c_ifftn(*a, **k):
        counts["fft"] += 1
        return o_ifftn(*a, **k)

    def c_fftn(*a, **k):
        counts["fft"] += 1
        return o_fftn(*a, **k)

    lib.ifftn, lib.fftn = c_ifftn, c_fftn
    try:
        sim, h0 = small_problem(N=128, M=128)
        r = lib.Residual(sim.data, h0, None, np.ones_like(sim.data))
        s = r.get_Lipschitz_constant_grad()
        # warm up L so most iterations need no extra backtracking
        x, *_, L = F.fista(h0, r, 20, 1e-6, backtrack=True, s=s, delay_for_inf=-4, zero_penalty_coords=np.array([[0, 0]]))
        counts["fft"] = 0
        n = 50
        F.fista(x, r, n, 1e-6, backtrack=True, s=L, delay_for_inf=-4, zero_penalty_coords=np.array([[0, 0]]))
    finally:
        lib.ifftn, lib.fftn = o_ifftn, o_fftn
    per = counts["fft"] / n
    return per > 6, f"{per:.1f} FFTs per FISTA iteration (3 needed: ifft(y), fft(grad), ifft(x))"


@check("F2", "construct_lambda_matrix rebuilt (with a Python loop over the support) at every prox call")
def f2():
    from fista import construct_lambda_matrix

    shape = (512, 512)
    rng = np.random.default_rng(0)
    supp = np.unique(np.stack([rng.integers(0, 512, 8000), rng.integers(0, 256, 8000)], 1), axis=0)
    t0 = time.perf_counter()
    for _ in range(20):
        construct_lambda_matrix(shape, 1.0, -4, zero_penalty_coords=supp)
    t_build = (time.perf_counter() - t0) / 20
    from scipy.fft import fftn

    a = rng.standard_normal(shape) + 0j
    t0 = time.perf_counter()
    for _ in range(20):
        fftn(a, workers=2)
    t_fft = (time.perf_counter() - t0) / 20
    return t_build > 0.5 * t_fft, f"512x512 with {len(supp)} approved: λ-matrix build {t_build*1e3:.1f} ms vs one FFT {t_fft*1e3:.1f} ms"


@check("F3", "hard_threshold / delete_coordinates_from_array are O(n·k) (list conversion per deletion)")
def f3():
    from lambda_loop import hard_threshold

    rng = np.random.default_rng(0)
    N = 512
    x = np.zeros((N, N), complex)
    idx = np.unique(np.stack([rng.integers(0, N, 20000), rng.integers(0, N, 20000)], 1), axis=0)
    x[idx[:, 0], idx[:, 1]] = rng.random(len(idx)) + 0j
    t0 = time.perf_counter()
    hard_threshold(x.copy(), idx, 1.0, 1.0, 0.1, coordinate_list=idx.copy())
    t = time.perf_counter() - t0
    return t > 1.0, f"{len(idx)} components, ~10% deleted: {t:.2f} s"


def main(argv):
    codes = argv or list(RESULTS)
    for c in codes:
        fn = RESULTS[c]
        buf = _io.StringIO()
        try:
            with redirect_stdout(buf):
                ok, detail = fn()
            status = "CONFIRMED" if ok else "NOT REPRODUCED"
        except Exception as e:  # pragma: no cover - a check itself failing is informative
            status, detail = "CHECK ERROR", f"{type(e).__name__}: {e}"
        print(f"[{c:>3}] {status:<14} {fn.title}\n        {detail}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
