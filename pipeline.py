"""
Programmatic H-FISTA driver.

This mirrors the loop used in the example notebooks (simulated_*.ipynb), so that
experiments and regression checks can run H-FISTA without a notebook.
"""
import time
from typing import Optional

import numpy as np

from lib import Residual
from lambda_loop import take_lambda_step, initialize_io_dict, get_initial_lambda
from stopping import check_stopping


def run_hfista(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    desired_components: int = 60,
    lambda_scale: float = 1.15,
    FISTA_niter: int = 80,
    delay_for_inf: int = -4,
    backtrack: bool = True,
    clean_RFI: bool = False,
    max_steps: int = 100,
    stop: bool = True,
    init_coords=(0, 0),
    step_callback=None,
    **step_kwargs,
):
    """Run H-FISTA as in the example notebooks.

    Returns:
        io dictionary, chosen step (-1 if no stopping criterion triggered), wall time in seconds
    """
    start = time.time()
    if mask is None:
        mask = np.ones_like(data)
    N, M = data.shape
    init_coords = list(init_coords)

    h_init = np.zeros((N, M), dtype=complex)
    h_init[init_coords[0], init_coords[1]] = np.sqrt(np.mean(data * mask)) * N * M + 0.0j

    io = initialize_io_dict()
    io["niters"][0] = 0
    io["masks"][0] = mask
    resid = Residual(data, h_init, None, mask)
    io["lambdas"][0] = get_initial_lambda(desired_components, delay_for_inf, resid) * lambda_scale
    io["models"][0] = h_init
    io["demerits"][0] = -1
    io["predictions"][0] = resid.prediction
    io["residuals"][0] = resid.residual

    approved = np.transpose(np.nonzero(h_init))
    chosen_step = -1
    for step in range(1, max_steps + 1):
        io["niters"][step] = FISTA_niter
        ref_lambda = io["lambdas"][step - 1] / lambda_scale
        config = dict(
            backtrack=backtrack,
            fix_phase_value=0.0,
            fix_phase_coords=init_coords,
            ref_lambda=ref_lambda,
            delay_for_inf=delay_for_inf,
            zero_penalty_coords=approved,
            clean_RFI=clean_RFI,
        )
        config.update(step_kwargs)
        approved = take_lambda_step(data, io, step, **config)
        io["lambdas"][step] = ref_lambda
        if step_callback is not None:
            step_callback(io, step)
        if stop and step > 1 and check_stopping(io, step):
            chosen_step = step - 1
            break
    return io, chosen_step, time.time() - start
