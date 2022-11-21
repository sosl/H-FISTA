from lib import Residual
from single_step_newton import get_minimum_demerit_resid
from fista import fista
import logger

log = logger.get_logger(__name__)


def get_dense_solution(sparse_wavefield, data, mask, method="FISTA", **kwargs):
    """
    Obtain a dense solution from a sparse wavefield.
    This can be done either using a single step to an estimated demerit minimum,
    or using unconstrained FISTA.

    Arguments:
    sparse_wavefield: input wavefield to use as a starting point for obtaining a dense solution
    data: dynamic spectrum used for optimisation of the wavefield.
    mask: mask to be applied to data, e.g., to mask RFI or gaps in observations.
    method: string defining the method of densification. Can be "FISTA", or "single step"
    kwargs: kwargs dictionary for any extra arguments the densifier can accept, such as number of iterations for FISTA. If none provided, default values will be used.
    """

    densifier = get_densifier(method)
    dense_wavefield = densifier(sparse_wavefield, data, mask, **kwargs)

    return dense_wavefield


def get_densifier(method):
    if method == "FISTA":
        return get_dense_solution_via_FISTA
    elif method == "single step":
        return get_dense_solution_via_single_step
    else:
        raise ValueError(method)


def get_dense_solution_via_single_step(sparse, data, mask):
    """
    Obtain a dense solution using a single step towards the estimated demerit minimum
    """
    resid_sparse = Residual(data, sparse, None, mask)
    resid_best = get_minimum_demerit_resid(resid_sparse, data)
    return resid_best.wavefield


def get_dense_solution_via_FISTA(sparse, data, mask, iterations=None):
    if iterations is None:
        iterations = 1000
    resid_refined = Residual(data, sparse, None, mask)

    delay_axis = 1

    fit, demerit, _, _, L = fista(
        sparse,
        resid_refined,
        iterations,
        backtrack=True,
        s=resid_refined.get_Lipschitz_constant_grad(),
        delay_for_inf=-data.shape[delay_axis],
        fix_phase_value=0.0,
        fix_phase_coords=[0, 0],
        verbose=False,
    )

    log.info(f"Densification with FISTA achieved demerit of {demerit[-1]}")

    return fit
