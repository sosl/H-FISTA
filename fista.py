import numpy as np
from scipy.linalg import norm
import logging, logger
from typing import Optional

from lib import Residual, extract_part_of_array

log = logger.setup_logger(is_debug=False)
log = logger.get_logger(__name__)


def construct_lambda_matrix(
    shape: list,
    _lambda: float,
    delay_for_inf=0,
    delay_offset=True,
    zero_penalty_coords=np.array([]),
    fix_support=np.array([]),
) -> np.ndarray:
    """
    shape: sequence of dimensions of the model

    _lambda: constant value for all values above zero delay, should be positive

    delay_for_inf: delay index after which the penalty should be infinite.

    delay_offset: if False, use (lambda(a*delay^2+1)) instead as penalty

    zero_penalty_coords: list of coordinates at which the model components are approved and should have zero penalty.
    Note: these coordinates are as expected in the Fourier space before shift, unlike the delay_for_inf above
    fix_support: list of coordinates which will have zero penalty. Unlike zero_penalty_coords, all other coordinates will have infinite penalty. This takes precedence over zero_penalty_coords

    returns an ndarray of floats
    """
    if len(fix_support) > 0:
        lambda_array = np.ones(shape) * np.inf
        coords = np.asarray(fix_support)
        lambda_array[coords[:, 0], coords[:, 1]] = 0.0
        return lambda_array

    if _lambda is None:
        _lambda = 0.0
    lambda_array = np.ones(shape) * _lambda
    log.debugv(f"Setting λ to infinite at negative delays (axis sized {shape[1]})")  # type: ignore
    # first index of the negative delays: fftfreq(M) is non-negative for indices < (M + 1) // 2
    lambda_array[:, (int(shape[1]) + 1) // 2 :] = np.inf

    if delay_for_inf < 0:
        if delay_offset:
            delay_offset = 1
        else:
            delay_offset = 0
        log.debugv(f"Setting λ to finite values for delays above {delay_for_inf}")  # type: ignore

        _, delay = np.meshgrid(np.arange(shape[0]), -np.arange(-delay_for_inf) - delay_offset)
        penalty = np.ones_like(delay).T * _lambda  # (a * _lambda * delay * delay + _lambda).T
        lambda_array[:, delay_for_inf + int(shape[1]) : int(shape[1])] = penalty[:, ::-1]

    if len(zero_penalty_coords) > 0:
        log.debugv(f"Setting λ to zero at {len(zero_penalty_coords)} chosen coordinates")  # type: ignore
        coords = np.asarray(zero_penalty_coords)
        lambda_array[coords[:, 0], coords[:, 1]] = 0.0
    return lambda_array


def apply_prox_operators(
    _lambda,
    delay_for_inf,
    fix_phase_value,
    fix_phase_coords,
    fix_support,
    zero_penalty_coordinates,
    x_np1,
    alpha,
    lambda_array=None,
):
    if lambda_array is None:
        lambda_array = construct_lambda_matrix(
            shape=x_np1.shape,
            _lambda=_lambda,
            delay_for_inf=delay_for_inf,
            zero_penalty_coords=zero_penalty_coordinates,
            fix_support=fix_support,
        )
    _lambda_array = lambda_array
    x_np1 = complex_prox_l1(x_np1, _lambda_array, 1.0 / alpha)
    if fix_phase_value is not None:
        x_np1 = complex_phase_fix(x_np1, [fix_phase_coords], fix_phase_value)

    return x_np1


# "reference": max(|x| - λ/L, 0) exp(i arg x), as in the published code.
# "fast": x max(1 - λ/(L|x|), 0), the same operator without evaluating arg and exp
#         (about 6x faster; differs from "reference" by rounding, ~1e-16 relative,
#         and leaves components with λ = 0 exactly unchanged).
PROX_IMPLEMENTATION = "reference"


def complex_prox_l1(x: np.ndarray, _lambda: np.ndarray, L: float):
    """
    Apply elementwise soft-thresholding, adjusted for complex domain (see e.g. end of chapter 3 of "A Mathematical Introduction to Compressive Sensing" by S. Foucart and H. Rauhut)
    This can also be used to constrain to support by setting _lambda to infinity at all coordinates not in support.
    """
    if PROX_IMPLEMENTATION == "fast":
        abs_x = np.abs(x)
        threshold = _lambda / L
        with np.errstate(divide="ignore", invalid="ignore"):
            out = x * np.where(abs_x > threshold, 1.0 - threshold / abs_x, 0.0)
    else:
        out = np.maximum(np.abs(x) - _lambda / L, 0) * np.exp(1j * np.angle(x))
    if np.count_nonzero(out) == 0:
        msg = "complex_prox_l1: λ/L value too large, no elements left"
        log.error(msg)
        raise ValueError(msg)
    return out


def complex_phase_fix(x: np.ndarray, coords: list, value=0):
    """
    Fix a phase of the component of a 2-dimensional array x at coords by setting the imaginary component to value.
    """
    for coord in coords:
        org_modulus = np.abs(x[coord[0], coord[1]])
        x[coord[0], coord[1]] = np.real(x[coord[0], coord[1]]) + value * 1j
        new_modulus = np.abs(x[coord[0], coord[1]])
        if new_modulus > 0:
            x[coord[0], coord[1]] *= org_modulus / new_modulus
        else:
            max_coords = np.unravel_index(np.abs(x).argmax(), x.shape)
            log.warn(
                f"complex_phase_fix: Modulus at the chosen coordinate is zero. Fixing phase of largest component at {max_coords}"
            )
            x *= np.exp(-1j * np.angle(x[max_coords[0], max_coords[1]]))

            org_modulus = np.abs(x[max_coords[0], max_coords[1]])
            x[max_coords[0], max_coords[1]] = np.real(x[max_coords[0], max_coords[1]]) + value * 1j
            new_modulus = np.abs(x[max_coords[0], max_coords[1]])
            x[max_coords[0], max_coords[1]] *= org_modulus / new_modulus

    return x


def take_fista_step(
    i,
    func,
    backtrack,
    alpha,
    s,
    eta,
    y_n,
    _lambda,
    delay_for_inf,
    zero_penalty_coords,
    fix_phase_value,
    fix_phase_coords,
    fix_support,
    t_n,
    x_n,
    demerits,
    model,
    control_indices,
    eps,
    lambda_array=None,
):
    # calculate the updated model, either with backtracking, or using fixed alpha
    # Apply proximal operators (in the case of backtracking, the operators are applied within the backtracking)
    if backtrack:
        if i == 0:
            alpha = 1.0 / s  # type: ignore
        # one inverse + one forward FFT for both the demerit and the gradient at y_n
        f_y, grad_y = func.get_func_val_and_derivative(y_n)
        L, x_np1, func_val = backtrack_B3(
            func.get_func_val,
            func.get_derivative,
            alpha,
            eta,
            y_n,
            _lambda,
            delay_for_inf=delay_for_inf,
            zero_penalty_coords=zero_penalty_coords,
            fix_phase_value=fix_phase_value,
            fix_phase_coords=fix_phase_coords,
            fix_support=fix_support,
            lambda_array=lambda_array,
            grad_y=grad_y,
            func_val_y=f_y,
            return_func_val=True,
        )
        alpha = 1.0 / L
    else:
        # Need to define L when not backtracking since we're returning it:
        L = 1.0 / alpha
        x_np1 = y_n - alpha * func.get_derivative(y_n)
        x_np1 = apply_prox_operators(
            _lambda,
            delay_for_inf,
            fix_phase_value,
            fix_phase_coords,
            fix_support,
            zero_penalty_coords,
            x_np1,
            alpha,
            lambda_array=lambda_array,
        )
        func_val = func.get_func_val(x_np1)

    t_np1 = (1 + np.sqrt(1 + 4 * np.power(t_n, 2))) / 2.0
    y_np1 = x_np1 + (t_n - 1.0) / t_np1 * (x_np1 - x_n)

    if control_indices:
        model = extract_part_of_array(model, x_np1, control_indices)
    demerits = np.append(demerits, func_val)
    x_n = x_np1
    y_n = y_np1
    t_n = t_np1
    if eps and len(demerits) > 1:
        if np.abs(func_val - demerits[-2]) < eps:
            log.info(f"Achieved required precision ({eps}) in iteration {i}, interrupting.")
    if np.count_nonzero(x_np1) == 0:
        log.info("solution brought to zero, stopping")
    if i % 50 == 0:
        log.info(
            f"in iteration {i}, x_np1 has {np.count_nonzero(x_np1)} non-zero elements with demerit {demerits[-1]:.3g}"
        )
    return x_n, y_n, L, t_n, model, demerits


def fista(
    x_0: np.ndarray,
    func: Residual,
    niter: int,
    _lambda: Optional[float] = None,
    alpha: float = -1.0,
    delay_for_inf: Optional[int] = 0,
    zero_penalty_coords: np.ndarray = np.array([]),
    eps: Optional[float] = None,
    backtrack: Optional[bool] = False,
    s: Optional[float] = 5.0,
    eta: Optional[float] = 1.1,
    control_indices: Optional[list] = [],
    fix_phase_value: Optional[float] = None,
    fix_phase_coords: Optional[list] = None,
    fix_support: np.ndarray = np.array([]),
    verbose=False,
    very_verbose=False,
):
    """
    x_0: starting point
    func: function object
    alpha: step size (used only if get_Lipschitz_constant_grad doesn't return a positive value, niter). Will get overwritten if the residual object returns a positive Lipschitz constant

    _lambda: Use LASSO with _lambda as L1 penalty weight.
    a, delay_for_inf: Allow negative delays with penalty specified as a*delay^2+_lambda up to delay_for_inf
    zero_penalty_coords: list of coordinates of pre-approved components for which the penalty should be set to 0
    fix_support: set zero penalty at these coordinates and infinite elsewhere. This takes precedence over zero_penalty_coords

    eps: stop iterations if the change of the demerit is less than eps

    backtrack: use backtracking to determine the Lipschitz constant>
    s: initial guess for backtracking
    eta: multiply Lipschitz guess by this factor during backtracking steps

    control_indices: A list of indices to slice the models to provide a control output for every iterration. Only useful for debugging
    fix_phase_value: Fix the complex phase of origin component to this value
    fix_phase_coords: Coordinates of the component at which to fix the phase
    """

    log.setLevel(logger.DEBUGV if very_verbose else (logging.DEBUG if verbose else logging.INFO))

    log.info(f"Running FISTA from an initial guess with {len(np.transpose(np.nonzero(x_0)))} components (approved: {len(zero_penalty_coords)} and fixed support {len(fix_support)})")  # type: ignore
    if len(fix_support) > 0:
        log.info(f"Fixing support")
    x_tmp = np.array([[]])
    if control_indices:
        x_tmp = extract_part_of_array(x_tmp, x_0, control_indices)
    model = x_tmp
    demerits = np.array([])

    n_comp = np.array([])
    n_comp_zero_penalty = np.array([])

    # L_k = 1 / alpha
    if func.get_Lipschitz_constant_grad() > 0:
        alpha = 1.0 / func.get_Lipschitz_constant_grad()

    t_n = 1.0

    # Λ depends only on the arguments of this call, so build it once rather than at every prox evaluation
    lambda_array = construct_lambda_matrix(
        shape=x_0.shape,
        _lambda=_lambda,
        delay_for_inf=delay_for_inf,
        zero_penalty_coords=zero_penalty_coords,
        fix_support=fix_support,
    )

    x_n = x_0
    # x_np1 = x_0
    y_n = x_0

    for i in range(niter):
        if (i + 1) % 50 == 0:
            log.debug(f"Completed {i+1} iterations")
        else:
            log.debugv(f"Completed {i+1} iterations")
        x_n, y_n, L, t_n, model, demerits = take_fista_step(
            i,
            func,
            backtrack,
            alpha,
            s,
            eta,
            y_n,
            _lambda,
            delay_for_inf,
            zero_penalty_coords,
            fix_phase_value,
            fix_phase_coords,
            fix_support,
            t_n,
            x_n,
            demerits,
            model,
            control_indices,
            eps,
            lambda_array=lambda_array,
        )

        alpha = 1 / L
        n_comp = np.append(n_comp, np.count_nonzero(x_n))
        n_comp_zero_penalty_tmp = len(zero_penalty_coords)
        if n_comp_zero_penalty_tmp == 0 and len(fix_support) > 0:
            n_comp_zero_penalty_tmp = len(fix_support)
        n_comp_zero_penalty = np.append(n_comp_zero_penalty, n_comp_zero_penalty_tmp)

    if control_indices:
        model = model.reshape(len(demerits) + 1, len(control_indices))
    else:
        model = x_n
    demerits = np.real_if_close(demerits)

    log.info(f"Arrived at model with {np.count_nonzero(model)} components/ {demerits[-1]:.3g} after {i+1} iterations")  # type: ignore

    return model, demerits, n_comp, n_comp_zero_penalty, L


def quad_approx(func, deriv, L, x, y):
    return func(y) + 2.0 * np.real(np.vdot(deriv(y), (x - y))) + L / 2.0 * np.power(norm((x - y)), 2)


def backtrack_B3(
    func,
    derivative,
    alpha,
    eta,
    y,
    _lambda=None,
    a=None,
    delay_for_inf=0,
    zero_penalty_coords=np.array([]),
    fix_phase_value=None,
    fix_phase_coords=None,
    fix_support=np.array([]),
    lambda_array=None,
    grad_y=None,
    func_val_y=None,
    return_func_val=False,
):
    """
    This is an implementation of the B3 procedure described in
    "First-order methods in optimization"
    by Amir Beck", 2017, chapter 10.7.1, p 291

    grad_y, func_val_y: optional precomputed derivative and demerit at y. They do not
    change while backtracking, so they are evaluated (at most) once per call.
    return_func_val: also return the demerit at the accepted point (already evaluated
    by the sufficient-decrease test).
    """
    s = 1.0 / alpha
    assert s > 0.0
    assert eta > 1.0

    if lambda_array is None:
        lambda_array = construct_lambda_matrix(
            shape=y.shape,
            _lambda=_lambda,
            delay_for_inf=delay_for_inf,
            zero_penalty_coords=zero_penalty_coords,
            fix_support=fix_support,
        )
    if grad_y is None:
        grad_y = derivative(y)
    if func_val_y is None:
        func_val_y = func(y)

    def prox(z, step):
        return apply_prox_operators(
            _lambda,
            delay_for_inf,
            fix_phase_value,
            fix_phase_coords,
            fix_support,
            zero_penalty_coords,
            z,
            step,
            lambda_array=lambda_array,
        )

    def quad(L, x):
        return func_val_y + 2.0 * np.real(np.vdot(grad_y, (x - y))) + L / 2.0 * np.power(norm((x - y)), 2)

    L_k = s

    L_kp1 = L_k

    y_kp1 = y - 1.0 / L_kp1 * grad_y
    # x_kp1 is what's denoted as T_L_k(y^k) in the book, i.e. "proxed" y_kp1
    x_kp1 = prox(y_kp1, 1 / L_kp1)

    i = 0

    f_x = func(x_kp1)
    while f_x > quad(L_kp1, x_kp1):
        i += 1
        L_kp1 = L_k * np.power(eta, i)

        y_kp1 = y - 1.0 / L_kp1 * grad_y
        x_kp1 = prox(y_kp1, 1 / L_kp1)
        f_x = func(x_kp1)
    if i > 0:
        log.info(f"backtrack_B3 found {L_kp1:.3g} after {i} iterations. Model has {np.count_nonzero(x_kp1)} el.")  # type: ignore
    else:
        log.debug(f"backtrack_B3 found {L_kp1:.3g} after {i} iterations. Model has {np.count_nonzero(x_kp1)} el.")  # type: ignore

    if return_func_val:
        return L_kp1, x_kp1, f_x
    return L_kp1, x_kp1
