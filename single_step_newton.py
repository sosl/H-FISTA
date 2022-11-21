from lib import Residual
import numpy as np
import logger

log = logger.get_logger(__name__)


def get_demerit_at_wavefield(data, wavefield, mask=None):
    if mask is None:
        mask = np.ones(wavefield.shape)
    resid = Residual(data, wavefield, None, mask)  # type: ignore
    return resid.get_full_demerit()


def get_demerits_and_offset(data, wavefield, mask, gradient, step, h_frac):
    """
    get demerits at wavefield + step * gradient as well as the step fractionally offset by h_frac, as well as the step size
    """
    demerit = get_demerit_at_wavefield(data, wavefield - step * gradient, mask)
    demerit_h = get_demerit_at_wavefield(data, wavefield - step * (1.0 + h_frac) * gradient, mask)
    h = step * h_frac
    return demerit, demerit_h, h


def estimate_derivative(value_at_x, value_at_x_plus_h, h):
    """
    Estimate derivative from definition but using a fininte value of h
    """
    return (value_at_x_plus_h - value_at_x) / h


def take_newton_step(x0, f_x, f_prime_x):
    """
    Take a step in Newton's method of finding roots
    """
    return x0 - f_x / f_prime_x


def get_minimum_demerit_step(step_guess, wavefield, wavefield_grad, mask, data, trial_fraction=0.1, h_fraction=0.01):
    """
    Find the best step size assuming the demerit is quadratic near the wavefield obtained with step_guess step size and known gradient wavefield_grad
    If the demerit is quadratic, we can find the minimum in one step by estimating the first and second derivatives since the latter is constant.
    """
    deriv_at_guess = estimate_derivative(
        *get_demerits_and_offset(data, wavefield, mask, wavefield_grad, step_guess, h_fraction)
    )
    deriv_at_trial = estimate_derivative(
        *get_demerits_and_offset(data, wavefield, mask, wavefield_grad, step_guess * (1 + trial_fraction), h_fraction)
    )
    second_deriv = estimate_derivative(deriv_at_guess, deriv_at_trial, step_guess * trial_fraction)

    return take_newton_step(step_guess, deriv_at_guess, second_deriv)


def get_analytical_step_size(resid):
    """
    Get an analytical step size for the provided residual object
    """
    grad = resid.get_derivative()
    real_grad_Hc = np.real(grad * np.conj(resid.H))

    numerator = np.sum(resid.RFI_mask * resid.residual * real_grad_Hc)
    denominator = 2 * np.sum(real_grad_Hc * real_grad_Hc)

    return numerator / denominator * np.prod(grad.shape), np.copy(grad)


def get_minimum_demerit_resid(resid, data, alternative_fractional_guess=0.01):
    step_guess, gradient = get_analytical_step_size(resid)
    demerit_guess = get_demerit_at_wavefield(data, resid.wavefield - step_guess * gradient, resid.RFI_mask)
    demerit_start = get_demerit_at_wavefield(
        data, resid.wavefield - alternative_fractional_guess * step_guess * gradient, resid.RFI_mask
    )

    if demerit_start < demerit_guess:
        step_guess = alternative_fractional_guess * step_guess
        log.info(
            f"Demerit at the alternate guess is lower than at analytical guess => Using {alternative_fractional_guess * step_guess:.1g} instead of ({step_guess:.1g}) as the initial guess"
        )
    else:
        log.info("Analytical guess was better than the alternative guess")

    best_step = get_minimum_demerit_step(step_guess, resid.wavefield, gradient, resid.RFI_mask, data)

    return Residual(data, np.copy(resid.wavefield - best_step * gradient), None, np.copy(resid.RFI_mask))


def get_minimum_demerit_resid_at_fixed_sparse(resid, data, alternative_fractional_guess=0.01):
    step_guess, gradient = get_analytical_step_size(resid)
    demerit_guess = get_demerit_at_wavefield(data, resid.wavefield - step_guess * gradient, resid.RFI_mask)
    demerit_start = get_demerit_at_wavefield(
        data, resid.wavefield - alternative_fractional_guess * step_guess * gradient, resid.RFI_mask
    )

    if demerit_start < demerit_guess:
        step_guess = alternative_fractional_guess * step_guess
        log.info(
            f"Demerit at the alternate guess is lower than at analytical guess => Using {alternative_fractional_guess * step_guess:.1g} instead of ({step_guess:.1g}) as the initial guess"
        )
    else:
        log.info("Analytical guess was better than the alternative guess")

    best_step = get_minimum_demerit_step(step_guess, resid.wavefield, gradient, resid.RFI_mask, data)
    dense_wf = np.copy(resid.wavefield) - best_step * gradient
    dense_wf[np.nonzero(resid.wavefield)] = resid.wavefield[np.nonzero(resid.wavefield)]

    return Residual(data, dense_wf, None, np.copy(resid.RFI_mask))
