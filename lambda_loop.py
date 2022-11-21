from lib import Residual
from fista import fista
from helpers import set_diff2d, get_new_component_coordinates
from auxiliary import generate_narrowband_RFI_mask, update_RFI_mask
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from scipy.fft import fftshift

import logging, logger

log = logger.get_logger(__name__)


def hard_threshold(
    x: np.ndarray, support: np.ndarray, factor: float, L: float, _lambda: float, coordinate_list=np.array([])
):
    """
    Apply a hard threshold (as opposed to soft threshold done during FISTA loop) to elements of x from among coordinates listed in support.
    That is, any elements below factor * _lambda / L will be set to zero.
    If coordinate list is provided, the thresholded component coordinates will be removed from the list if present

    Return the thresholded array x and count of hard-thresholded components
    """
    deleted = []
    counter = 0
    threshold = factor * _lambda / L
    for i, j in support:
        if np.abs(x[i, j]) < threshold:
            x[i, j] = 0.0 + 0.0j
            counter += 1
            deleted.append([i, j])
            if len(coordinate_list) > 0:
                coordinate_list = delete_coordinates_from_array(coordinate_list, i, j)
    log.debug(f"Hard-thresholded {counter} elements with threshold {threshold} (factor was {factor})")
    log.debugv(f"Coordinates of the hard-thresholded components: {deleted}")
    return x, counter, coordinate_list, deleted


def delete_coordinates_from_array(input: np.ndarray, i: int, j: int):
    # find the component in the input list and delete if present
    input_list = input.tolist()
    try:
        index = input_list.index([i, j])
        log.debugv(f"Found [{i}, {j}] on the list, deleting")  # type: ignore
        return np.delete(input, index, axis=0)
    except ValueError:
        log.debugv(f"When deleting, [{i}, {j}] was not on the list")  # type: ignore
        return input


def plot_a_step(
    io: dict,
    step: int,
    init_coords=[0, 0],
    x_centre=135,
    y_centre=512,
    xrange=135,
    yrange=512,
    deleted_x={},
    deleted_y={},
):
    plt.figure()
    coords_λ = np.nonzero(fftshift(io["models_λ"][step]))
    plt.scatter(coords_λ[0], coords_λ[1])
    plt.title(f"model after λ>0 in step {step}")
    plt.xlim(x_centre - xrange, x_centre + xrange)
    plt.ylim([y_centre - 12, y_centre + yrange])

    ax = plt.gca()
    ax.axhline(y_centre, color="r")
    ax.axvline(x_centre, color="r")

    ax.axhline(y_centre + init_coords[1], color="orange")
    ax.axvline(x_centre + init_coords[0], color="orange")

    plt.figure()
    coords = np.nonzero(fftshift(io["models"][step]))
    plt.scatter(coords[0], coords[1])
    plt.title("model after λ=0 in step {step}")
    plt.xlim(x_centre - xrange, x_centre + xrange)
    plt.ylim([y_centre - 12, y_centre + yrange])

    ax = plt.gca()
    ax.axhline(y_centre, color="r")
    ax.axvline(x_centre, color="r")

    ax.axhline(y_centre + init_coords[1], color="orange")
    ax.axvline(x_centre + init_coords[0], color="orange")

    plt.figure()
    new_components = get_new_component_coordinates(io, step)
    new_components_scat = np.transpose(new_components.tolist())
    plt.scatter(new_components_scat[0], new_components_scat[1])
    plt.title(f"new components in step {step}")
    plt.xlim(x_centre - xrange, x_centre + xrange)
    plt.ylim([y_centre - 12, y_centre + yrange])

    ax = plt.gca()
    ax.axhline(y_centre, color="r")
    ax.axvline(x_centre, color="r")

    ax.axhline(y_centre + init_coords[1], color="orange")
    ax.axvline(x_centre + init_coords[0], color="orange")

    plt.figure()
    plt.scatter((np.array(deleted_x[step]) + x_centre) % 270, (np.array(deleted_y[step]) + y_centre) % 1024)
    plt.title(f"deleted components in step {step}")
    plt.xlim(x_centre - xrange, x_centre + xrange)
    plt.ylim([y_centre - 12, y_centre + yrange])

    ax = plt.gca()
    ax.axhline(y_centre, color="r")
    ax.axvline(x_centre, color="r")

    ax.axhline(y_centre + init_coords[1], color="orange")
    ax.axvline(x_centre + init_coords[0], color="orange")


def initialize_io_dict():
    """
    Initialize the input / output dictionary for the H-FISTA loop
    """
    io = {}
    io["niters"] = {}
    io["lambdas"] = {}
    io["Ls"] = {}
    io["models"] = {}
    io["models_λ"] = {}
    io["residuals"] = {}
    io["predictions"] = {}
    io["demerits"] = {}
    io["n_comp"] = {}
    io["n_comp_zero"] = {}
    io["substep"] = {}
    io["FISTAs"] = {}
    io["masks"] = {}

    return io


def take_lambda_step(
    data,
    io,
    step,
    ref_lambda=None,
    delay_for_inf=0,
    zero_penalty_coords=np.array([]),
    alpha=-1.0,
    eps=None,
    s=None,
    backtrack=False,
    fix_phase_value=None,
    fix_phase_coords=None,
    threshold_factor=1.0,
    verbose=False,
    very_verbose=False,
    RFI_threshold=5.0,
    RFI_window=21,
    clean_RFI=True,
    perform_ht_pre_debias=False,
    FFT_workers=2,
):
    """
    Take a step at a fixed λ. This includes FISTA with approved components, hard thresholding, updating list of approved components,
    and re-running FISTA + hard-thresholding without further updates to the support.

    Args:
      data: the data to fit
      io: the io dictionary for storing results and auxiliary variables
      step: the current step
      ref_lambda: the regularization parameter trialed in the current step
      delay_for_inf: int. Defaults to 0. Max delay for which non-infinite penalty applies. If negative, the penalty will be quadratically decreasing towards the value of λ at zero delay.
      a: float. Defaults to None. The value of a in the quadratic penalty. If None, the value of a will be set to the value of λ at zero delay.
      zero_penalty_coords: the coordinates of the approved components for which the penalty is fixed at zero (i.e., no regularization)
      alpha: the initial value of the step size. Used only if the returned Lipschitz constant is negative
      eps: the tolerance for the FISTA stopping criterion.
      s: the initial value of the step size TODO
      backtrack: bool. Defaults to False. If True, the step size in FISTA will be determined by backtracking.
      fix_phase_value: float. Fix the phase of the component specified by fix_phase_coords at the given value.
      fix_phase_coords: a tuple of two ints. This component of the model will have a fixed phase to prevent the phase from rotating during the fit.
      threshold_factor: float. Scale the threshold by this factor when applying the hard threshold.
      verbose: bool. Defaults to False. Controls the verbosity of the code
      very_verbose: bool. Defaults to False. Controls the verbosity of the code
      RFI_threshold: threshold (in units of MAD) for RFI median zapping. Defaults to 5
      RFI_window: window size for RFI median zapping. Defaults to 21 channels
      clean_RFI: bool. Defaults to True. If True, the RFI mask is updated using median zapping after every FISTA run

    Returns:
      The return value is the support of the model.
      The model itself and any auxiliary variables are stored in the io dictionary.
    """

    log.setLevel(logger.DEBUGV if very_verbose else logging.INFO)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.debug(f"Starting a λ step with {len(zero_penalty_coords)} approved components")

    # use the model and mask from the previous step as the initial values
    _resid = Residual(data, io["models"][step - 1], ref_lambda, io["masks"][step - 1], workers=FFT_workers)

    if backtrack and not s:
        s = _resid.get_Lipschitz_constant_grad()

    fit, demerit_out, n_comp_out, n_comp_zero_out, L = fista(
        io["models"][step - 1],
        _resid,
        io["niters"][step],
        ref_lambda,
        alpha=alpha,
        delay_for_inf=delay_for_inf,
        zero_penalty_coords=zero_penalty_coords,
        eps=eps,
        s=s,
        backtrack=backtrack,
        fix_phase_value=fix_phase_value,
        fix_phase_coords=fix_phase_coords,
        verbose=verbose,
        very_verbose=very_verbose,
    )
    io["models_λ"][step] = fit

    if perform_ht_pre_debias:
        fit, count_zeroed, _, deleted = hard_threshold(fit, np.transpose(np.nonzero(fit)), threshold_factor, L, ref_lambda)  # type: ignore

    # generate a new mask and update the current mask
    if clean_RFI:
        mask = generate_narrowband_RFI_mask(_resid.residual, 0, RFI_threshold, RFI_window)  # 0 is the time axis index
        io["masks"][step] = update_RFI_mask(io["masks"][step - 1], mask)
    else:
        io["masks"][step] = io["masks"][step - 1]

    # approve all components which survived
    support = np.transpose(np.nonzero(fit))  # type: ignore
    log.debug(f"Updated approved components count to {len(support)}")

    fit_refined = fit

    substep = 0

    prev_deleted = []

    while True:
        substep += 1
        _resid = Residual(data, fit_refined, None, io["masks"][step], workers=FFT_workers)  # type: ignore
        fit_refined, demerit, n_comp, n_comp_zero, L = fista(
            fit_refined,
            _resid,
            io["niters"][step],
            ref_lambda,
            alpha=alpha,
            delay_for_inf=delay_for_inf,
            # zero_penalty_coords=zero_penalty_coords,
            fix_support=support,
            eps=eps,
            s=s,
            backtrack=backtrack,
            fix_phase_value=fix_phase_value,
            fix_phase_coords=fix_phase_coords,
            verbose=verbose,
            very_verbose=very_verbose,
        )
        if clean_RFI:
            # 0 is the time axis index
            mask = generate_narrowband_RFI_mask(_resid.residual, 0, RFI_threshold, RFI_window)
            io["masks"][step] = update_RFI_mask(io["masks"][step], mask)

        demerit_out = np.append(demerit_out, demerit)
        n_comp_out = np.append(n_comp_out, n_comp)
        n_comp_zero_out = np.append(n_comp_zero_out, n_comp_zero)

        fit_refined, count_zeroed, support, deleted = hard_threshold(
            fit_refined, np.transpose(np.nonzero(fit_refined)), threshold_factor, L, ref_lambda, coordinate_list=support  # type: ignore
        )

        if count_zeroed == 0:
            log.info(
                f"No components hard-thresholded in substep {substep} with threshold {threshold_factor * ref_lambda/L}"
            )
            break

        # this shouldn't happen, but just in case, attempt to detect an infinite loop
        if deleted == prev_deleted:
            log.info("Stuck in a loop of deleting same components, interrupting")
            break
        prev_deleted = deleted

        log.debug(f"hard-thresholded {count_zeroed} components in substep {substep}")
        log.debugv(f"In substep {substep} we have {len(support)} approved components")  # type: ignore

    if not clean_RFI:
        io["masks"][step] = io["masks"][step - 1]

    log.info(
        f"RFI mask now includes {(1-np.count_nonzero(io['masks'][step])/np.prod(data.shape))*100:.2f} per cent of data"
    )

    io["FISTAs"][step] = substep + 1
    io["models"][step] = fit_refined
    io["demerits"][step] = demerit_out
    io["n_comp"][step] = n_comp_out
    io["n_comp_zero"][step] = n_comp_zero_out
    io["predictions"][step] = _resid.prediction
    io["residuals"][step] = _resid.residual
    io["Ls"][step] = L
    return support


def spy_waveform_progressive(
    models,
    zoom=None,
    xzoom=20,
    yzoom=20,
    y_bottom_pad=2,
    per_column_legend=25,
    first_step=None,
    max_step=None,
    cmap=get_cmap("viridis"),
):
    x = np.array([])
    y = np.array([])
    classes = np.array([])

    if first_step is None:
        first_step = 0
    elif first_step not in models.keys():
        raise ValueError(f"first_step {first_step} is not a valid choice")

    if max_step is not None:
        if max_step not in models.keys():
            raise ValueError(f"max_step {max_step} is not a valid choice")
    else:
        steps = sorted(models.keys())
        max_step = steps[-1]

    color_norm = max_step - first_step

    prev = np.nonzero(fftshift(models[first_step]))
    x = np.append(x, prev[0])
    y = np.append(y, prev[1])
    classes = np.append(classes, [1] * len(prev[0]))

    plt.scatter(x, y, s=[35] * len(x), label=f"step 1: {len(x)} comp.", color=cmap(1 / max_step))

    for step in range(first_step, max_step + 1):
        if step in models.keys():
            model = models[step]
            coords_list = np.nonzero(fftshift(model))
            new_coords = set_diff2d(np.transpose(coords_list), np.transpose(prev))  # type: ignore
            prev = coords_list
            scat = np.transpose(new_coords.tolist())

            if len(scat) > 0:
                plt.scatter(
                    scat[0],
                    scat[1],
                    s=[35] * len(scat[0]),
                    label=f"step {step}: {len(scat[0])} new",
                    color=cmap(step / color_norm),
                )
        else:
            log.warn(f"step {step} not among the available iterations")

    sizes = np.array([35.0] * len(classes))

    ax = plt.gca()
    ax.get_xaxis().set_major_locator(MaxNLocator(integer=True, steps=[2]))
    ax.get_yaxis().set_major_locator(MaxNLocator(integer=True, steps=[2]))

    if zoom:
        xzoom = zoom
        yzoom = zoom
    if "step" in locals():
        x_dim, y_dim = models[step].shape
        x_centre = int(x_dim / 2)
        y_centre = int(y_dim / 2)
        plt.xlim([x_centre - xzoom, x_centre + xzoom])
        plt.ylim([y_centre - y_bottom_pad, y_centre + yzoom])

        if per_column_legend > 0:
            plt.legend(ncol=max(int(np.max(steps) / per_column_legend), 1))


def get_initial_lambda(desired_components: int, max_negative_delay: int, resid: Residual) -> np.float64:
    """
    Get initial λ for H-FISTA algorithm.

    This is based on the number of non-zero components in the first step.

    Args:
        desired_components: number of components to be present in the model during the first FISTA step
        max_negative_delay: maximum negative delay to be allowed in the model
        resid: residual object used at the start of H-FISTA

    Returns:
        initial λ for H-FISTA algorithm
    """

    # get the gradient and roll it so that the allowed negative delays are at the start

    gradient = resid.get_derivative(None)
    rolled_gradient = np.roll(gradient, np.abs(max_negative_delay), axis=1)

    # slice the parts of the gradient which we are allowing, taking the absolute value and flatten

    abs_flat_grad_part = np.abs(
        rolled_gradient[:, : int(gradient.shape[1] / 2) + np.abs(max_negative_delay)]
    ).flatten()

    # identify the (n+1)-th largest value in the flattened gradient
    # offset by one because we everything equal to λ will be zeroed by the thresholding operator

    λ_from_target = np.partition(
        abs_flat_grad_part,
        -desired_components - 1,
    )[-desired_components - 1]

    return λ_from_target
