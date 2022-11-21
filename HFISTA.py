import argparse
import numpy as np

from helpers import get_data
from lib import Residual
from lambda_loop import take_lambda_step, initialize_io_dict, get_initial_lambda
from stopping import check_stopping

from auxiliary import (
    normalise_by_mean,
    generate_narrowband_RFI_mask,
    update_RFI_mask,
    write_object,
    free_mem_from_dict,
)
from densify import get_dense_solution

import time

import logger

log = logger.get_logger(__name__)


def populate_step_zero_io(io, h_init, mask, args):
    step = 0
    io["niters"][step] = 0
    io["masks"][step] = mask

    resid = Residual(data, h_init, None, mask)
    io["residuals"][step] = resid.residual
    io["predictions"][step] = resid.prediction

    io["models"][step] = h_init
    io["demerits"][step] = -1

    if args._lambda is not None:
        io["lambdas"][step] = args._lambda
    else:
        io["lambdas"][step] = get_initial_lambda(args.Nzero, args.max_negative_delay, resid) * args.eta_lambda


def initialise_mask(data, args):
    """
    Initialise the RFI mask for data
    """

    mask = np.ones_like(data)
    if args.RFI_time_init:
        time_mask = generate_narrowband_RFI_mask(
            data, 1, threshold=args.RFI_time_threshold, window=args.RFI_time_window
        )
        mask = update_RFI_mask(mask, time_mask)
    return mask


def initialise_wavefield(data, mask, init_coords):
    """
    Initialise the wavefield with all power in one wave at the origin.

    The power is chosen so that the corresponding dynamic spectrum has same average as masked data
    """
    N, M = data.shape
    h_init = np.zeros((N, M)) + np.zeros((N, M)) * 1.0j
    init_value = np.sqrt(np.mean(data * mask)) * N * M + 0.0j
    h_init[init_coords[0], init_coords[1]] = init_value

    return h_init


def get_lambda_loop_config(args, init_coords=[0, 0]):
    lambda_loop_config = {}
    fix_phase_value = 0.0

    lambda_loop_config = {
        "backtrack": not args.no_backtrack,
        "fix_phase_coords": init_coords,
        "fix_phase_value": 0.0,
        "force_more": False,
        "threshold_factor": args.T_H,
        "verbose": args.verbose,
        "delay_for_inf": -np.abs(args.max_negative_delay),
        "zero_penalty_coords": np.transpose(np.nonzero(h_init)),
        "RFI_threshold": args.RFI_chan_threshold,
        "RFI_window": args.RFI_chan_window,
    }
    return lambda_loop_config


def setup_argparse():
    parser = argparse.ArgumentParser(
        description="Perform phase retrieval with H-FISTA", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    input = parser.add_argument_group("Input")
    input.add_argument("--data", type=str, help="Data file")
    input.add_argument("--time_axis", type=int, default=0, help="Time axis in data, 0-indexed")
    input.add_argument("--striation", help="Remove striation by normalising by mean", action="store_true")

    FISTA_config = parser.add_argument_group("FISTA configuration")
    FISTA_config.add_argument("--Niter", type=int, default=80, help="Number of iterations")
    FISTA_config.add_argument("--no-backtrack", help="Backtrack line search for step size", action="store_true")
    FISTA_config.add_argument("--max_negative_delay", type=int, default=-4, help="Maximum negative delay index")
    FISTA_config.add_argument("--verbose", help="Print verbose messages", action="store_true")

    RFI_config = parser.add_argument_group("RFI masking")
    RFI_config.add_argument("--RFI_chan_window", type=int, default=21, help="Window size for median RFI filtering")
    RFI_config.add_argument(
        "--RFI_chan_threshold", type=float, default=5.0, help="Threshold for median narrowband RFI filtering"
    )
    RFI_config.add_argument(
        "--RFI_time_init",
        action="store_true",
        help="Initialise RFI mask across subintegrations. Useful for masking gaps in data.",
    )
    RFI_config.add_argument("--RFI_time_window", type=int, default=51, help="Time window for mask initialisation")
    RFI_config.add_argument(
        "--RFI_time_threshold", type=float, default=5.0, help="Time window for mask initialisation"
    )

    HFISTA_config = parser.add_argument_group("HFISTA configuration")
    HFISTA_config.add_argument(
        "--Nzero", help="Desired number of components in the wavefield after first FISTA step", type=int, default=60
    )
    HFISTA_config.add_argument(
        "--eta_lambda", type=float, default=1.15, help="Scaling factor for λ. Should be >1 so that λ decreases."
    )
    HFISTA_config.add_argument(
        "--hard_threshold_factor", dest="T_H", metavar="T_H", default=1.0, type=float, help="Hard threshold factor"
    )
    HFISTA_config.add_argument(
        "--lambda",
        dest="_lambda",  # lambda is a keyword
        type=float,
        default=None,
        help="Initial value of λ. If provided, Nzero will be ingored.",
    )
    HFISTA_config.add_argument("--spatial_stopping", default=True, help="Use the spatial stopping criterion")
    HFISTA_config.add_argument(
        "--spatial_threshold", default=1e-10, help="Threshold for the spatial stopping criterion"
    )
    HFISTA_config.add_argument("--sparsity_stopping", default=True, help="Use the sparsity stopping criterion")
    HFISTA_config.add_argument(
        "--sparsity_threshold", default=0.03, help="Threshold for the sparsity stopping criterion"
    )
    HFISTA_config.add_argument("--max_iterations", type=int, default=100, help="Maximum number of H-FISTA iterations")

    output_config = parser.add_argument_group("Output")
    output_config.add_argument("--prefix", type=str, help="Prefix for output files", default="HFISTA_out")
    output_config.add_argument(
        "--sparse", type=bool, default=True, help="Save sparse wavefield in addition to the dense wavefield"
    )
    output_config.add_argument(
        "--full", type=bool, default=False, help="Save full output, useful for resuming, debugging, etc."
    )
    output_config.add_argument(
        "--partial-frequency",
        dest="outsteps",
        metavar="OUTSTEPS",
        type=int,
        default=0,
        help="Save io dictionary every OUTSTEPS steps",
    )
    return parser


if __name__ == "__main__":
    start = time.time()
    # hardcoded defaults
    init_coords = [0, 0]

    # parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # initialise everything needed for first FISTA
    if args.time_axis == 0:
        data = get_data(args.data)
    else:
        data = get_data(args.data).T
    if args.striation:
        # 0 because we ensure time axis is 0 when loading, even if originally it was 1
        data = normalise_by_mean(data, axis=0)

    mask = initialise_mask(data, args)
    h_init = initialise_wavefield(data, mask, init_coords)
    io = initialize_io_dict()
    populate_step_zero_io(io, h_init, mask, args)

    lambda_loop_config = get_lambda_loop_config(args)
    chosen_step = -1

    for step in range(1, args.max_iterations + 1):

        io["lambdas"][step] = io["lambdas"][step - 1] / args.eta_lambda
        lambda_loop_config["ref_lambda"] = io["lambdas"][step]
        io["niters"][step] = args.Niter

        log.info(f"### λ iteration {step} with λ={lambda_loop_config['ref_lambda']} and scaling {args.eta_lambda} ###")

        lambda_loop_config["zero_penalty_coords"] = take_lambda_step(data, io, step, **lambda_loop_config)

        if args.outsteps > 0:
            if args.sparse and step % args.outsteps == 0:
                write_object(args.prefix, postfix=f"part_at_{step}", object=io)
                free_mem_from_dict(io)

        if step > 1:
            if check_stopping(
                io,
                step,
                check_sparse=args.sparsity_stopping,
                sparsity_threshold=args.sparsity_threshold,
                check_doppler_distribution=args.spatial_stopping,
                spatial_threshold=args.spatial_threshold,
            ):
                chosen_step = step - 1
                log.info(f"Stopping criterion triggered in step {step}")
                break

    end_run = time.time()

    if args.full:
        write_object(args.prefix, postfix=None, object=io)
    if chosen_step > 1:
        if args.sparse:
            write_object(args.prefix, postfix="sparse", object=io["models"][chosen_step])
        dense_wavefield = get_dense_solution(io["models"][chosen_step], data, io["masks"][chosen_step])
        write_object(args.prefix, postfix="dense", object=dense_wavefield)
    else:
        log.warn("No sparse solution was chosen, will not provide a dense solution.")
        if not args.full:
            log.warn("Writing out the full io dicitonary for inspection")
            write_object(args.prefix, postfix=None, object=io)

    end_unload = time.time()

    log.debug(f"elapsed minutes: {(end_run - start)/60:.2f}")
