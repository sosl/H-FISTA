import numpy as np
from scipy.stats import kstest
from helpers import get_new_component_coordinates

import logging, logger

log = logger.get_logger(__name__)


def check_sparsity(wavefield: np.ndarray, threshold=0.03) -> bool:
    """Check if the number of non-zero components in an array exceeds the sparsity threshold

    Args:
        wavefield (np.ndarray): array to check for number of non-zero components.
        threshold (float, optional): Return True if fraction of non-zero components is above. Defaults to 0.03.

    Returns:
        bool: True if the threshold is exceeded
    """
    log.debugv(
        f"{np.count_nonzero(wavefield)}/{np.prod(wavefield.shape)}  = {np.count_nonzero(wavefield) / np.prod(wavefield.shape)}"
    )
    return np.count_nonzero(wavefield) / np.prod(wavefield.shape) > threshold


def check_spatial(
    new_components: np.ndarray,
    max_doppler: int,
    min_doppler=0,
    doppler_axis=0,
    p_threshold=1e-10,
    min_new_components=100,
) -> bool:
    """Check the spatial stopping criterion

    Check if the spatial (in doppler-shift) distribution of the new components is
    starting to be consistent with a uniform distribution. We check with a KS test
    with a p-value threshold.

    Note that the threshold is much lower than what would normally be accepted as
    we are trying to catch the moment when the distribution is becoming to be uniform

    The criterion is only applied if the number of new components is greater than min_new_components


    Args:
        new_components (np.ndarray): An array with a list of new components.
        max_doppler (float): Max doppler index to consider
        min_doppler (int, optional): Min doppler index to consider. Defaults to 0.
        doppler_axis (int, optional): Which axis is the doppler axis. Defaults to 0.
        p_threshold (_type_, optional): P-threshold for the KS test. Defaults to 1e-10.
        min_new_components (int, optional): Minimum count of new components above which this test applies. Defaults to 100.

    Returns:
        bool: True if the p_value exceeded the threshold
    """
    new_components = np.array(new_components.tolist()).transpose()
    # Sometimes there will be no new components (in particular, in noise free cases), or too few:
    if len(new_components) == 0 or min_new_components > len(new_components[0]):
        return False
    doppler_scaled = (new_components[doppler_axis] - min_doppler) / (max_doppler - min_doppler)
    doppler_ks = kstest(doppler_scaled, "uniform")
    log.debug(f"new components count {len(new_components[0]):.3g}")
    log.debug(f"doppler ks {doppler_ks.pvalue:.3g}")

    return doppler_ks.pvalue > p_threshold


def check_stopping(
    io: dict,
    step: int,
    doppler_axis=0,
    check_sparse=True,
    sparsity_threshold=0.03,
    check_doppler_distribution=True,
    spatial_threshold=1e-10,
):
    """Check if the stopping criteria are met

    Can check both spatial and sparsity criteria

    Args:
        io (dict): Input/output dictionary as used by lambda loop
        step (int): Step at which to check the stopping criteria
        doppler_axis (int, optional): Which axis is the doppler axi. Defaults to 0.
        check_sparse (bool, optional): Apply the sparsity criterion. Defaults to True.
        sparsity_threshold (float, optional): Sparsity threshold. Defaults to 0.03.
        check_doppler_distribution (bool, optional): Apply the spatial criterion. Defaults to True.
        spatial_threshold (_type_, optional): Spatial test p-value threshold. Defaults to 1e-10.

    Returns:
        _type_: True if either stopping criterion is triggered
    """
    new_components = get_new_component_coordinates(io, step)
    wavefield = io["models"][step]
    if check_sparse:
        log.debug("Checking sparse")
        if check_sparsity(wavefield, threshold=sparsity_threshold):
            log.info("Sparsity triggered")
            return True
    if check_doppler_distribution:
        log.debug("Checking spatial")
        if check_spatial(
            new_components, wavefield.shape[doppler_axis], doppler_axis=doppler_axis, p_threshold=spatial_threshold
        ):
            log.info("Spatial triggered")
            return True
    log.debug("No stopping criterion fulfilled")
    return False
