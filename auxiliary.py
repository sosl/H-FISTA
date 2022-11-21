from scipy.signal import sosfiltfilt, lfilter, freqz, sosfreqz, medfilt
from scipy.fft import fft, fftshift, fftfreq
from scipy.stats import median_abs_deviation

from matplotlib import pyplot as plt
import numpy as np
from typing import Optional

import pickle
import logger

log = logger.get_logger(__name__)


def write_object(prefix: str, postfix: Optional[str] = None, object={}):
    """Write out a pickled object to a file

    Filename is constructed from pre- and post-fix

    Args:
        prefix (str): Prefix for the output filename
        postfix (Optional[str], optional): Optional postfix for the output filename. Defaults to None.
        object (dict, optional): Object to pickle and write. Defaults to {}.
    """
    if postfix is not None:
        postfix = f"_{postfix}"
    else:
        postfix = ""
    with open(f"{prefix}{postfix}.pkl", "wb") as fh:
        pickle.dump(object, fh)


def free_mem_from_dict(io, keys=None, max_int=None):
    """
    This function is used to reduce memory by deleting entries in a nested dictionary.
    The inner keys are expected to be integers, and the largest value will be retained.

    Arguments:
        keys - optional list or generator keys. If provided, only the nested dictionaries under these keys will be deleted. Otherwise all are affected
        max_int - optional maximum value of an integer key to delete. If not provided, all but largest present will be affected.

    Returns:
        dictionary with all but highest-integer value entries deleted
    """
    if keys is None:
        keys = io.keys()

    if max_int is None:
        max_int = max(io[key].keys())
    for key in keys:
        for i in range(min(io[key].keys()), max_int):
            try:
                del io[key][i]
            except (KeyError, ValueError) as e:
                log.warn("Freeing up memory from dict exception: {e}")
                pass

    return io


def generate_narrowband_RFI_mask(data: np.ndarray, time_axis: int, threshold: float, window=21) -> np.ndarray:
    """Single pass median-filter-based RFI mask generator

    Generate an RFI mask, based on a median filter. This is the same method as used by PSRchive.

    Args:
        data (np.ndarray): Dynamic spectrum to check for narrowband RFI
        time_axis (int): which axis is the time axis
        threshold (float): threshold for the median filter
        window (int, optional): How many channels to use for the median filter. Defaults to 21.

    Raises:
        ValueError: Raise exception if time axis is not 0 or 1

    Returns:
        np.ndarray: The mask, returned as one array of ones and zeroes
    """
    mask = np.ones_like(data)
    time_average = np.average(data, axis=time_axis)
    median_smoothed = medfilt(time_average, window)
    diff = time_average - median_smoothed
    std_dev = median_abs_deviation(diff, scale="normal")  # type: ignore

    RFI_location = np.where(np.abs(diff) > threshold * std_dev)
    if time_axis == 0:
        mask[:, RFI_location] = 0
    elif time_axis == 1:
        mask[RFI_location, :] = 0
    else:
        raise ValueError("time_axis must be 0 or 1")

    return mask


def update_RFI_mask(previous_mask: np.ndarray, new_mask: np.ndarray) -> np.ndarray:
    """Combine two RFI masks

    Returns a mask which masks everything masked in either of the input masks

    Args:
        previous_mask (np.ndarray): First of the input masks
        new_mask (np.ndarray): Second of the input masks

    Returns:
        np.ndarray: Combined mask
    """
    return previous_mask * new_mask


def normalise_by_mean(data: np.ndarray, axis: int) -> np.ndarray:
    """Normalise the data by the mean along axis

    Useful for removing striation from dynamic spectra

    Args:
        data (np.ndarray): Data to normalise
        axis (int): Which axis to average along, normally the frequency axis

    Returns:
        np.ndarray: Data after normalisation
    """
    # return data / np.mean(data, axis=axis)
    mean = np.average(np.transpose(data), axis=axis)
    data = np.divide(np.copy(data), np.reshape(mean, [len(mean), 1]))
    return data
