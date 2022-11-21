import pickle
import astropy.io.fits as fits
import logger
import numpy as np
from astropy.time import Time
import astropy.units as u

from scipy.fft import fftshift

log = logger.get_logger(__name__)


class ParsingError(ValueError):
    """Generic error when parsing data files"""

    pass


def get_data(fname: str, byte_swap=True) -> np.ndarray:
    """Load a dynamic spectrum from a file

    Supports plain ascii (from psrflux), a FITS file, or a pickle.

    Args:
        fname (str): Name of the file with the dynamic spectrum
        byte_swap (bool, optional): _description_. Defaults to True. Swap bytes when loading a FITS file

    Raises:
        ParsingError: Couldn't load the dynamic spectrum

    Returns:
        np.ndarray: dynamic spectrum array
    """
    try:
        data, _, _, _, _ = read_psrflux(fname)
        return data
    except UnicodeDecodeError:
        log.warning(f"Loading {fname} as a psrflux ASCII file failed. Trying to load as a FITS file.")

    try:
        with fits.open(fname) as ds:
            data_be = ds[0].data  # type: ignore
            # FITS uses big endian so let's swap the byte ordering
            if byte_swap:
                data = data_be.newbyteorder().byteswap()
            else:
                data = data_be
            return data
    except FileNotFoundError:
        log.error(f"File {fname} not found")
        raise
    except OSError:
        log.warning(f"Loading {fname} as a FITS file failed. Trying to load as a pickle")

    try:
        with open(fname, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        log.warning(f"Loading {fname} as a pickle file failed with exception.")

    raise ParsingError("Could not load dynamic spectrum")


def read_psrflux(fname: str):
    """Read in a psrflux ASCII dynamic spectrum

    Contributed by Rob Main, adjusted by Stefan Oslowski.

    Args:
        fname (str): Name of the file with the dynamic spectrum

    Returns:
        tuple: Returns a tuple of dynspec, dynspec_err, T, F, source
    """
    source = None

    try:
        with open(fname, "r") as file:
            for line in file:
                if line.startswith("#"):
                    headline = str.strip(line[1:])
                    if str.split(headline)[0] == "MJD0:":
                        # MJD of start of obs
                        mjd = float(str.split(headline)[1])
                    if str.split(headline)[0] == "source:":
                        # MJD of start of obs
                        source = str.split(headline)[1]
                    if str.split(headline)[0] == "telescope:":
                        # MJD of start of obs
                        telescope = str.split(headline)[1]
    except FileNotFoundError:
        log.error(f"File {fname} not found")
        raise

    if source is None:
        source = ""

    data = np.loadtxt(fname)
    dt = int(np.max(data[:, 0]) + 1)
    df = int(np.max(data[:, 1]) + 1)

    t = data[::df, 2] * u.min
    F = data[:df, 3] * u.MHz
    dynspec = (data[:, 4]).reshape(dt, df)
    dynspec_err = (data[:, 5]).reshape(dt, df)
    T = Time(float(mjd), format="mjd") + t

    return dynspec, dynspec_err, T, F, source


def set_diff2d(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return difference A-B of sets of 2D arrays, e.g., coordinates

    Args:
        A (np.ndarray): First coordinate array
        B (np.ndarray): Second coordinate array

    Returns:
        np.ndarray: Set difference of the two input coordinate arrays
    """
    # set diff A-B
    nrows, ncols = A.shape
    dtype = {"names": ["x", "y"], "formats": ncols * [A.dtype]}
    return np.setdiff1d(A.copy().view(dtype), B.copy().view(dtype))


def get_new_component_coordinates(io: dict, step: int) -> np.ndarray:
    """Return an array of tuples with new coordinates

    Find coordinates of non-zero components present in the wavefield in the
    input/output dictionary io at step and not present at step - 1

    Args:
        io (dict): input/output dictionary as used by the lambda loop
        step (int): step at which we want to find new components

    Raises:
        ValueError: Raise an error if either step or step-1 is not present in the io dictionary

    Returns:
        np.ndarray: Array of the coordinates of new components
    """
    if np.all([i in io["models"].keys() for i in [step - 1, step]]):
        return set_diff2d(
            np.transpose(np.nonzero(fftshift(io["models"][step]))),  # type: ignore
            np.transpose(np.nonzero(fftshift(io["models"][step - 1]))),  # type: ignore
        )
    else:
        raise ValueError(f"Wavefield for step {step} or {step-1} not present in the dictionary")
