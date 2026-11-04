import numpy as np
from scipy.fft import fft, ifft, fftfreq, ifftshift, fftshift, fftn, ifftn
from scipy.linalg import norm
from typing import Optional

import logger

log = logger.setup_logger(is_debug=True)
log = logger.get_logger(__name__)


def extract_part_of_array(part: np.ndarray, full: np.ndarray, indices: list):
    for index in indices:
        part = np.append(part, full[index[0]][index[1]])
    return part


class Residual:
    """
    To update the prediction, derivative, and residuals, either call get_func_val with the new wavefield, or set wavefield and call initialise

    By default, use up to two workers to perform the FFTs

    TODO this class is actually poorly adjusted to use with FISTA. E.g., because we need to calculate the derivative at a different point than the value
    """

    def __init__(
        self,
        data: np.ndarray,
        wavefield: np.ndarray,
        l1_penalty: float,
        RFI_mask: np.ndarray,
        l2_penalty=None,
        workers=2,
    ):
        self.workers = workers

        self.data = data
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty

        self.sampling_time = 1.0
        self.sampling_freq = 1.0

        self.Lipschitz_constant = -1
        self.RFI_mask = RFI_mask

        self._set_state(wavefield)
        self.set_Lipschitz_constant_grad()

    # ------------------------------------------------------------------------------
    # State handling
    #
    # The prediction and residual are always computed for the current wavefield (one
    # inverse FFT). The derivative needs one more (forward) FFT and is only evaluated
    # when requested, then cached until the wavefield changes. Evaluating the demerit
    # therefore costs a single FFT, rather than three as before.
    # ------------------------------------------------------------------------------
    def _set_state(self, wavefield: np.ndarray):
        self.wavefield = wavefield
        self.H = ifftn(wavefield, workers=self.workers)
        self.set_prediction()
        self.set_residual()
        self._derivative = None

    @property
    def derivative(self):
        if self._derivative is None:
            self.set_derivative()
        return self._derivative

    @derivative.setter
    def derivative(self, value):
        self._derivative = value

    def set_Lipschitz_constant_grad(self):
        # calculated from definition, and using the fact that we will also be constraining solution to positive values of tau
        prod = np.prod(self.data.shape)
        self.Lipschitz_constant = 4.0 * np.mean(self.data * self.RFI_mask) / prod

    def get_Lipschitz_constant_grad(self):
        return self.Lipschitz_constant

    def set_prediction(self):
        self.prediction = np.real(self.H * np.conj(self.H))

    def set_residual(self):
        # Not using the mask here, only when calculating the residual squared
        self.residual = self.prediction - self.data

    def get_residual(self):
        return self.residual

    def get_func_val(self, wavefield: np.ndarray):
        """
        Calculate the sum of squared residuals for the provided wavefield.
        If wavefield is None, return the sum of squared residuals for the current wavefield
        """
        if wavefield is not None:
            self._set_state(wavefield)

        sum_resid_squared = np.sum(np.square(self.residual * self.RFI_mask))
        return sum_resid_squared

    def get_full_demerit(self):
        sum_resid_squared = self.get_func_val(self.wavefield)
        if self.l1_penalty:
            sum_resid_squared += self.l1_penalty * np.sum(np.abs(self.wavefield))
        if self.l2_penalty is not None:
            sum_resid_squared += self.l2_penalty * np.sum(self.wavefield * np.conj(self.wavefield))
        return sum_resid_squared

    def set_derivative(self):
        prod = np.prod(self.data.shape)
        derivative = 2 / prod * fftn(self.residual * self.RFI_mask * self.H, workers=self.workers)
        if self.l2_penalty is not None:
            derivative += self.l2_penalty * self.wavefield
        self._derivative = derivative

    def get_derivative(self, wavefield: Optional[np.ndarray] = None):
        """
        Calculate the new values for the provided wavefield.
        If wavefield is None, return values for the current wavefield
        """
        if wavefield is not None:
            self._set_state(wavefield)
        return self.derivative

    def get_func_val_and_derivative(self, wavefield: Optional[np.ndarray] = None):
        """Demerit and derivative at wavefield, sharing the inverse FFT (2 FFTs in total)."""
        if wavefield is not None:
            self._set_state(wavefield)
        return self.get_func_val(None), self.derivative

    def initialise(self):
        self._set_state(self.wavefield)
