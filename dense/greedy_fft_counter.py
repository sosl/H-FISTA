"""FFT counter for lib.Residual (copied from claude/greedy-pursuit greedy/pursuit.py)."""
from lib import Residual


class FFTCounter:
    """Count FFTs done by lib.Residual (one inverse per state change, one forward per gradient)."""

    n = 0

    @classmethod
    def install(cls):
        if getattr(Residual, "_counted", False):
            return
        set_state, set_der = Residual._set_state, Residual.set_derivative

        def _set_state(self, w):
            cls.n += 1
            return set_state(self, w)

        def set_derivative(self):
            cls.n += 1
            return set_der(self)

        Residual._set_state, Residual.set_derivative, Residual._counted = _set_state, set_derivative, True
