from lib import Residual
import numpy as np
import pytest


class Model2componentTestCaseData(Residual):
    """
    Generate a 2-component model: one component in the origin, one off axis, both real valued.
    The start point is a single component field with only a single component at the origin

    const_component is the real amplitude of the component at the origin
    exp_component is the real amplitude of the off-axis component

    N, M - dimensions of the data space
    n, m frequencies of the non-origin component
    """

    def __init__(self, l1_penalty, N=1024, M=1024, const_component=2.0, exp_component=1.0, n=16, m=32):
        self.N = N
        self.M = M
        self.n = n
        self.m = m
        wavefield = np.zeros((N, M)) + np.zeros((N, M)) * 1.0j
        self.wavefield_1st_init_val = const_component + 0.0 * 1j
        self.wavefield_2nd_init_val = exp_component + 0.0 * 1j
        # multiply by N*M below due to how the FFTs are defined in scipy
        wavefield[0, 0] = self.wavefield_1st_init_val * N * M
        wavefield[n, m] = self.wavefield_2nd_init_val * N * M
        # Make data uniform with one component at the origin, which is what we would do in a real case.
        # The amplitude is determined to be the average of the model
        data = np.ones_like(wavefield) * (const_component * const_component + exp_component * exp_component)

        mask = np.ones_like(data)

        super().__init__(data, wavefield, l1_penalty, mask)


@pytest.fixture
def resid_fixture():
    _lambda = 1e-5
    test_data = Model2componentTestCaseData(_lambda).prediction

    N = 1024
    M = 1024
    K = N * M

    A = 2.0
    B = 1.0
    C = A * A + B * B
    n = 16
    m = 32

    h_init = np.zeros((N, M)) + np.zeros((N, M)) * 1.0j
    h_init[0, 0] = np.sqrt(C) * K + 0.0 * 1j

    mask = np.ones_like(test_data)
    resid = Residual(test_data, h_init, _lambda, mask)

    return A, B, C, K, N, M, n, m, h_init, resid


@pytest.fixture
def array_fixture():
    """
    A fixture which provides a small (4 by 4) sparse (4 non-zero elements) complex array.
    Two elements have a small amplitude (<0.125), and two a large amplitude (>0.125)
    The fixture also provides ndarrays of the small and large element coordinates
    """
    array = np.zeros((4, 4)) + 0.0j
    array[0, 0] = 3.0 + 1.0j
    array[1, 2] = 0.1 + 0.001j
    array[3, 3] = -3.9 + 3.2j
    array[2, 0] = -0.04 + 0.02j

    return array, np.array([[1, 2], [2, 0]]), np.array([[0, 0], [3, 3]])
