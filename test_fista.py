from numpy.lib.function_base import angle
from lib import Residual
from test_helpers import Model2componentTestCaseData, resid_fixture
import numpy as np

from fista import (
    backtrack_B3,
    complex_phase_fix,
    fista,
    complex_prox_l1,
)


def test_first_step(resid_fixture):
    # test based on analytical expectations, see notes_gradient_L
    _lambda = 1e-5
    test_data = Model2componentTestCaseData(_lambda).prediction

    A, B, C, K, _, _, n, m, h_init, resid = resid_fixture

    alpha = 1e5
    niter = 1

    indices_of_interest = [[0, 0], [n, m]]

    x, _, _, _, _ = fista(h_init, resid, niter, _lambda, control_indices=indices_of_interest)

    vanilla_0_0 = np.sqrt(C) * K
    vanilla_n_m = A * B * K / np.sqrt(C) / 2

    expected_0_0 = (np.abs(vanilla_0_0) - _lambda / resid.get_Lipschitz_constant_grad()) * np.exp(
        1j * np.angle(vanilla_0_0)
    )
    expected_n_m = (np.abs(vanilla_n_m) - _lambda / resid.get_Lipschitz_constant_grad()) * np.exp(
        1j * np.angle(vanilla_n_m)
    )

    assert np.abs((resid.wavefield[0, 0] - expected_0_0) / expected_0_0) < 1e-9
    assert np.abs((resid.wavefield[n, m] - expected_n_m) / expected_n_m) < 1e-9
    assert np.count_nonzero(resid.wavefield) == 2


def test_complex_prox_l1():
    eps = 1e-13

    rng = np.random.default_rng()
    x, y = rng.standard_normal(2)
    c = x + y * 1j

    _lambda = rng.uniform(0, 10)

    c_vec = np.array([c / np.abs(c) * _lambda / 2.0, c / np.abs(c) * _lambda * 2])

    L = 1.0

    cprox = complex_prox_l1(c_vec, _lambda, L)

    assert np.real(cprox[0]) == 0
    assert np.imag(cprox[0]) == 0

    assert np.abs(np.angle(cprox[1]) - np.angle(c)) < eps
    assert np.abs(np.abs(cprox[1] - c_vec[1]) - _lambda) < eps


def test_complex_phase_fix():
    N = 1024
    M = 1024

    test_input = np.zeros((N, M)) + np.zeros((N, M)) * 1.0j
    n = 16
    m = 32

    test_input[0, 0] = 2.0 - 2.0j

    expected_origin = 2.0 * np.sqrt(2)

    test_output = complex_phase_fix(test_input, [[0, 0]])

    assert test_output[0, 0] == expected_origin


def test_backtrack_B3(resid_fixture):
    resid = resid_fixture[-1]
    true_L = resid.Lipschitz_constant
    eta = 1.1

    L_backtrack, _ = backtrack_B3(resid.get_func_val, resid.get_derivative, 1 / true_L, eta, resid.wavefield)
    assert L_backtrack >= true_L

    test_L = true_L / 2.0
    L_backtrack, _ = backtrack_B3(resid.get_func_val, resid.get_derivative, 1 / test_L, eta, resid.wavefield)
    assert L_backtrack >= 7.5e-6
