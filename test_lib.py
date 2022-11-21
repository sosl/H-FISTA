from lib import Residual, extract_part_of_array
from test_helpers import Model2componentTestCaseData, resid_fixture
import numpy as np


def test_data():
    # test based on analytical expectations, see notes_gradient_L
    _lambda = 1e-5
    test_resid = Model2componentTestCaseData(_lambda)

    A = test_resid.wavefield_1st_init_val
    B = test_resid.wavefield_2nd_init_val
    a = test_resid.n
    b = test_resid.m
    N = test_resid.N
    M = test_resid.M

    # [X, Y] = np.meshgrid(2 * np.pi * np.arange(N) / 256, 2 * np.pi * np.arange(M) / 256)
    [X, Y] = np.meshgrid(2 * np.pi * np.arange(N) / (N / 4), 2 * np.pi * np.arange(M) / (M / 4))

    expected_H = A * np.ones((N, M)) + B * np.exp(1j * (b / 4 * X + a / 4 * Y))
    expected_pred = (A * A + B * B) * np.ones((N, M)) + 2 * A * B * np.cos(b / 4 * X + a / 4 * Y)

    assert np.abs(np.max(test_resid.H - expected_H)) < 1e-13 + 1j * 1e-12
    assert np.abs(np.max(test_resid.prediction - expected_pred)) < 1e-12


def test_gradient(resid_fixture):
    _lambda = 1e-5
    test_data = Model2componentTestCaseData(_lambda).prediction
    A, B, C, _, N, M, n, m, h_init, resid = resid_fixture

    expected_grad = np.zeros((N, M)) + np.zeros((N, M)) * 1.0j
    expected_grad[n, m] = -2 * A * B * np.sqrt(C)
    expected_grad[-n, -m] = -2 * A * B * np.sqrt(C)

    deriv = resid.get_derivative(h_init)
    deviation = deriv - expected_grad

    assert (np.abs(resid.get_derivative(h_init) - expected_grad) < 1e-14).all()


def test_Lipschitz_constant(resid_fixture):
    _, _, C, K, _, _, _, _, _, resid = resid_fixture

    L_expected = 4 * C / K
    L = resid.Lipschitz_constant

    assert L_expected == L


def test_extract_part_of_array():
    tmp = np.array([[]])

    N = 1024
    M = 1024

    n = 16
    m = 32

    test_array = np.zeros((N, M)) + np.zeros((N, M)) * 1.0j
    test_array[0, 0] = 21 + 1j * 0
    test_array[n, m] = 0 + 1j * 21

    test_indices = [[0, 0], [16, 32]]

    expected_out = [21 + 1j * 0, 1j * 21]
    out = extract_part_of_array(tmp, test_array, test_indices)

    assert (out == expected_out).all()
