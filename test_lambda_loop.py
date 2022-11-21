from lambda_loop import delete_coordinates_from_array, hard_threshold, set_diff2d, get_initial_lambda
from test_helpers import array_fixture, resid_fixture
import numpy as np


def test_hard_threshold(array_fixture):
    factor = 0.25
    L = 0.4
    _lambda = 0.2

    input, small, large = array_fixture

    support = np.transpose(np.nonzero(input))  # type: ignore

    out, count, _, deleted = hard_threshold(input, support, factor, L, _lambda)
    assert count == 2
    assert len(np.transpose(np.nonzero(out))) == 2  # type: ignore
    for coords in small:
        assert out[coords[0], coords[1]] == 0.0 + 0.0j
    for coords in large:
        assert np.abs(out[coords[0], coords[1]]) > 0.0
    for coords in small:
        assert coords.tolist() in deleted
    for coords in large:
        assert coords.tolist() not in deleted


def test_delete_coordinates_from_array(array_fixture):
    array, small, large = array_fixture
    small_org = np.copy(small)

    # check deleting all
    for coord in small_org:
        small = delete_coordinates_from_array(small, coord[0], coord[1])
    assert len(small) == 0

    # check deleting second / last
    test = np.append(large, small_org, axis=0)
    test = delete_coordinates_from_array(test, small_org[-1][0], small_org[-1][1])
    assert len(test) == len(small_org) + len(large) - 1
    assert (test[len(large)] == small_org[0]).all()

    # check deleting mid
    test = np.append(large, small_org, axis=0)
    test = delete_coordinates_from_array(test, small_org[0][0], small_org[0][1])
    assert len(test) == len(small_org) + len(large) - 1
    assert (test[len(large)] == small_org[1]).all()

    # check deleting first
    small = np.copy(small_org)
    small = delete_coordinates_from_array(small, small[0][0], small[0][1])
    assert len(small) == len(small_org) - 1
    assert (small[0] == small_org[1]).all()

    # check not deleting any
    small = np.copy(small_org)
    for coord in large:
        small = delete_coordinates_from_array(small, coord[0], coord[1])
    assert len(small) == len(small_org)


def test_set_diff2d(array_fixture):
    _, small, large = array_fixture
    diff = set_diff2d(small, large)
    assert (diff.view(np.int64).reshape(small.shape) == small).all()

    diff = set_diff2d(large, small)
    assert (diff.view(np.int64).reshape(large.shape) == large).all()

    all = np.append(small, large, axis=0)
    diff = set_diff2d(all, large)
    assert (diff.view(np.int64).reshape(small.shape) == small).all()

    diff = set_diff2d(all, small)
    assert (diff.view(np.int64).reshape(large.shape) == large).all()


def test_get_initial_lambda(mocker, resid_fixture):
    fake_gradient = np.arange(-1000, 1000).reshape(40, 50)
    mocker.patch("lib.Residual.get_derivative", return_value=fake_gradient)

    assert get_initial_lambda(50, -4, resid_fixture[-1]) == 954
