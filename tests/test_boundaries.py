import numpy as np
import pytest

from spdepy import boundaries

VALID_VALUES = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (-1, 1),
    (1, -1),
    (-2, -5),
    (0.5, 1.5),
    (np.int64(2), np.float32(0.5)),
]

INVALID_TYPES = [
    True,
    False,
    np.bool_(True),
    "1",
    1 + 2j,
    1 + 0j,
    np.array([1]),
    [1],
    None,
]

NONFINITE_VALUES = [
    np.nan,
    np.inf,
    -np.inf,
]


def test_default_construction():
    bc = boundaries.Dirichlet()

    assert bc.values == (0.0, 0.0)
    assert bc.is_homogeneous is True


@pytest.mark.parametrize("left, right", VALID_VALUES)
def test_dirichlet_construction(left, right):
    bc = boundaries.Dirichlet(left, right)

    assert bc.values == (left, right)


@pytest.mark.parametrize("left, right", VALID_VALUES)
def test_normalized_types(left, right):
    bc = boundaries.Dirichlet(left, right)

    assert type(bc.values[0]) is float
    assert type(bc.values[1]) is float


@pytest.mark.parametrize(
    "left, right",
    [(0, 0), (0.0, 0.0), (-0.0, 0.0)],
)
def test_homogeneous(left, right):
    bc = boundaries.Dirichlet(left, right)

    assert bc.is_homogeneous is True


@pytest.mark.parametrize(
    "left, right",
    [
        (0, 1),
        (1, 0),
        (1, 1),
        (-1, -1),
        (-1, 1),
        (1e-15, 0),
        (0, 1e-15),
    ],
)
def test_nonhomogeneous(left, right):
    bc = boundaries.Dirichlet(left, right)

    assert bc.is_homogeneous is False


@pytest.mark.parametrize("value", INVALID_TYPES)
def test_reject_invalid_left_type(value):
    pytest.raises(TypeError, boundaries.Dirichlet, value, 0)


@pytest.mark.parametrize("value", INVALID_TYPES)
def test_reject_invalid_right_type(value):
    pytest.raises(TypeError, boundaries.Dirichlet, 0, value)


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_reject_nonfinite_left_value(value):
    pytest.raises(ValueError, boundaries.Dirichlet, value, 0)


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_reject_nonfinite_right_value(value):
    pytest.raises(ValueError, boundaries.Dirichlet, 0, value)


@pytest.mark.parametrize("left, right", VALID_VALUES)
def test_individual_properties(left, right):
    bc = boundaries.Dirichlet(left, right)

    assert bc.left == left
    assert bc.right == right


@pytest.mark.parametrize("attribute", ["left", "right"])
def test_endpoints_read_only(attribute):
    bc = boundaries.Dirichlet()

    with pytest.raises(AttributeError):
        setattr(bc, attribute, 1.0)


def test_values_read_only():
    bc = boundaries.Dirichlet()

    with pytest.raises(AttributeError):
        bc.values = (1.0, 2.0)


def test_is_homogeneous_read_only():
    bc = boundaries.Dirichlet()

    with pytest.raises(AttributeError):
        bc.is_homogeneous = False
