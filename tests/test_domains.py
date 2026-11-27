import numpy as np
import pytest

from spdepy import domains

VALID_INTERVALS = [
    (0, np.pi),
    (0, 1),
    (-1, 1),
    (1, 5),
    (-6, -2),
    (0, 0.001),
    (np.int64(0), np.float32(0.5)),
]
RELATIVE_POSITIONS = [0, 0.25, 0.5, 0.75, 1]


# Construction and properties
@pytest.mark.parametrize("a, b", VALID_INTERVALS)
def test_interval_construction(a, b):
    interval = domains.Interval(a, b)
    assert interval.a == float(a)
    assert interval.b == float(b)
    assert type(interval.a) is float
    assert type(interval.b) is float


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
def test_interval_length_and_midpoint(a, b):
    interval = domains.Interval(a, b)
    assert interval.length == pytest.approx(float(b) - float(a))
    assert interval.midpoint == pytest.approx((float(a) + float(b)) / 2)


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
def test_interval_dimension(a, b):
    interval = domains.Interval(a, b)
    assert interval.dimension == 1


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
def test_interval_bounds(a, b):
    interval = domains.Interval(a, b)
    assert interval.bounds == (float(a), float(b))


@pytest.mark.parametrize(
    "attribute", ["a", "b", "length", "midpoint", "dimension", "bounds"]
)
def test_read_only_properties(attribute):
    interval = domains.Interval(0, 1)
    with pytest.raises(AttributeError):
        setattr(interval, attribute, 10)


# Membership
@pytest.mark.parametrize("a, b", VALID_INTERVALS)
@pytest.mark.parametrize("r", RELATIVE_POSITIONS)
def test_interval_contains(a, b, r):
    interval = domains.Interval(a, b)
    x = (1 - r) * float(a) + r * float(b)
    assert interval.contains(x) is True


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
def test_interval_excludes_external_points(a, b):
    interval = domains.Interval(a, b)
    assert interval.contains(float(a) - 0.1) is False
    assert interval.contains(float(b) + 0.1) is False


# Coordinate transformations
@pytest.mark.parametrize("a, b", VALID_INTERVALS)
@pytest.mark.parametrize("r", RELATIVE_POSITIONS)
def test_interval_to_reference(a, b, r):
    interval = domains.Interval(a, b)
    x = (1 - r) * float(a) + r * float(b)
    assert interval.to_reference(x) == pytest.approx(r * np.pi)


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
@pytest.mark.parametrize("r", RELATIVE_POSITIONS)
def test_interval_from_reference(a, b, r):
    interval = domains.Interval(a, b)
    x = (1 - r) * float(a) + r * float(b)
    assert interval.from_reference(r * np.pi) == pytest.approx(x)


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
@pytest.mark.parametrize("r", RELATIVE_POSITIONS)
def test_interval_to_and_from_reference(a, b, r):
    interval = domains.Interval(a, b)
    x = (1 - r) * float(a) + r * float(b)
    assert interval.from_reference(interval.to_reference(x)) == pytest.approx(x)


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
@pytest.mark.parametrize("r", RELATIVE_POSITIONS)
def test_interval_from_and_to_reference(a, b, r):
    interval = domains.Interval(a, b)
    ref_x = r * np.pi
    assert interval.to_reference(interval.from_reference(ref_x)) == pytest.approx(ref_x)


# Array support
def test_to_reference_array():
    interval = domains.Interval(2, 6)
    x = np.array([2, 3, 4, 5, 6])
    expected = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    result = interval.to_reference(x)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_from_reference_array():
    interval = domains.Interval(2, 6)
    ref_x = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    expected = np.array([2, 3, 4, 5, 6])
    result = interval.from_reference(ref_x)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
def test_array_round_trip(a, b):
    interval = domains.Interval(a, b)
    x = np.linspace(float(a), float(b), 100)
    result = interval.from_reference(interval.to_reference(x))
    np.testing.assert_allclose(result, x, atol=1e-12)


# Invalid endpoint values
@pytest.mark.parametrize(
    "a, b",
    [
        (1, 0),
        (1, 1),
        (0, np.inf),
        (-np.inf, 0),
        (np.inf, 1),
        (1, -np.inf),
        (0, np.nan),
        (np.nan, 1),
        (-1e308, 1e308),
        (2**53, 2**53 + 1),
    ],
)
def test_reject_invalid_interval(a, b):
    with pytest.raises(ValueError):
        domains.Interval(a, b)


# Invalid endpoint types
@pytest.mark.parametrize(
    "invalid_endpoint",
    [
        True,
        False,
        np.bool_(True),
        "1",
        1 + 0j,
        1 + 2j,
        np.complex128(1),
        None,
        [1],
        (1,),
        np.array([1]),
        np.array(1),
    ],
)
def test_reject_invalid_endpoint_type(invalid_endpoint):
    with pytest.raises(TypeError):
        domains.Interval(invalid_endpoint, 2)
    with pytest.raises(TypeError):
        domains.Interval(0, invalid_endpoint)


# Numerical stability
def test_large_finite_interval():
    interval = domains.Interval(1e308, 1.5e308)
    assert np.isfinite(interval.length)
    assert np.isfinite(interval.midpoint)
    assert interval.midpoint == pytest.approx(1.25e308)


def test_large_interval_to_reference():
    interval = domains.Interval(0, 1e308)
    assert interval.to_reference(1e308) == pytest.approx(np.pi)


# Extrapolation
def test_to_reference_extrapolation():
    interval = domains.Interval(2, 6)
    assert interval.to_reference(0) == pytest.approx(-np.pi / 2)
    assert interval.to_reference(8) == pytest.approx(3 * np.pi / 2)


def test_from_reference_extrapolation():
    interval = domains.Interval(2, 6)
    assert interval.from_reference(-np.pi / 2) == pytest.approx(0)
    assert interval.from_reference(3 * np.pi / 2) == pytest.approx(8)
