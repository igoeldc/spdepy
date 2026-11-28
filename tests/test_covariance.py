import numpy as np
import pytest

from spdepy.basis import SineBasis
from spdepy.covariance import SpectralCovariance
from spdepy.domains import Interval

VALID_INTERVALS = [
    (0, np.pi),
    (0, 1),
    (-1, 1),
    (1, 5),
    (-6, -2),
    (0, 0.001),
]

INVALID_MODE_TYPES = [
    True,
    False,
    np.bool_(True),
    1.5,
    2.0,
    "2",
    1 + 0j,
    None,
]

INVALID_MODE_VALUES = [0, -1, -10]

INVALID_EIGENVALUE_TYPES = [
    True,
    False,
    np.bool_(True),
    "1",
    1 + 0j,
    1 + 2j,
    [1],
    np.array([1]),
    None,
]

INVALID_EIGENVALUE_VALUES = [
    -1,
    -0.5,
    -1e-15,
    np.nan,
    np.inf,
    -np.inf,
]

ATOL = 1e-10
RTOL = 1e-10


@pytest.fixture(params=VALID_INTERVALS)
def basis(request):
    a, b = request.param
    return SineBasis(Interval(a, b))


@pytest.fixture
def covariance(basis):
    return SpectralCovariance(basis, lambda k: 1 / k**2)


def test_construction(basis):
    operator = SpectralCovariance(basis, lambda k: 1 / k**2)

    assert operator.basis is basis


@pytest.mark.parametrize(
    "basis",
    [None, 1, "basis", Interval(0, 1), (0, 1)],
)
def test_invalid_basis(basis):
    with pytest.raises(TypeError):
        SpectralCovariance(basis, lambda k: 1.0)


@pytest.mark.parametrize(
    "eigenvalues",
    [None, 1, 1.0, "function", [1, 2, 3], np.array([1, 2])],
)
def test_non_callable_eigenvalues(basis, eigenvalues):
    with pytest.raises(TypeError):
        SpectralCovariance(basis, eigenvalues)


def test_basis_read_only(covariance):
    with pytest.raises(AttributeError):
        covariance.basis = SineBasis(Interval(0, 2))


def test_lazy_evaluation(basis):
    calls = []

    def eigenvalues(k):
        calls.append(k)
        return 1.0

    operator = SpectralCovariance(basis, eigenvalues)

    assert calls == []

    operator.eigenvalue(3)

    assert calls == [3]


def test_lazy_validation(basis):
    operator = SpectralCovariance(basis, lambda k: -1.0)

    with pytest.raises(ValueError):
        operator.eigenvalue(1)


def test_partial_eigenvalue_validation(basis):
    operator = SpectralCovariance(
        basis,
        lambda k: 1.0 if k <= 3 else -1.0,
    )

    np.testing.assert_array_equal(
        operator.eigenvalues(3),
        np.ones(3),
    )

    with pytest.raises(ValueError):
        operator.eigenvalues(4)


@pytest.mark.parametrize(
    "k, expected",
    [
        (1, 1.0),
        (2, 0.25),
        (3, 1 / 9),
        (5, 0.04),
        (10, 0.01),
    ],
)
def test_eigenvalue_known_values(covariance, k, expected):
    assert covariance.eigenvalue(k) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.0,
        -0.0,
        1,
        0.5,
        1e-15,
        np.int64(2),
        np.float32(0.5),
        np.float64(0.25),
    ],
)
def test_valid_eigenvalue_types(basis, value):
    operator = SpectralCovariance(basis, lambda k: value)
    result = operator.eigenvalue(1)

    assert type(result) is float
    assert result == pytest.approx(float(value))


def test_eigenvalue_numpy_integer(covariance):
    assert covariance.eigenvalue(np.int64(3)) == pytest.approx(1 / 9)


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
def test_eigenvalues_shape(covariance, num_modes):
    result = covariance.eigenvalues(num_modes)

    assert isinstance(result, np.ndarray)
    assert result.shape == (num_modes,)
    assert result.dtype == np.float64


def test_eigenvalues_known_values(covariance):
    np.testing.assert_allclose(
        covariance.eigenvalues(5),
        [1, 1 / 4, 1 / 9, 1 / 16, 1 / 25],
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10, 100])
def test_eigenvalues_consistency(covariance, num_modes):
    expected = np.array([covariance.eigenvalue(k) for k in range(1, num_modes + 1)])

    np.testing.assert_allclose(
        covariance.eigenvalues(num_modes),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_eigenvalues_numpy_integer(covariance):
    np.testing.assert_array_equal(
        covariance.eigenvalues(np.int64(5)),
        covariance.eigenvalues(5),
    )


def test_scalar_only_callable(basis):
    def eigenvalues(k):
        assert type(k) is int
        return 1 / k**2

    operator = SpectralCovariance(basis, eigenvalues)

    np.testing.assert_allclose(
        operator.eigenvalues(3),
        [1, 1 / 4, 1 / 9],
        atol=ATOL,
        rtol=RTOL,
    )


def test_callable_mode_order(basis):
    calls = []

    def eigenvalues(k):
        calls.append(k)
        return float(k)

    operator = SpectralCovariance(basis, eigenvalues)

    operator.eigenvalues(5)

    assert calls == [1, 2, 3, 4, 5]


def test_nonmonotonic_eigenvalues(basis):
    values = [1.0, 3.0, 0.0, 2.0, 0.5]
    operator = SpectralCovariance(basis, lambda k: values[k - 1])

    np.testing.assert_array_equal(
        operator.eigenvalues(5),
        values,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10, 100])
def test_white_noise_covariance(basis, num_modes):
    operator = SpectralCovariance(basis, lambda k: 1.0)

    np.testing.assert_array_equal(
        operator.eigenvalues(num_modes),
        np.ones(num_modes),
    )

    np.testing.assert_array_equal(
        operator.matrix(num_modes),
        np.eye(num_modes),
    )

    np.testing.assert_array_equal(
        operator.sqrt_matrix(num_modes),
        np.eye(num_modes),
    )

    assert operator.trace(num_modes) == float(num_modes)


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_zero_covariance(basis, num_modes):
    operator = SpectralCovariance(basis, lambda k: 0.0)

    np.testing.assert_array_equal(
        operator.eigenvalues(num_modes),
        np.zeros(num_modes),
    )

    np.testing.assert_array_equal(
        operator.matrix(num_modes),
        np.zeros((num_modes, num_modes)),
    )

    np.testing.assert_array_equal(
        operator.sqrt_matrix(num_modes),
        np.zeros((num_modes, num_modes)),
    )

    np.testing.assert_array_equal(
        operator.apply(np.ones(num_modes)),
        np.zeros(num_modes),
    )

    assert operator.trace(num_modes) == 0.0


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_finite_rank_covariance(basis, rank):
    operator = SpectralCovariance(
        basis,
        lambda k: 1.0 if k <= rank else 0.0,
    )

    num_modes = 10
    expected = np.zeros(num_modes)
    expected[:rank] = 1.0

    np.testing.assert_array_equal(
        operator.eigenvalues(num_modes),
        expected,
    )

    assert np.linalg.matrix_rank(operator.matrix(num_modes)) == rank

    assert operator.trace(num_modes) == float(rank)


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
def test_matrix_shape(covariance, num_modes):
    result = covariance.matrix(num_modes)

    assert isinstance(result, np.ndarray)
    assert result.shape == (num_modes, num_modes)
    assert result.dtype == np.float64


def test_matrix_known_values(covariance):
    expected = np.array(
        [
            [1, 0, 0],
            [0, 1 / 4, 0],
            [0, 0, 1 / 9],
        ]
    )

    np.testing.assert_allclose(
        covariance.matrix(3),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_matrix_diagonal(covariance, num_modes):
    matrix = covariance.matrix(num_modes)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)

    np.testing.assert_array_equal(
        off_diagonal,
        np.zeros((num_modes, num_modes)),
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_matrix_diagonal_entries(covariance, num_modes):
    np.testing.assert_array_equal(
        np.diag(covariance.matrix(num_modes)),
        covariance.eigenvalues(num_modes),
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_matrix_symmetric(covariance, num_modes):
    matrix = covariance.matrix(num_modes)

    np.testing.assert_array_equal(matrix, matrix.T)


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_matrix_positive_semidefinite(covariance, num_modes):
    eigenvalues = np.linalg.eigvalsh(covariance.matrix(num_modes))

    assert np.all(eigenvalues >= 0)


def test_singular_matrix_positive_semidefinite(basis):
    operator = SpectralCovariance(
        basis,
        lambda k: 1 / k**2 if k <= 3 else 0.0,
    )

    eigenvalues = np.linalg.eigvalsh(operator.matrix(5))

    assert np.all(eigenvalues >= 0)
    assert np.count_nonzero(eigenvalues == 0) == 2


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
def test_sqrt_matrix_shape(covariance, num_modes):
    result = covariance.sqrt_matrix(num_modes)

    assert isinstance(result, np.ndarray)
    assert result.shape == (num_modes, num_modes)
    assert result.dtype == np.float64


def test_sqrt_matrix_known_values(covariance):
    expected = np.array(
        [
            [1, 0, 0],
            [0, 1 / 2, 0],
            [0, 0, 1 / 3],
        ]
    )

    np.testing.assert_allclose(
        covariance.sqrt_matrix(3),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_sqrt_matrix_consistency(covariance, num_modes):
    root = covariance.sqrt_matrix(num_modes)

    np.testing.assert_allclose(
        root @ root.T,
        covariance.matrix(num_modes),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_sqrt_matrix_symmetric(covariance, num_modes):
    root = covariance.sqrt_matrix(num_modes)

    np.testing.assert_array_equal(root, root.T)


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_sqrt_matrix_positive_semidefinite(covariance, num_modes):
    eigenvalues = np.linalg.eigvalsh(covariance.sqrt_matrix(num_modes))

    assert np.all(eigenvalues >= 0)


def test_singular_sqrt_matrix(basis):
    operator = SpectralCovariance(
        basis,
        lambda k: 1 / k**2 if k <= 3 else 0.0,
    )

    root = operator.sqrt_matrix(5)

    np.testing.assert_allclose(
        root @ root.T,
        operator.matrix(5),
        atol=ATOL,
        rtol=RTOL,
    )

    assert np.linalg.matrix_rank(root) == 3


@pytest.mark.parametrize(
    "coeffs",
    [
        [1],
        [1, 2, 3],
        (1, 2, 3),
        np.array([1, 2, 3]),
        np.array([1, 2, 3], dtype=np.float32),
        np.array([1, 2, 3], dtype=np.int64),
    ],
)
def test_apply_shape(covariance, coeffs):
    result = covariance.apply(coeffs)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(coeffs),)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "coeffs, expected",
    [
        ([1], [1]),
        ([1, 2, 3], [1, 1 / 2, 1 / 3]),
        ([2, -3, 5], [2, -3 / 4, 5 / 9]),
        ([0, 1, 0], [0, 1 / 4, 0]),
        ([0, 0, 1], [0, 0, 1 / 9]),
        ([0, 0, 0], [0, 0, 0]),
    ],
)
def test_apply_known_values(covariance, coeffs, expected):
    np.testing.assert_allclose(
        covariance.apply(coeffs),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10, 100])
def test_apply_zero_coefficients(covariance, num_modes):
    np.testing.assert_array_equal(
        covariance.apply(np.zeros(num_modes)),
        np.zeros(num_modes),
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10, 100])
def test_apply_matrix_consistency(covariance, num_modes):
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal(num_modes)

    np.testing.assert_allclose(
        covariance.apply(coeffs),
        covariance.matrix(num_modes) @ coeffs,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_apply_eigenfunction(covariance, k):
    coeffs = np.zeros(10)
    coeffs[k - 1] = 1.0

    np.testing.assert_allclose(
        covariance.apply(coeffs),
        covariance.eigenvalue(k) * coeffs,
        atol=ATOL,
        rtol=RTOL,
    )


def test_apply_white_noise_covariance(basis):
    operator = SpectralCovariance(basis, lambda k: 1.0)
    coeffs = np.array([1.0, -2.0, 3.0, 0.5])

    np.testing.assert_array_equal(
        operator.apply(coeffs),
        coeffs,
    )


def test_apply_finite_rank_covariance(basis):
    operator = SpectralCovariance(
        basis,
        lambda k: 1.0 if k <= 2 else 0.0,
    )

    np.testing.assert_array_equal(
        operator.apply([1, 2, 3, 4, 5]),
        [1, 2, 0, 0, 0],
    )


@pytest.mark.parametrize("alpha", [-3, -1, 0, 0.5, 1, 3])
def test_apply_homogeneity(covariance, alpha):
    coeffs = np.array([1.5, -2, 0, 3, -0.5])

    np.testing.assert_allclose(
        covariance.apply(alpha * coeffs),
        alpha * covariance.apply(coeffs),
        atol=ATOL,
        rtol=RTOL,
    )


def test_apply_additivity(covariance):
    u = np.array([1, 2, -3, 0, 5])
    v = np.array([-2, 0, 1, 4, -1])

    np.testing.assert_allclose(
        covariance.apply(u + v),
        covariance.apply(u) + covariance.apply(v),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "alpha, beta",
    [
        (1, 1),
        (2, -3),
        (0, 1),
        (1, 0),
        (0.5, -0.25),
        (-2, -5),
    ],
)
def test_apply_linearity(covariance, alpha, beta):
    u = np.array([1, 2, -3, 0, 5])
    v = np.array([-2, 0, 1, 4, -1])

    np.testing.assert_allclose(
        covariance.apply(alpha * u + beta * v),
        alpha * covariance.apply(u) + beta * covariance.apply(v),
        atol=ATOL,
        rtol=RTOL,
    )


def test_apply_does_not_modify_input(covariance):
    coeffs = np.array([1.0, -2.0, 3.0])
    original = coeffs.copy()

    covariance.apply(coeffs)

    np.testing.assert_array_equal(coeffs, original)


def test_apply_positive_quadratic_form(covariance):
    coeffs = np.array([1.0, -2.0, 3.0, 0.5])

    assert coeffs @ covariance.apply(coeffs) > 0


def test_apply_zero_quadratic_form(basis):
    operator = SpectralCovariance(
        basis,
        lambda k: 1.0 if k <= 2 else 0.0,
    )

    coeffs = np.array([0.0, 0.0, 1.0, -2.0])

    assert coeffs @ operator.apply(coeffs) == 0.0


def test_apply_self_adjointness(covariance):
    u = np.array([1.0, -2.0, 3.0, 0.5])
    v = np.array([-1.0, 4.0, 2.0, -3.0])

    assert u @ covariance.apply(v) == pytest.approx(covariance.apply(u) @ v)


def test_apply_and_reconstruct(covariance):
    a, b = covariance.basis.domain.bounds
    length = covariance.basis.domain.length
    xs = np.linspace(a, b, 100)

    coeffs = np.array([2.0, -3.0, 5.0])

    result = covariance.basis.reconstruct(covariance.apply(coeffs), xs)

    t = np.pi * (xs - a) / length

    expected = np.sqrt(2 / length) * (
        2 * np.sin(t) - 3 / 4 * np.sin(2 * t) + 5 / 9 * np.sin(3 * t)
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_project_apply_reconstruct(covariance):
    a, b = covariance.basis.domain.bounds
    length = covariance.basis.domain.length

    def f(x):
        t = np.pi * (x - a) / length
        return 2 * np.sin(t) - 3 * np.sin(3 * t)

    xs = np.linspace(a, b, 100)
    coeffs = covariance.basis.project(f, 5)

    result = covariance.basis.reconstruct(covariance.apply(coeffs), xs)

    t = np.pi * (xs - a) / length

    expected = 2 * np.sin(t) - 1 / 3 * np.sin(3 * t)

    np.testing.assert_allclose(
        result,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "num_modes, expected",
    [
        (1, 1.0),
        (2, 1 + 1 / 4),
        (3, 1 + 1 / 4 + 1 / 9),
        (4, 1 + 1 / 4 + 1 / 9 + 1 / 16),
    ],
)
def test_trace_known_values(covariance, num_modes, expected):
    assert covariance.trace(num_modes) == pytest.approx(expected)


def test_trace_output_type(covariance):
    assert type(covariance.trace(5)) is float


@pytest.mark.parametrize("num_modes", [1, 5, 10, 100])
def test_trace_matrix_consistency(covariance, num_modes):
    assert covariance.trace(num_modes) == pytest.approx(
        np.trace(covariance.matrix(num_modes))
    )


def test_trace_nondecreasing(covariance):
    traces = np.array([covariance.trace(n) for n in range(1, 21)])

    assert np.all(np.diff(traces) >= 0)


def test_trace_finite_rank_saturation(basis):
    operator = SpectralCovariance(
        basis,
        lambda k: 1.0 if k <= 3 else 0.0,
    )

    assert operator.trace(3) == 3.0
    assert operator.trace(10) == 3.0
    assert operator.trace(100) == 3.0


def test_trace_summable_covariance(covariance):
    expected = np.pi**2 / 6
    result = covariance.trace(10_000)

    assert result < expected
    assert result == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    "method",
    [
        "eigenvalue",
        "eigenvalues",
        "matrix",
        "sqrt_matrix",
        "trace",
    ],
)
@pytest.mark.parametrize("value", INVALID_MODE_TYPES)
def test_invalid_mode_type(method, value):
    operator = SpectralCovariance(
        SineBasis(Interval(0, 1)),
        lambda k: 1.0,
    )

    with pytest.raises(TypeError):
        getattr(operator, method)(value)


@pytest.mark.parametrize(
    "method",
    [
        "eigenvalue",
        "eigenvalues",
        "matrix",
        "sqrt_matrix",
        "trace",
    ],
)
@pytest.mark.parametrize("value", INVALID_MODE_VALUES)
def test_invalid_mode_value(method, value):
    operator = SpectralCovariance(
        SineBasis(Interval(0, 1)),
        lambda k: 1.0,
    )

    with pytest.raises(ValueError):
        getattr(operator, method)(value)


@pytest.mark.parametrize("value", INVALID_EIGENVALUE_TYPES)
def test_invalid_eigenvalue_type(value):
    operator = SpectralCovariance(
        SineBasis(Interval(0, 1)),
        lambda k: value,
    )

    with pytest.raises(TypeError):
        operator.eigenvalue(1)


@pytest.mark.parametrize("value", INVALID_EIGENVALUE_VALUES)
def test_invalid_eigenvalue_value(value):
    operator = SpectralCovariance(
        SineBasis(Interval(0, 1)),
        lambda k: value,
    )

    with pytest.raises(ValueError):
        operator.eigenvalue(1)


@pytest.mark.parametrize(
    "method",
    ["eigenvalues", "matrix", "sqrt_matrix", "trace"],
)
def test_invalid_eigenvalue_propagation(method):
    operator = SpectralCovariance(
        SineBasis(Interval(0, 1)),
        lambda k: 1.0 if k < 3 else -1.0,
    )

    with pytest.raises(ValueError):
        getattr(operator, method)(3)


def test_apply_invalid_eigenvalue():
    operator = SpectralCovariance(
        SineBasis(Interval(0, 1)),
        lambda k: 1.0 if k < 3 else -1.0,
    )

    with pytest.raises(ValueError):
        operator.apply([1, 2, 3])


@pytest.mark.parametrize(
    "coeffs",
    [
        [True],
        [1, False],
        [np.bool_(True)],
        ["1"],
        [1, "2"],
        [1 + 0j],
        [1 + 2j],
        [None],
        np.array([True, False]),
        np.array([1, 2], dtype=object),
    ],
)
def test_invalid_coefficient_type(coeffs):
    operator = SpectralCovariance(
        SineBasis(Interval(0, 1)),
        lambda k: 1.0,
    )

    with pytest.raises(TypeError):
        operator.apply(coeffs)


@pytest.mark.parametrize(
    "coeffs",
    [
        [],
        np.array([]),
        1.0,
        np.array(1.0),
        [[1, 2]],
        np.zeros((2, 2, 2)),
        [np.nan],
        [np.inf],
        [-np.inf],
        [1, np.nan],
        [1, np.inf],
    ],
)
def test_invalid_coefficient_value(coeffs):
    operator = SpectralCovariance(
        SineBasis(Interval(0, 1)),
        lambda k: 1.0,
    )

    with pytest.raises(ValueError):
        operator.apply(coeffs)
