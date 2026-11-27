import numpy as np
import pytest

from spdepy.basis import SineBasis
from spdepy.boundaries import Dirichlet
from spdepy.domains import Interval
from spdepy.operators import Laplacian

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

ATOL = 1e-10
RTOL = 1e-10


@pytest.fixture(params=VALID_INTERVALS)
def laplacian(request):
    a, b = request.param
    return Laplacian(Interval(a, b), Dirichlet())


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
def test_construction(a, b):
    domain = Interval(a, b)
    bc = Dirichlet()
    operator = Laplacian(domain, bc)

    assert operator.domain is domain
    assert operator.boundary_conditions is bc
    assert isinstance(operator.eigenbasis, SineBasis)
    assert operator.eigenbasis.domain is domain


@pytest.mark.parametrize("domain", [None, 1, "interval", (0, 1)])
def test_invalid_domain(domain):
    with pytest.raises(TypeError):
        Laplacian(domain, Dirichlet())


@pytest.mark.parametrize("bc", [None, 0, "Dirichlet", (0, 0)])
def test_invalid_boundary_conditions(bc):
    with pytest.raises(TypeError):
        Laplacian(Interval(0, 1), bc)


@pytest.mark.parametrize(
    "left, right",
    [
        (0, 1),
        (1, 0),
        (1, 1),
        (-1, 1),
        (-2, -5),
        (1e-15, 0),
        (0, -1e-15),
    ],
)
def test_nonhomogeneous_boundary_conditions(left, right):
    with pytest.raises(ValueError):
        Laplacian(Interval(0, 1), Dirichlet(left, right))


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("domain", Interval(0, 2)),
        ("boundary_conditions", Dirichlet()),
        ("eigenbasis", SineBasis(Interval(0, 2))),
    ],
)
def test_properties_read_only(laplacian, attribute, value):
    with pytest.raises(AttributeError):
        setattr(laplacian, attribute, value)


def test_eigenbasis_orthonormal(laplacian):
    assert laplacian.eigenbasis.is_orthonormal


def test_eigenbasis_identity(laplacian):
    assert laplacian.eigenbasis is laplacian.eigenbasis


@pytest.mark.parametrize(
    "k, expected",
    [
        (1, -1.0),
        (2, -4.0),
        (3, -9.0),
        (5, -25.0),
        (10, -100.0),
    ],
)
def test_eigenvalue_reference_interval(k, expected):
    operator = Laplacian(Interval(0, np.pi), Dirichlet())

    assert operator.eigenvalue(k) == pytest.approx(expected)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_eigenvalue_unit_interval(k):
    operator = Laplacian(Interval(0, 1), Dirichlet())

    assert operator.eigenvalue(k) == pytest.approx(-(k**2) * np.pi**2)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_eigenvalue_arbitrary_interval(laplacian, k):
    length = laplacian.domain.length
    expected = -(k**2) * np.pi**2 / length**2

    assert laplacian.eigenvalue(k) == pytest.approx(expected)


def test_eigenvalue_output_type(laplacian):
    assert type(laplacian.eigenvalue(1)) is float


def test_eigenvalue_numpy_integer(laplacian):
    assert laplacian.eigenvalue(np.int64(3)) == pytest.approx(laplacian.eigenvalue(3))


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
def test_eigenvalues_shape(laplacian, num_modes):
    result = laplacian.eigenvalues(num_modes)

    assert isinstance(result, np.ndarray)
    assert result.shape == (num_modes,)
    assert result.dtype == np.float64


def test_eigenvalues_reference_interval():
    operator = Laplacian(Interval(0, np.pi), Dirichlet())

    np.testing.assert_allclose(
        operator.eigenvalues(5),
        [-1, -4, -9, -16, -25],
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10, 100])
def test_eigenvalues_consistency(laplacian, num_modes):
    expected = np.array([laplacian.eigenvalue(k) for k in range(1, num_modes + 1)])

    np.testing.assert_allclose(
        laplacian.eigenvalues(num_modes),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_eigenvalues_numpy_integer(laplacian):
    np.testing.assert_allclose(
        laplacian.eigenvalues(np.int64(5)),
        laplacian.eigenvalues(5),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10, 100])
def test_eigenvalues_negative(laplacian, num_modes):
    assert np.all(laplacian.eigenvalues(num_modes) < 0)


@pytest.mark.parametrize("num_modes", [2, 5, 10, 100])
def test_eigenvalues_strictly_decreasing(laplacian, num_modes):
    assert np.all(np.diff(laplacian.eigenvalues(num_modes)) < 0)


@pytest.mark.parametrize("num_modes", [5, 10, 100])
def test_eigenvalues_quadratic_growth(laplacian, num_modes):
    ks = np.arange(1, num_modes + 1)
    expected = laplacian.eigenvalue(1) * ks**2

    np.testing.assert_allclose(
        laplacian.eigenvalues(num_modes),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("shift", [-10, -2, 1, 5, 100])
def test_translation_invariance(shift):
    original = Laplacian(Interval(0, 2), Dirichlet())
    translated = Laplacian(Interval(shift, shift + 2), Dirichlet())

    np.testing.assert_allclose(
        original.eigenvalues(10),
        translated.eigenvalues(10),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("scale", [0.1, 0.5, 2, 5, 10])
def test_domain_scaling(scale):
    original = Laplacian(Interval(0, 1), Dirichlet())
    scaled = Laplacian(Interval(0, scale), Dirichlet())

    np.testing.assert_allclose(
        scaled.eigenvalues(10),
        original.eigenvalues(10) / scale**2,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
def test_matrix_shape(laplacian, num_modes):
    result = laplacian.matrix(num_modes)

    assert isinstance(result, np.ndarray)
    assert result.shape == (num_modes, num_modes)
    assert result.dtype == np.float64


def test_matrix_reference_interval():
    operator = Laplacian(Interval(0, np.pi), Dirichlet())

    expected = np.array(
        [
            [-1, 0, 0],
            [0, -4, 0],
            [0, 0, -9],
        ]
    )

    np.testing.assert_allclose(
        operator.matrix(3),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_matrix_diagonal(laplacian, num_modes):
    matrix = laplacian.matrix(num_modes)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)

    np.testing.assert_array_equal(
        off_diagonal,
        np.zeros((num_modes, num_modes)),
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_matrix_diagonal_entries(laplacian, num_modes):
    np.testing.assert_allclose(
        np.diag(laplacian.matrix(num_modes)),
        laplacian.eigenvalues(num_modes),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_matrix_symmetric(laplacian, num_modes):
    matrix = laplacian.matrix(num_modes)

    np.testing.assert_array_equal(matrix, matrix.T)


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_matrix_negative_definite(laplacian, num_modes):
    eigenvalues = np.linalg.eigvalsh(laplacian.matrix(num_modes))

    assert np.all(eigenvalues < 0)


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
def test_apply_shape(laplacian, coeffs):
    result = laplacian.apply(coeffs)

    assert isinstance(result, np.ndarray)
    assert result.shape == (len(coeffs),)
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "coeffs, expected",
    [
        ([1], [-1]),
        ([1, 2, 3], [-1, -8, -27]),
        ([2, -3, 5], [-2, 12, -45]),
        ([0, 1, 0], [0, -4, 0]),
        ([0, 0, 1], [0, 0, -9]),
        ([0, 0, 0], [0, 0, 0]),
    ],
)
def test_apply_known_values(coeffs, expected):
    operator = Laplacian(Interval(0, np.pi), Dirichlet())

    np.testing.assert_allclose(
        operator.apply(coeffs),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10, 100])
def test_apply_zero_coefficients(laplacian, num_modes):
    np.testing.assert_array_equal(
        laplacian.apply(np.zeros(num_modes)),
        np.zeros(num_modes),
    )


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_apply_eigenfunction(laplacian, k):
    coeffs = np.zeros(10)
    coeffs[k - 1] = 1

    np.testing.assert_allclose(
        laplacian.apply(coeffs),
        laplacian.eigenvalue(k) * coeffs,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10, 100])
def test_apply_matrix_consistency(laplacian, num_modes):
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal(num_modes)

    np.testing.assert_allclose(
        laplacian.apply(coeffs),
        laplacian.matrix(num_modes) @ coeffs,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("alpha", [-3, -1, 0, 0.5, 1, 3])
def test_apply_homogeneity(laplacian, alpha):
    coeffs = np.array([1.5, -2, 0, 3, -0.5])

    np.testing.assert_allclose(
        laplacian.apply(alpha * coeffs),
        alpha * laplacian.apply(coeffs),
        atol=ATOL,
        rtol=RTOL,
    )


def test_apply_additivity(laplacian):
    u = np.array([1, 2, -3, 0, 5])
    v = np.array([-2, 0, 1, 4, -1])

    np.testing.assert_allclose(
        laplacian.apply(u + v),
        laplacian.apply(u) + laplacian.apply(v),
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
def test_apply_linearity(laplacian, alpha, beta):
    u = np.array([1, 2, -3, 0, 5])
    v = np.array([-2, 0, 1, 4, -1])

    np.testing.assert_allclose(
        laplacian.apply(alpha * u + beta * v),
        alpha * laplacian.apply(u) + beta * laplacian.apply(v),
        atol=ATOL,
        rtol=RTOL,
    )


def test_apply_does_not_modify_input(laplacian):
    coeffs = np.array([1.0, -2.0, 3.0])
    original = coeffs.copy()

    laplacian.apply(coeffs)

    np.testing.assert_array_equal(coeffs, original)


def test_apply_negative_quadratic_form(laplacian):
    coeffs = np.array([1.0, -2.0, 3.0, 0.5])

    assert coeffs @ laplacian.apply(coeffs) < 0


def test_apply_self_adjointness(laplacian):
    u = np.array([1.0, -2.0, 3.0, 0.5])
    v = np.array([-1.0, 4.0, 2.0, -3.0])

    assert u @ laplacian.apply(v) == pytest.approx(laplacian.apply(u) @ v)


def test_apply_and_reconstruct(laplacian):
    a, b = laplacian.domain.bounds
    length = laplacian.domain.length
    xs = np.linspace(a, b, 100)

    coeffs = np.array([2.0, -3.0, 5.0])
    result = laplacian.eigenbasis.reconstruct(laplacian.apply(coeffs), xs)

    t = np.pi * (xs - a) / length

    expected = (
        np.sqrt(2 / length)
        * (np.pi / length) ** 2
        * (-2 * np.sin(t) + 12 * np.sin(2 * t) - 45 * np.sin(3 * t))
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol=ATOL * max(1.0, np.max(np.abs(expected))),
        rtol=RTOL,
    )


def test_project_apply_reconstruct(laplacian):
    a, b = laplacian.domain.bounds
    length = laplacian.domain.length

    def f(x):
        t = np.pi * (x - a) / length
        return 2 * np.sin(t) - 3 * np.sin(3 * t)

    xs = np.linspace(a, b, 100)
    coeffs = laplacian.eigenbasis.project(f, 5)
    result = laplacian.eigenbasis.reconstruct(laplacian.apply(coeffs), xs)

    t = np.pi * (xs - a) / length

    expected = (np.pi / length) ** 2 * (-2 * np.sin(t) + 27 * np.sin(3 * t))

    np.testing.assert_allclose(
        result,
        expected,
        atol=ATOL * max(1.0, np.max(np.abs(expected))),
        rtol=RTOL,
    )


@pytest.mark.parametrize("method", ["eigenvalue", "eigenvalues", "matrix"])
@pytest.mark.parametrize("value", INVALID_MODE_TYPES)
def test_invalid_mode_type(method, value):
    operator = Laplacian(Interval(0, 1), Dirichlet())

    with pytest.raises(TypeError):
        getattr(operator, method)(value)


@pytest.mark.parametrize("method", ["eigenvalue", "eigenvalues", "matrix"])
@pytest.mark.parametrize("value", INVALID_MODE_VALUES)
def test_invalid_mode_value(method, value):
    operator = Laplacian(Interval(0, 1), Dirichlet())

    with pytest.raises(ValueError):
        getattr(operator, method)(value)


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
    operator = Laplacian(Interval(0, 1), Dirichlet())

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
    operator = Laplacian(Interval(0, 1), Dirichlet())

    with pytest.raises(ValueError):
        operator.apply(coeffs)
