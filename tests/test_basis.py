import itertools

import numpy as np
import pytest
from scipy.integrate import quad

from spdepy.basis import SineBasis
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

ATOL = 1e-10
RTOL = 1e-10


@pytest.fixture(params=VALID_INTERVALS)
def basis(request):
    a, b = request.param
    return SineBasis(Interval(a, b))


@pytest.mark.parametrize("a, b", VALID_INTERVALS)
def test_construction(a, b):
    domain = Interval(a, b)
    basis = SineBasis(domain)

    assert basis.domain is domain


@pytest.mark.parametrize("domain", [None, 1, "interval", (0, 1)])
def test_invalid_domain(domain):
    with pytest.raises(TypeError):
        SineBasis(domain)


def test_is_orthonormal(basis):
    assert basis.is_orthonormal is True


def test_domain_read_only(basis):
    with pytest.raises(AttributeError):
        basis.domain = Interval(0, 1)


def test_is_orthonormal_read_only(basis):
    with pytest.raises(AttributeError):
        basis.is_orthonormal = False


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_evaluate_boundaries(basis, k):
    a, b = basis.domain.bounds

    np.testing.assert_allclose(
        basis.evaluate(k, [a, b]),
        [0.0, 0.0],
        atol=ATOL,
        rtol=RTOL,
    )


def test_evaluate_known_values(basis):
    a, b = basis.domain.bounds
    L = b - a
    normalization = np.sqrt(2 / L)

    xs = [a, a + L / 4, a + L / 2, a + 3 * L / 4, b]

    expected = normalization * np.array(
        [
            0,
            np.sqrt(2) / 2,
            1,
            np.sqrt(2) / 2,
            0,
        ]
    )

    np.testing.assert_allclose(
        basis.evaluate(1, xs),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_evaluate_second_mode(basis):
    a, b = basis.domain.bounds
    L = b - a

    xs = [a, a + L / 4, a + L / 2, a + 3 * L / 4, b]
    expected = np.sqrt(2 / L) * np.array([0, 1, 0, -1, 0])

    np.testing.assert_allclose(
        basis.evaluate(2, xs),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_evaluate_scalar_output(basis):
    result = basis.evaluate(1, basis.domain.midpoint)

    assert isinstance(result, float)


def test_evaluate_zero_dimensional_array(basis):
    x = np.array(basis.domain.midpoint)

    assert isinstance(basis.evaluate(1, x), float)


def test_evaluate_single_element_array(basis):
    result = basis.evaluate(1, [basis.domain.midpoint])

    assert isinstance(result, np.ndarray)
    assert result.shape == (1,)
    assert result.dtype == np.float64


@pytest.mark.parametrize("shape", [(5,), (2, 3), (2, 3, 4)])
def test_evaluate_shape_preservation(basis, shape):
    a, b = basis.domain.bounds
    xs = np.linspace(a, b, np.prod(shape)).reshape(shape)

    result = basis.evaluate(1, xs)

    assert result.shape == shape
    assert result.dtype == np.float64


def test_evaluate_empty_array(basis):
    result = basis.evaluate(1, [])

    assert result.shape == (0,)


def test_evaluate_numpy_integer_mode(basis):
    x = basis.domain.midpoint

    assert basis.evaluate(np.int64(1), x) == pytest.approx(basis.evaluate(1, x))


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
def test_normalization(basis, k):
    a, b = basis.domain.bounds

    integral, _ = quad(
        lambda x: basis.evaluate(k, x) ** 2,
        a,
        b,
    )

    assert integral == pytest.approx(1.0, abs=ATOL, rel=RTOL)


@pytest.mark.parametrize(
    "j, k",
    [(1, 2), (1, 3), (2, 3), (2, 5), (5, 10)],
)
def test_orthogonality(basis, j, k):
    a, b = basis.domain.bounds

    integral, _ = quad(
        lambda x: basis.evaluate(j, x) * basis.evaluate(k, x),
        a,
        b,
    )

    assert integral == pytest.approx(0.0, abs=ATOL)


def test_gram_matrix(basis):
    a, b = basis.domain.bounds
    num_modes = 5
    gram = np.zeros((num_modes, num_modes))

    for j in range(1, num_modes + 1):
        for k in range(1, num_modes + 1):
            integral, _ = quad(
                lambda x, j, k: basis.evaluate(j, x) * basis.evaluate(k, x),
                a,
                b,
                args=(j, k),
            )
            gram[j - 1, k - 1] = integral

    np.testing.assert_allclose(
        gram,
        np.eye(num_modes),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "num_modes, num_coords",
    [(1, 1), (1, 10), (3, 5), (5, 3), (10, 100)],
)
def test_matrix_shape(basis, num_modes, num_coords):
    a, b = basis.domain.bounds
    xs = np.linspace(a, b, num_coords)

    matrix = basis.matrix(num_modes, xs)

    assert matrix.shape == (num_coords, num_modes)
    assert matrix.dtype == np.float64


def test_matrix_known_values(basis):
    a, b = basis.domain.bounds
    L = b - a

    xs = [a, a + L / 4, a + L / 2, b]

    expected = np.sqrt(2 / L) * np.array(
        [
            [0, 0, 0],
            [np.sqrt(2) / 2, 1, np.sqrt(2) / 2],
            [1, 0, -1],
            [0, 0, 0],
        ]
    )

    np.testing.assert_allclose(
        basis.matrix(3, xs),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_matrix_columns_match_evaluate(basis):
    a, b = basis.domain.bounds
    xs = np.linspace(a, b, 20)
    num_modes = 5

    matrix = basis.matrix(num_modes, xs)

    for k in range(1, num_modes + 1):
        np.testing.assert_allclose(
            matrix[:, k - 1],
            basis.evaluate(k, xs),
            atol=ATOL,
            rtol=RTOL,
        )


def test_matrix_empty_coordinates(basis):
    matrix = basis.matrix(5, [])

    assert matrix.shape == (0, 5)


def test_project_shape(basis):
    coeffs = basis.project(lambda x: 1.0, 5)

    assert coeffs.shape == (5,)
    assert coeffs.dtype == np.float64


def test_project_zero_function(basis):
    coeffs = basis.project(lambda x: 0.0, 5)

    np.testing.assert_allclose(
        coeffs,
        np.zeros(5),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_project_constant_function(basis, num_modes):
    L = basis.domain.length
    k = np.arange(1, num_modes + 1)

    expected = np.sqrt(2 * L) * (1 - (-1) ** k) / (k * np.pi)

    coeffs = basis.project(lambda x: 1.0, num_modes)

    np.testing.assert_allclose(
        coeffs,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_project_basis_function(basis, k):
    num_modes = 5

    expected = np.zeros(num_modes)
    expected[k - 1] = 1.0

    coeffs = basis.project(
        lambda x: basis.evaluate(k, x),
        num_modes,
    )

    np.testing.assert_allclose(
        coeffs,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_project_linear_combination(basis):
    def f(x):
        return (
            2 * basis.evaluate(1, x)
            - 3 * basis.evaluate(2, x)
            + 0.5 * basis.evaluate(4, x)
        )

    expected = np.array([2, -3, 0, 0.5, 0])

    np.testing.assert_allclose(
        basis.project(f, 5),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_project_mode_outside_truncation(basis):
    coeffs = basis.project(
        lambda x: basis.evaluate(6, x),
        5,
    )

    np.testing.assert_allclose(
        coeffs,
        np.zeros(5),
        atol=ATOL,
        rtol=RTOL,
    )


def test_reconstruct_shape(basis):
    a, b = basis.domain.bounds
    xs = np.linspace(a, b, 20)

    result = basis.reconstruct([1, 2, 3], xs)

    assert result.shape == (20,)
    assert result.dtype == np.float64


def test_reconstruct_zero_coefficients(basis):
    a, b = basis.domain.bounds
    xs = np.linspace(a, b, 20)

    np.testing.assert_allclose(
        basis.reconstruct([0, 0, 0], xs),
        np.zeros(20),
        atol=ATOL,
        rtol=RTOL,
    )


def test_reconstruct_single_mode(basis):
    a, b = basis.domain.bounds
    xs = np.linspace(a, b, 20)

    np.testing.assert_allclose(
        basis.reconstruct([0, 0, 2], xs),
        2 * basis.evaluate(3, xs),
        atol=ATOL,
        rtol=RTOL,
    )


def test_reconstruct_known_values(basis):
    a, b = basis.domain.bounds
    L = b - a

    xs = [a, a + L / 4, a + L / 2, a + 3 * L / 4, b]

    expected = np.sqrt(2 / L) * np.array(
        [
            0,
            np.sqrt(2) - 3,
            2,
            np.sqrt(2) + 3,
            0,
        ]
    )

    np.testing.assert_allclose(
        basis.reconstruct([2, -3], xs),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_reconstruct_boundaries(basis):
    a, b = basis.domain.bounds

    np.testing.assert_allclose(
        basis.reconstruct([1, -2, 3, -4, 5], [a, b]),
        [0, 0],
        atol=ATOL,
        rtol=RTOL,
    )


def test_reconstruct_empty_coordinates(basis):
    result = basis.reconstruct([1, 2, 3], [])

    assert result.shape == (0,)


def test_reconstruct_then_project(basis):
    original = np.array([1.5, -2.0, 0.0, 3.0, -0.5])

    def f(x):
        return basis.reconstruct(original, [x])[0]

    recovered = basis.project(f, original.size)

    np.testing.assert_allclose(
        recovered,
        original,
        atol=ATOL,
        rtol=RTOL,
    )


def test_project_then_reconstruct(basis):
    a, b = basis.domain.bounds
    L = b - a
    xs = np.linspace(a, b, 100)

    def f(x):
        return 2 * np.sin(np.pi * (x - a) / L) - 3 * np.sin(3 * np.pi * (x - a) / L)

    coeffs = basis.project(f, 5)
    reconstructed = basis.reconstruct(coeffs, xs)

    np.testing.assert_allclose(
        reconstructed,
        f(xs),
        atol=ATOL,
        rtol=RTOL,
    )


def test_parseval_identity(basis):
    a, b = basis.domain.bounds
    coeffs = np.array([1.0, -2.0, 3.0, 0.5])

    def squared_function(x):
        return basis.reconstruct(coeffs, [x])[0] ** 2

    integral, _ = quad(squared_function, a, b)

    assert integral == pytest.approx(
        np.sum(coeffs**2),
        abs=ATOL,
        rel=RTOL,
    )


def test_projection_error_decreases(basis):
    a, b = basis.domain.bounds

    def projection_error(num_modes):
        coeffs = basis.project(lambda x: 1.0, num_modes)

        integral, _ = quad(
            lambda x: (1.0 - basis.reconstruct(coeffs, [x])[0]) ** 2,
            a,
            b,
        )

        return integral

    errors = [projection_error(n) for n in [1, 3, 5, 9]]

    assert all(later < earlier for earlier, later in itertools.pairwise(errors))


@pytest.mark.parametrize("method", ["evaluate", "matrix", "project"])
@pytest.mark.parametrize("mode", INVALID_MODE_TYPES)
def test_invalid_mode_type(method, mode):
    basis = SineBasis(Interval(0, 1))

    with pytest.raises(TypeError):
        if method == "evaluate":
            basis.evaluate(mode, 0.5)
        elif method == "matrix":
            basis.matrix(mode, [0.5])
        else:
            basis.project(lambda x: 1.0, mode)


@pytest.mark.parametrize("method", ["evaluate", "matrix", "project"])
@pytest.mark.parametrize("mode", INVALID_MODE_VALUES)
def test_invalid_mode_value(method, mode):
    basis = SineBasis(Interval(0, 1))

    with pytest.raises(ValueError):
        if method == "evaluate":
            basis.evaluate(mode, 0.5)
        elif method == "matrix":
            basis.matrix(mode, [0.5])
        else:
            basis.project(lambda x: 1.0, mode)


@pytest.mark.parametrize(
    "xs",
    [
        [True],
        [0.5, False],
        ["0.5"],
        [0.5, "0.7"],
        [0.5 + 0j],
        [None],
    ],
)
@pytest.mark.parametrize("method", ["evaluate", "matrix", "reconstruct"])
def test_invalid_coordinate_type(xs, method):
    basis = SineBasis(Interval(0, 1))

    with pytest.raises(TypeError):
        if method == "evaluate":
            basis.evaluate(1, xs)
        elif method == "matrix":
            basis.matrix(1, xs)
        else:
            basis.reconstruct([1], xs)


@pytest.mark.parametrize(
    "xs",
    [
        [-0.1],
        [1.1],
        [0.5, 1.1],
        [np.nan],
        [np.inf],
        [-np.inf],
    ],
)
@pytest.mark.parametrize("method", ["evaluate", "matrix", "reconstruct"])
def test_invalid_coordinate_value(xs, method):
    basis = SineBasis(Interval(0, 1))

    with pytest.raises(ValueError):
        if method == "evaluate":
            basis.evaluate(1, xs)
        elif method == "matrix":
            basis.matrix(1, xs)
        else:
            basis.reconstruct([1], xs)


@pytest.mark.parametrize(
    "xs",
    [0.5, [[0.2, 0.5]], np.zeros((2, 2, 2))],
)
def test_matrix_invalid_coordinate_dimension(xs):
    basis = SineBasis(Interval(0, 1))

    with pytest.raises(ValueError):
        basis.matrix(3, xs)


@pytest.mark.parametrize(
    "xs",
    [0.5, [[0.2, 0.5]], np.zeros((2, 2, 2))],
)
def test_reconstruct_invalid_coordinate_dimension(xs):
    basis = SineBasis(Interval(0, 1))

    with pytest.raises(ValueError):
        basis.reconstruct([1, 2], xs)


@pytest.mark.parametrize(
    "coeffs",
    [
        [True],
        [1, False],
        ["1"],
        [1 + 0j],
        [None],
    ],
)
def test_invalid_coefficient_type(coeffs):
    basis = SineBasis(Interval(0, 1))

    with pytest.raises(TypeError):
        basis.reconstruct(coeffs, [0.5])


@pytest.mark.parametrize(
    "coeffs",
    [
        [],
        1.0,
        [[1, 2]],
        np.zeros((2, 2, 2)),
        [np.nan],
        [np.inf],
        [-np.inf],
    ],
)
def test_invalid_coefficient_value(coeffs):
    basis = SineBasis(Interval(0, 1))

    with pytest.raises(ValueError):
        basis.reconstruct(coeffs, [0.5])


@pytest.mark.parametrize("f", [None, 1, "function", [1, 2]])
def test_project_non_callable(f):
    basis = SineBasis(Interval(0, 1))

    with pytest.raises(TypeError):
        basis.project(f, 5)
