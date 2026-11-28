import numpy as np
import pytest

from spdepy.basis import SineBasis
from spdepy.boundaries import Dirichlet
from spdepy.covariance import SpectralCovariance
from spdepy.domains import Interval
from spdepy.models import LinearSpectralModel
from spdepy.noise import SpectralWienerNoise
from spdepy.operators import Laplacian

VALID_INTERVALS = [
    (0, np.pi),
    (0, 1),
    (-1, 1),
    (1, 5),
    (-6, -2),
    (0, 0.001),
]

INVALID_INT_TYPES = [
    True,
    False,
    np.bool_(True),
    1.5,
    2.0,
    "2",
    1 + 0j,
    None,
]

INVALID_REAL_TYPES = [
    True,
    False,
    np.bool_(True),
    "1",
    1 + 0j,
    [1.0],
    np.array([1.0]),
    None,
]

INVALID_NUM_MODES_VALUES = [0, -1, -10]

INVALID_THETA_VALUES = [
    0,
    -0.0,
    -1,
    -0.1,
    np.nan,
    np.inf,
    -np.inf,
]

INVALID_SIGMA_VALUES = [
    -1,
    -0.1,
    -1e-15,
    np.nan,
    np.inf,
    -np.inf,
]

INVALID_COEFF_TYPES = [
    True,
    "1",
    None,
    [True, 2, 3, 4, 5],
    [1, "2", 3, 4, 5],
    [1, 2, 3 + 0j, 4, 5],
    np.ones(5, dtype=bool),
    np.ones(5, dtype=complex),
    np.ones(5, dtype=object),
]

INVALID_COEFF_VALUES = [
    [],
    1.0,
    [1],
    [1, 2, 3, 4],
    [1, 2, 3, 4, 5, 6],
    [[1, 2, 3, 4, 5]],
    [[1], [2], [3], [4], [5]],
    [1, 2, np.nan, 4, 5],
    [1, 2, np.inf, 4, 5],
    [1, 2, -np.inf, 4, 5],
]

ATOL = 1e-10
RTOL = 1e-10
SEED = 42
NUM_MODES = 5
THETA = 0.5
SIGMA = 0.2


@pytest.fixture(params=VALID_INTERVALS)
def domain(request):
    return Interval(*request.param)


@pytest.fixture
def operator(domain):
    return Laplacian(domain, Dirichlet())


@pytest.fixture
def covariance(operator):
    return SpectralCovariance(
        operator.eigenbasis,
        lambda k: 1 / k**2,
    )


@pytest.fixture
def noise(covariance):
    return SpectralWienerNoise(covariance)


@pytest.fixture
def model(operator, noise):
    return LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)


@pytest.fixture
def reference_operator():
    return Laplacian(
        Interval(0, np.pi),
        Dirichlet(),
    )


@pytest.fixture
def reference_noise(reference_operator):
    covariance = SpectralCovariance(
        reference_operator.eigenbasis,
        lambda k: 1 / k**2,
    )
    return SpectralWienerNoise(covariance)


@pytest.fixture
def reference_model(reference_operator, reference_noise):
    return LinearSpectralModel(
        reference_operator,
        reference_noise,
        NUM_MODES,
        THETA,
        SIGMA,
    )


def test_construction(model, operator, noise):
    assert model.operator is operator
    assert model.noise is noise
    assert model.eigenbasis is operator.eigenbasis
    assert model.domain is operator.domain
    assert model.num_modes == NUM_MODES
    assert model.theta == THETA
    assert model.sigma == SIGMA


def test_property_types(model):
    assert isinstance(model.operator, Laplacian)
    assert isinstance(model.noise, SpectralWienerNoise)
    assert isinstance(model.eigenbasis, SineBasis)
    assert isinstance(model.domain, Interval)
    assert type(model.num_modes) is int
    assert type(model.theta) is float
    assert type(model.sigma) is float


@pytest.mark.parametrize(
    "property_name",
    [
        "operator",
        "noise",
        "eigenbasis",
        "domain",
        "num_modes",
        "theta",
        "sigma",
        "drift_eigenvalues",
        "drift_matrix",
        "diffusion_eigenvalues",
        "diffusion_matrix",
        "diffusion_covariance",
    ],
)
def test_read_only_properties(model, property_name):
    with pytest.raises(AttributeError):
        setattr(model, property_name, None)


def test_same_basis_compatibility(model):
    assert model.eigenbasis is model.noise.basis


def test_equivalent_basis_compatibility(operator):
    domain = Interval(*operator.domain.bounds)
    basis = SineBasis(domain)
    covariance = SpectralCovariance(basis, lambda k: 1 / k**2)
    noise = SpectralWienerNoise(covariance)

    model = LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)

    assert model.eigenbasis is not noise.basis
    assert model.domain is not noise.basis.domain
    assert model.domain.bounds == noise.basis.domain.bounds


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 1),
        (0, 2 * np.pi),
        (1, 1 + np.pi),
        (-np.pi, 0),
        (-1, 1),
        (0, np.pi + 1e-12),
    ],
)
def test_incompatible_domains(reference_operator, bounds):
    basis = SineBasis(Interval(*bounds))
    covariance = SpectralCovariance(basis, lambda k: 1.0)
    noise = SpectralWienerNoise(covariance)

    with pytest.raises(ValueError):
        LinearSpectralModel(
            reference_operator,
            noise,
            NUM_MODES,
            THETA,
            SIGMA,
        )


@pytest.mark.parametrize(
    "operator",
    [
        None,
        1,
        "operator",
        np.eye(5),
        Interval(0, 1),
        SineBasis(Interval(0, 1)),
    ],
)
def test_invalid_operator(reference_noise, operator):
    with pytest.raises(TypeError):
        LinearSpectralModel(
            operator,
            reference_noise,
            NUM_MODES,
            THETA,
            SIGMA,
        )


@pytest.mark.parametrize(
    "noise",
    [
        None,
        1,
        "noise",
        np.eye(5),
        np.random.default_rng(SEED),
        SpectralCovariance(
            SineBasis(Interval(0, 1)),
            lambda k: 1.0,
        ),
    ],
)
def test_invalid_noise(reference_operator, noise):
    with pytest.raises(TypeError):
        LinearSpectralModel(
            reference_operator,
            noise,
            NUM_MODES,
            THETA,
            SIGMA,
        )


@pytest.mark.parametrize("num_modes", INVALID_INT_TYPES)
def test_invalid_num_modes_type(reference_operator, reference_noise, num_modes):
    with pytest.raises(TypeError):
        LinearSpectralModel(
            reference_operator,
            reference_noise,
            num_modes,
            THETA,
            SIGMA,
        )


@pytest.mark.parametrize("num_modes", INVALID_NUM_MODES_VALUES)
def test_invalid_num_modes_value(reference_operator, reference_noise, num_modes):
    with pytest.raises(ValueError):
        LinearSpectralModel(
            reference_operator,
            reference_noise,
            num_modes,
            THETA,
            SIGMA,
        )


@pytest.mark.parametrize("theta", INVALID_REAL_TYPES)
def test_invalid_theta_type(reference_operator, reference_noise, theta):
    with pytest.raises(TypeError):
        LinearSpectralModel(
            reference_operator,
            reference_noise,
            NUM_MODES,
            theta,
            SIGMA,
        )


@pytest.mark.parametrize("theta", INVALID_THETA_VALUES)
def test_invalid_theta_value(reference_operator, reference_noise, theta):
    with pytest.raises(ValueError):
        LinearSpectralModel(
            reference_operator,
            reference_noise,
            NUM_MODES,
            theta,
            SIGMA,
        )


@pytest.mark.parametrize("sigma", INVALID_REAL_TYPES)
def test_invalid_sigma_type(reference_operator, reference_noise, sigma):
    with pytest.raises(TypeError):
        LinearSpectralModel(
            reference_operator,
            reference_noise,
            NUM_MODES,
            THETA,
            sigma,
        )


@pytest.mark.parametrize("sigma", INVALID_SIGMA_VALUES)
def test_invalid_sigma_value(reference_operator, reference_noise, sigma):
    with pytest.raises(ValueError):
        LinearSpectralModel(
            reference_operator,
            reference_noise,
            NUM_MODES,
            THETA,
            sigma,
        )


@pytest.mark.parametrize(
    "num_modes",
    [1, 5, np.int32(5), np.int64(5)],
)
def test_valid_num_modes_types(reference_operator, reference_noise, num_modes):
    model = LinearSpectralModel(
        reference_operator,
        reference_noise,
        num_modes,
        THETA,
        SIGMA,
    )

    assert type(model.num_modes) is int
    assert model.num_modes == num_modes


@pytest.mark.parametrize(
    "theta",
    [
        1,
        0.5,
        np.int64(1),
        np.float32(0.5),
        np.float64(0.5),
    ],
)
def test_valid_theta_types(reference_operator, reference_noise, theta):
    model = LinearSpectralModel(
        reference_operator,
        reference_noise,
        NUM_MODES,
        theta,
        SIGMA,
    )

    assert type(model.theta) is float
    assert model.theta == float(theta)


@pytest.mark.parametrize(
    "sigma",
    [
        0,
        -0.0,
        1,
        0.5,
        np.int64(1),
        np.float32(0.5),
        np.float64(0.5),
    ],
)
def test_valid_sigma_types(reference_operator, reference_noise, sigma):
    model = LinearSpectralModel(
        reference_operator,
        reference_noise,
        NUM_MODES,
        THETA,
        sigma,
    )

    assert type(model.sigma) is float
    assert model.sigma == float(sigma)


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
def test_drift_eigenvalues_shape(operator, noise, num_modes):
    model = LinearSpectralModel(operator, noise, num_modes, THETA, SIGMA)

    assert model.drift_eigenvalues.shape == (num_modes,)
    assert model.drift_eigenvalues.dtype == np.float64


def test_drift_eigenvalues_known(reference_model):
    expected = np.array([-0.5, -2.0, -4.5, -8.0, -12.5])

    np.testing.assert_allclose(
        reference_model.drift_eigenvalues,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_eigenvalues_arbitrary_domain(model):
    ks = np.arange(1, model.num_modes + 1)
    length = model.domain.length

    expected = -model.theta * (ks * np.pi / length) ** 2

    np.testing.assert_allclose(
        model.drift_eigenvalues,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_eigenvalues_operator_consistency(model):
    expected = model.theta * model.operator.eigenvalues(model.num_modes)

    np.testing.assert_allclose(
        model.drift_eigenvalues,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_eigenvalues_negative(model):
    assert np.all(model.drift_eigenvalues < 0)


def test_drift_eigenvalues_strictly_decreasing(model):
    assert np.all(np.diff(model.drift_eigenvalues) < 0)


def test_drift_eigenvalues_quadratic_growth(model):
    ks = np.arange(1, model.num_modes + 1)

    np.testing.assert_allclose(
        model.drift_eigenvalues / model.drift_eigenvalues[0],
        ks**2,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("factor", [0.25, 0.5, 2, 4, 10])
def test_drift_eigenvalues_theta_scaling(operator, noise, factor):
    model1 = LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)
    model2 = LinearSpectralModel(operator, noise, NUM_MODES, factor * THETA, SIGMA)

    np.testing.assert_allclose(
        model2.drift_eigenvalues,
        factor * model1.drift_eigenvalues,
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_matrix_shape(model):
    assert model.drift_matrix.shape == (
        model.num_modes,
        model.num_modes,
    )
    assert model.drift_matrix.dtype == np.float64


def test_drift_matrix_known(reference_model):
    expected = np.diag([-0.5, -2.0, -4.5, -8.0, -12.5])

    np.testing.assert_allclose(
        reference_model.drift_matrix,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_matrix_diagonal(model):
    matrix = model.drift_matrix

    np.testing.assert_array_equal(
        matrix,
        np.diag(np.diag(matrix)),
    )


def test_drift_matrix_eigenvalue_consistency(model):
    np.testing.assert_array_equal(
        np.diag(model.drift_matrix),
        model.drift_eigenvalues,
    )


def test_drift_matrix_symmetric(model):
    np.testing.assert_array_equal(
        model.drift_matrix,
        model.drift_matrix.T,
    )


def test_drift_matrix_negative_definite(model):
    eigenvalues = np.linalg.eigvalsh(model.drift_matrix)

    assert np.all(eigenvalues < 0)


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
def test_diffusion_eigenvalues_shape(operator, noise, num_modes):
    model = LinearSpectralModel(operator, noise, num_modes, THETA, SIGMA)

    assert model.diffusion_eigenvalues.shape == (num_modes,)
    assert model.diffusion_eigenvalues.dtype == np.float64


def test_diffusion_eigenvalues_known(reference_model):
    ks = np.arange(1, reference_model.num_modes + 1)
    expected = SIGMA / ks

    np.testing.assert_allclose(
        reference_model.diffusion_eigenvalues,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_diffusion_eigenvalues_covariance_consistency(model):
    expected = model.sigma * np.sqrt(
        model.noise.covariance.eigenvalues(model.num_modes)
    )

    np.testing.assert_allclose(
        model.diffusion_eigenvalues,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_diffusion_eigenvalues_nonnegative(model):
    assert np.all(model.diffusion_eigenvalues >= 0)


@pytest.mark.parametrize("factor", [0, 0.25, 0.5, 2, 4])
def test_diffusion_eigenvalues_sigma_scaling(operator, noise, factor):
    model1 = LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)
    model2 = LinearSpectralModel(operator, noise, NUM_MODES, THETA, factor * SIGMA)

    np.testing.assert_allclose(
        model2.diffusion_eigenvalues,
        factor * model1.diffusion_eigenvalues,
        atol=ATOL,
        rtol=RTOL,
    )


def test_diffusion_matrix_shape(model):
    assert model.diffusion_matrix.shape == (
        model.num_modes,
        model.num_modes,
    )
    assert model.diffusion_matrix.dtype == np.float64


def test_diffusion_matrix_known(reference_model):
    ks = np.arange(1, reference_model.num_modes + 1)
    expected = np.diag(SIGMA / ks)

    np.testing.assert_allclose(
        reference_model.diffusion_matrix,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_diffusion_matrix_diagonal(model):
    matrix = model.diffusion_matrix

    np.testing.assert_array_equal(
        matrix,
        np.diag(np.diag(matrix)),
    )


def test_diffusion_matrix_eigenvalue_consistency(model):
    np.testing.assert_array_equal(
        np.diag(model.diffusion_matrix),
        model.diffusion_eigenvalues,
    )


def test_diffusion_matrix_symmetric(model):
    np.testing.assert_array_equal(
        model.diffusion_matrix,
        model.diffusion_matrix.T,
    )


def test_diffusion_matrix_positive_semidefinite(model):
    eigenvalues = np.linalg.eigvalsh(model.diffusion_matrix)

    assert np.all(eigenvalues >= 0)


def test_diffusion_covariance_shape(model):
    assert model.diffusion_covariance.shape == (
        model.num_modes,
        model.num_modes,
    )
    assert model.diffusion_covariance.dtype == np.float64


def test_diffusion_covariance_known(reference_model):
    ks = np.arange(1, reference_model.num_modes + 1)
    expected = np.diag(SIGMA**2 / ks**2)

    np.testing.assert_allclose(
        reference_model.diffusion_covariance,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_diffusion_covariance_diagonal(model):
    matrix = model.diffusion_covariance

    np.testing.assert_array_equal(
        matrix,
        np.diag(np.diag(matrix)),
    )


def test_diffusion_covariance_symmetric(model):
    np.testing.assert_array_equal(
        model.diffusion_covariance,
        model.diffusion_covariance.T,
    )


def test_diffusion_covariance_positive_semidefinite(model):
    eigenvalues = np.linalg.eigvalsh(model.diffusion_covariance)

    assert np.all(eigenvalues >= 0)


def test_diffusion_covariance_noise_consistency(model):
    expected = model.sigma**2 * model.noise.covariance.matrix(model.num_modes)

    np.testing.assert_allclose(
        model.diffusion_covariance,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_diffusion_covariance_matrix_consistency(model):
    diffusion = model.diffusion_matrix

    np.testing.assert_allclose(
        model.diffusion_covariance,
        diffusion @ diffusion.T,
        atol=ATOL,
        rtol=RTOL,
    )


def test_diffusion_covariance_eigenvalue_consistency(model):
    np.testing.assert_allclose(
        np.diag(model.diffusion_covariance),
        model.diffusion_eigenvalues**2,
        atol=ATOL,
        rtol=RTOL,
    )


def test_diffusion_covariance_trace(model):
    expected = model.sigma**2 * model.noise.covariance.trace(model.num_modes)

    assert np.trace(model.diffusion_covariance) == pytest.approx(
        expected, abs=ATOL, rel=RTOL
    )


@pytest.mark.parametrize("factor", [0, 0.25, 0.5, 2, 4])
def test_diffusion_covariance_sigma_scaling(operator, noise, factor):
    model1 = LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)
    model2 = LinearSpectralModel(operator, noise, NUM_MODES, THETA, factor * SIGMA)

    np.testing.assert_allclose(
        model2.diffusion_covariance,
        factor**2 * model1.diffusion_covariance,
        atol=ATOL,
        rtol=RTOL,
    )


def test_zero_sigma(operator, noise):
    model = LinearSpectralModel(operator, noise, NUM_MODES, THETA, 0)

    np.testing.assert_array_equal(
        model.diffusion_eigenvalues,
        np.zeros(NUM_MODES),
    )

    np.testing.assert_array_equal(
        model.diffusion_matrix,
        np.zeros((NUM_MODES, NUM_MODES)),
    )

    np.testing.assert_array_equal(
        model.diffusion_covariance,
        np.zeros((NUM_MODES, NUM_MODES)),
    )

    assert np.all(model.drift_eigenvalues < 0)


def test_zero_covariance(operator):
    covariance = SpectralCovariance(operator.eigenbasis, lambda k: 0.0)
    noise = SpectralWienerNoise(covariance)

    model = LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)

    np.testing.assert_array_equal(
        model.diffusion_eigenvalues,
        np.zeros(NUM_MODES),
    )

    np.testing.assert_array_equal(
        model.diffusion_matrix,
        np.zeros((NUM_MODES, NUM_MODES)),
    )

    np.testing.assert_array_equal(
        model.diffusion_covariance,
        np.zeros((NUM_MODES, NUM_MODES)),
    )


def test_white_noise(operator):
    covariance = SpectralCovariance(operator.eigenbasis, lambda k: 1.0)
    noise = SpectralWienerNoise(covariance)

    model = LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)

    np.testing.assert_allclose(
        model.diffusion_eigenvalues,
        np.full(NUM_MODES, SIGMA),
        atol=ATOL,
        rtol=RTOL,
    )

    np.testing.assert_allclose(
        model.diffusion_matrix,
        SIGMA * np.eye(NUM_MODES),
        atol=ATOL,
        rtol=RTOL,
    )

    np.testing.assert_allclose(
        model.diffusion_covariance,
        SIGMA**2 * np.eye(NUM_MODES),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("rank", [1, 2, 3, 5])
def test_finite_rank_covariance(operator, rank):
    covariance = SpectralCovariance(
        operator.eigenbasis,
        lambda k: 1 / k**2 if k <= rank else 0.0,
    )
    noise = SpectralWienerNoise(covariance)

    model = LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)

    assert np.linalg.matrix_rank(model.diffusion_matrix) == rank
    assert np.linalg.matrix_rank(model.diffusion_covariance) == rank

    np.testing.assert_array_equal(
        model.diffusion_eigenvalues[rank:],
        np.zeros(NUM_MODES - rank),
    )

    np.testing.assert_allclose(
        model.diffusion_covariance,
        model.diffusion_matrix @ model.diffusion_matrix.T,
        atol=ATOL,
        rtol=RTOL,
    )


def test_nonconsecutive_zero_covariance_modes(operator):
    covariance = SpectralCovariance(
        operator.eigenbasis,
        lambda k: 1.0 if k % 2 else 0.0,
    )
    noise = SpectralWienerNoise(covariance)

    model = LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)

    expected = SIGMA * np.array([1, 0, 1, 0, 1])

    np.testing.assert_allclose(
        model.diffusion_eigenvalues,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )

    np.testing.assert_allclose(
        model.diffusion_covariance,
        np.diag(expected**2),
        atol=ATOL,
        rtol=RTOL,
    )


def test_nonmonotonic_covariance(operator):
    values = np.array([1.0, 3.0, 0.0, 2.0, 0.5])
    covariance = SpectralCovariance(
        operator.eigenbasis,
        lambda k: values[k - 1],
    )
    noise = SpectralWienerNoise(covariance)

    model = LinearSpectralModel(operator, noise, NUM_MODES, THETA, SIGMA)

    np.testing.assert_allclose(
        model.diffusion_eigenvalues,
        SIGMA * np.sqrt(values),
        atol=ATOL,
        rtol=RTOL,
    )

    np.testing.assert_allclose(
        model.diffusion_covariance,
        np.diag(SIGMA**2 * values),
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_independent_of_sigma(operator, noise):
    model1 = LinearSpectralModel(operator, noise, NUM_MODES, THETA, 0)
    model2 = LinearSpectralModel(operator, noise, NUM_MODES, THETA, 2)

    np.testing.assert_array_equal(
        model1.drift_eigenvalues,
        model2.drift_eigenvalues,
    )

    np.testing.assert_array_equal(
        model1.drift_matrix,
        model2.drift_matrix,
    )


def test_diffusion_independent_of_theta(operator, noise):
    model1 = LinearSpectralModel(operator, noise, NUM_MODES, 0.5, SIGMA)
    model2 = LinearSpectralModel(operator, noise, NUM_MODES, 2.0, SIGMA)

    np.testing.assert_array_equal(
        model1.diffusion_eigenvalues,
        model2.diffusion_eigenvalues,
    )

    np.testing.assert_array_equal(
        model1.diffusion_matrix,
        model2.diffusion_matrix,
    )

    np.testing.assert_array_equal(
        model1.diffusion_covariance,
        model2.diffusion_covariance,
    )


def test_drift_independent_of_covariance(operator):
    covariance1 = SpectralCovariance(operator.eigenbasis, lambda k: 0.0)
    covariance2 = SpectralCovariance(operator.eigenbasis, lambda k: 1.0)

    model1 = LinearSpectralModel(
        operator,
        SpectralWienerNoise(covariance1),
        NUM_MODES,
        THETA,
        SIGMA,
    )
    model2 = LinearSpectralModel(
        operator,
        SpectralWienerNoise(covariance2),
        NUM_MODES,
        THETA,
        SIGMA,
    )

    np.testing.assert_array_equal(
        model1.drift_eigenvalues,
        model2.drift_eigenvalues,
    )


@pytest.mark.parametrize(
    "coeffs",
    [
        [1, 2, 3, 4, 5],
        (1, 2, 3, 4, 5),
        np.arange(1, 6, dtype=np.int64),
        np.arange(1, 6, dtype=np.float32),
        np.arange(1, 6, dtype=np.float64),
    ],
)
def test_drift_shape_and_dtype(model, coeffs):
    result = model.drift(coeffs)

    assert isinstance(result, np.ndarray)
    assert result.shape == (model.num_modes,)
    assert result.dtype == np.float64


def test_drift_known(reference_model):
    coeffs = np.array([2, -3, 5, 0, 1])
    expected = np.array([-1, 6, -22.5, 0, -12.5])

    np.testing.assert_allclose(
        reference_model.drift(coeffs),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_zero(model):
    np.testing.assert_array_equal(
        model.drift(np.zeros(model.num_modes)),
        np.zeros(model.num_modes),
    )


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_drift_eigenfunction(model, k):
    coeffs = np.zeros(model.num_modes)
    coeffs[k - 1] = 1.0

    expected = model.theta * model.operator.eigenvalue(k) * coeffs

    np.testing.assert_allclose(
        model.drift(coeffs),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_matrix_consistency(model):
    rng = np.random.default_rng(SEED)
    coeffs = rng.standard_normal(model.num_modes)

    np.testing.assert_allclose(
        model.drift(coeffs),
        model.drift_matrix @ coeffs,
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_operator_consistency(model):
    rng = np.random.default_rng(SEED)
    coeffs = rng.standard_normal(model.num_modes)

    np.testing.assert_allclose(
        model.drift(coeffs),
        model.theta * model.operator.apply(coeffs),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("scalar", [-3, -1, 0, 0.5, 2, 10])
def test_drift_homogeneity(model, scalar):
    rng = np.random.default_rng(SEED)
    coeffs = rng.standard_normal(model.num_modes)

    np.testing.assert_allclose(
        model.drift(scalar * coeffs),
        scalar * model.drift(coeffs),
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_additivity(model):
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(model.num_modes)
    y = rng.standard_normal(model.num_modes)

    np.testing.assert_allclose(
        model.drift(x + y),
        model.drift(x) + model.drift(y),
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_linearity(model):
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(model.num_modes)
    y = rng.standard_normal(model.num_modes)

    alpha = 2.5
    beta = -1.5

    np.testing.assert_allclose(
        model.drift(alpha * x + beta * y),
        alpha * model.drift(x) + beta * model.drift(y),
        atol=ATOL,
        rtol=RTOL,
    )


def test_drift_dissipative(model):
    rng = np.random.default_rng(SEED)
    coeffs = rng.standard_normal(model.num_modes)

    assert np.dot(coeffs, model.drift(coeffs)) < 0


def test_drift_self_adjoint(model):
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal(model.num_modes)
    y = rng.standard_normal(model.num_modes)

    left = np.dot(x, model.drift(y))
    right = np.dot(model.drift(x), y)

    assert left == pytest.approx(right, abs=ATOL, rel=RTOL)


def test_drift_no_input_modification(model):
    rng = np.random.default_rng(SEED)
    coeffs = rng.standard_normal(model.num_modes)
    original = coeffs.copy()

    model.drift(coeffs)

    np.testing.assert_array_equal(coeffs, original)


def test_drift_reconstruction(model):
    rng = np.random.default_rng(SEED)
    coeffs = rng.standard_normal(model.num_modes)

    xs = np.linspace(
        model.domain.a,
        model.domain.b,
        101,
    )

    result = model.eigenbasis.reconstruct(model.drift(coeffs), xs)

    expected = np.zeros_like(xs)

    for k in range(1, model.num_modes + 1):
        eigenvalue = -((k * np.pi / model.domain.length) ** 2)

        expected += (
            model.theta * eigenvalue * coeffs[k - 1] * model.eigenbasis.evaluate(k, xs)
        )

    tolerance = ATOL * max(1.0, np.max(np.abs(expected)))

    np.testing.assert_allclose(
        result,
        expected,
        atol=tolerance,
        rtol=RTOL,
    )


@pytest.mark.parametrize("coeffs", INVALID_COEFF_TYPES)
def test_drift_invalid_coeffs_type(reference_model, coeffs):
    with pytest.raises(TypeError):
        reference_model.drift(coeffs)


@pytest.mark.parametrize("coeffs", INVALID_COEFF_VALUES)
def test_drift_invalid_coeffs_value(reference_model, coeffs):
    with pytest.raises(ValueError):
        reference_model.drift(coeffs)


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
def test_model_dimension(operator, noise, num_modes):
    model = LinearSpectralModel(operator, noise, num_modes, THETA, SIGMA)

    assert model.num_modes == num_modes
    assert model.drift_eigenvalues.shape == (num_modes,)
    assert model.diffusion_eigenvalues.shape == (num_modes,)
    assert model.drift_matrix.shape == (num_modes, num_modes)
    assert model.diffusion_matrix.shape == (num_modes, num_modes)
    assert model.diffusion_covariance.shape == (num_modes, num_modes)
    assert model.drift(np.ones(num_modes)).shape == (num_modes,)


def test_truncation_consistency(operator, noise):
    small = LinearSpectralModel(operator, noise, 5, THETA, SIGMA)
    large = LinearSpectralModel(operator, noise, 10, THETA, SIGMA)

    np.testing.assert_array_equal(
        small.drift_eigenvalues,
        large.drift_eigenvalues[:5],
    )

    np.testing.assert_array_equal(
        small.diffusion_eigenvalues,
        large.diffusion_eigenvalues[:5],
    )

    np.testing.assert_array_equal(
        small.drift_matrix,
        large.drift_matrix[:5, :5],
    )

    np.testing.assert_array_equal(
        small.diffusion_matrix,
        large.diffusion_matrix[:5, :5],
    )

    np.testing.assert_array_equal(
        small.diffusion_covariance,
        large.diffusion_covariance[:5, :5],
    )

    coeffs = np.arange(1, 11, dtype=np.float64)

    np.testing.assert_array_equal(
        small.drift(coeffs[:5]),
        large.drift(coeffs)[:5],
    )


@pytest.mark.parametrize(
    "property_name",
    [
        "drift_eigenvalues",
        "drift_matrix",
        "diffusion_eigenvalues",
        "diffusion_matrix",
        "diffusion_covariance",
    ],
)
def test_returned_array_modification_does_not_affect_model(model, property_name):
    expected = getattr(model, property_name).copy()
    result = getattr(model, property_name)

    result[...] = 999

    np.testing.assert_array_equal(
        getattr(model, property_name),
        expected,
    )


@pytest.mark.parametrize(
    "value, exception",
    [
        (-1.0, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
        ("1", TypeError),
        (1 + 0j, TypeError),
        (None, TypeError),
    ],
)
@pytest.mark.parametrize(
    "property_name",
    [
        "diffusion_eigenvalues",
        "diffusion_matrix",
        "diffusion_covariance",
    ],
)
def test_invalid_covariance_eigenvalue(
    reference_operator, value, exception, property_name
):
    covariance = SpectralCovariance(
        reference_operator.eigenbasis,
        lambda k: 1.0 if k < 3 else value,
    )
    noise = SpectralWienerNoise(covariance)

    model = LinearSpectralModel(
        reference_operator,
        noise,
        NUM_MODES,
        THETA,
        SIGMA,
    )

    with pytest.raises(exception):
        getattr(model, property_name)


def test_partial_covariance_validity(reference_operator):
    covariance = SpectralCovariance(
        reference_operator.eigenbasis,
        lambda k: 1.0 if k <= 3 else -1.0,
    )
    noise = SpectralWienerNoise(covariance)

    small = LinearSpectralModel(reference_operator, noise, 3, THETA, SIGMA)
    large = LinearSpectralModel(reference_operator, noise, 5, THETA, SIGMA)

    assert small.diffusion_eigenvalues.shape == (3,)
    assert small.diffusion_matrix.shape == (3, 3)
    assert small.diffusion_covariance.shape == (3, 3)

    with pytest.raises(ValueError):
        _ = large.diffusion_eigenvalues

    with pytest.raises(ValueError):
        _ = large.diffusion_matrix

    with pytest.raises(ValueError):
        _ = large.diffusion_covariance


def test_drift_does_not_evaluate_covariance(reference_operator):
    def eigenvalues(k):
        raise RuntimeError("Covariance evaluated unexpectedly")

    covariance = SpectralCovariance(reference_operator.eigenbasis, eigenvalues)
    noise = SpectralWienerNoise(covariance)

    model = LinearSpectralModel(
        reference_operator,
        noise,
        NUM_MODES,
        THETA,
        SIGMA,
    )

    assert model.drift_eigenvalues.shape == (NUM_MODES,)
    assert model.drift_matrix.shape == (NUM_MODES, NUM_MODES)
    assert model.drift(np.ones(NUM_MODES)).shape == (NUM_MODES,)


def test_drift_does_not_construct_dense_matrix(model, monkeypatch):
    def fail(self):
        raise RuntimeError("Dense drift matrix constructed")

    monkeypatch.setattr(
        LinearSpectralModel,
        "drift_matrix",
        property(fail),
    )

    coeffs = np.arange(1, model.num_modes + 1)

    expected = model.drift_eigenvalues * coeffs

    np.testing.assert_array_equal(
        model.drift(coeffs),
        expected,
    )
