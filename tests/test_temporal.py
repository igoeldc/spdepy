import numpy as np
import pytest

from spdepy.boundaries import Dirichlet
from spdepy.covariance import SpectralCovariance
from spdepy.domains import Interval
from spdepy.models import LinearSpectralModel
from spdepy.noise import SpectralWienerNoise
from spdepy.operators import Laplacian
from spdepy.temporal import ExactLinear

VALID_INTERVALS = [
    (0, np.pi),
    (0, 1),
    (-1, 1),
    (1, 5),
    (-6, -2),
    (0, 0.001),
]

VALID_DT = [1e-8, 0.001, 0.01, 0.1, 0.5, 1.0, 10.0]

INVALID_DT_TYPES = [
    True,
    False,
    np.bool_(True),
    "0.1",
    1 + 0j,
    [0.1],
    np.array([0.1]),
    None,
]

INVALID_DT_VALUES = [
    0,
    -0.0,
    -1,
    -0.1,
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

INVALID_RNG_TYPES = [
    None,
    42,
    "42",
    np.random.RandomState(42),
    np.random.PCG64(42),
]

TRANSITION_METHODS = [
    "transition_eigenvalues",
    "transition_variances",
    "transition_matrix",
    "transition_covariance_matrix",
]

ATOL = 1e-12
RTOL = 1e-10
SEED = 42
NUM_MODES = 5
THETA = 0.5
SIGMA = 0.2
DT = 0.1
NUM_SAMPLES = 20_000
NUM_STD = 6


def make_model(
    bounds=(0, np.pi),
    num_modes=NUM_MODES,
    theta=THETA,
    sigma=SIGMA,
    eigenvalues=None,
):
    if eigenvalues is None:
        eigenvalues = lambda k: 1 / k**2

    domain = Interval(*bounds)
    operator = Laplacian(domain, Dirichlet())
    covariance = SpectralCovariance(operator.eigenbasis, eigenvalues)
    noise = SpectralWienerNoise(covariance)

    return LinearSpectralModel(operator, noise, num_modes, theta, sigma)


@pytest.fixture(params=VALID_INTERVALS)
def model(request):
    return make_model(bounds=request.param)


@pytest.fixture
def integrator(model):
    return ExactLinear(model)


@pytest.fixture
def reference_model():
    return make_model()


@pytest.fixture
def reference_integrator(reference_model):
    return ExactLinear(reference_model)


def test_construction(integrator, model):
    assert integrator.model is model


def test_model_property_type(integrator):
    assert isinstance(integrator.model, LinearSpectralModel)


def test_model_property_read_only(integrator):
    with pytest.raises(AttributeError):
        integrator.model = None


@pytest.mark.parametrize(
    "model",
    [
        None,
        1,
        "model",
        np.eye(5),
        Interval(0, 1),
        Laplacian(Interval(0, 1), Dirichlet()),
    ],
)
def test_invalid_model(model):
    with pytest.raises(TypeError):
        ExactLinear(model)


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_eigenvalues_shape(integrator, dt):
    result = integrator.transition_eigenvalues(dt)

    assert isinstance(result, np.ndarray)
    assert result.shape == (integrator.model.num_modes,)
    assert result.dtype == np.float64


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_eigenvalues_known(reference_integrator, dt):
    ks = np.arange(1, NUM_MODES + 1)
    expected = np.exp(-THETA * ks**2 * dt)

    np.testing.assert_allclose(
        reference_integrator.transition_eigenvalues(dt),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_eigenvalues_arbitrary_domain(integrator, dt):
    model = integrator.model
    ks = np.arange(1, model.num_modes + 1)
    length = model.domain.length

    expected = np.exp(-model.theta * (ks * np.pi / length) ** 2 * dt)

    np.testing.assert_allclose(
        integrator.transition_eigenvalues(dt),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_eigenvalues_range(integrator, dt):
    result = integrator.transition_eigenvalues(dt)

    assert np.all(result >= 0)
    assert np.all(result <= 1)


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_eigenvalues_mode_order(integrator, dt):
    result = integrator.transition_eigenvalues(dt)

    assert np.all(np.diff(result) <= 0)


def test_transition_eigenvalues_decrease_with_time(integrator):
    early = integrator.transition_eigenvalues(0.01)
    late = integrator.transition_eigenvalues(1.0)

    assert np.all(late <= early)


def test_transition_eigenvalues_small_time(reference_integrator):
    dt = 1e-12

    result = reference_integrator.transition_eigenvalues(dt)

    np.testing.assert_allclose(
        result,
        np.ones(NUM_MODES),
        atol=2e-11,
        rtol=0,
    )


def test_transition_eigenvalues_large_time(reference_integrator):
    result = reference_integrator.transition_eigenvalues(1e6)

    np.testing.assert_array_equal(
        result,
        np.zeros(NUM_MODES),
    )


@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_transition_eigenvalues_theta_dependence(theta):
    integrator = ExactLinear(make_model(theta=theta))
    ks = np.arange(1, NUM_MODES + 1)

    expected = np.exp(-theta * ks**2 * DT)

    np.testing.assert_allclose(
        integrator.transition_eigenvalues(DT),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_transition_eigenvalues_independent_of_sigma():
    integrator1 = ExactLinear(make_model(sigma=0))
    integrator2 = ExactLinear(make_model(sigma=10))

    np.testing.assert_array_equal(
        integrator1.transition_eigenvalues(DT),
        integrator2.transition_eigenvalues(DT),
    )


def test_transition_eigenvalues_independent_of_covariance():
    integrator1 = ExactLinear(make_model(eigenvalues=lambda k: 0.0))
    integrator2 = ExactLinear(make_model(eigenvalues=lambda k: 10.0))

    np.testing.assert_array_equal(
        integrator1.transition_eigenvalues(DT),
        integrator2.transition_eigenvalues(DT),
    )


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_variances_shape(integrator, dt):
    result = integrator.transition_variances(dt)

    assert isinstance(result, np.ndarray)
    assert result.shape == (integrator.model.num_modes,)
    assert result.dtype == np.float64


@pytest.mark.parametrize("dt", [0.001, 0.01, 0.1, 1.0, 10.0])
def test_transition_variances_known(reference_integrator, dt):
    ks = np.arange(1, NUM_MODES + 1)

    expected = SIGMA**2 * (1 - np.exp(-2 * THETA * ks**2 * dt)) / (2 * THETA * ks**4)

    np.testing.assert_allclose(
        reference_integrator.transition_variances(dt),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("dt", [0.001, 0.01, 0.1, 1.0])
def test_transition_variances_arbitrary_domain(integrator, dt):
    model = integrator.model
    ks = np.arange(1, model.num_modes + 1)
    length = model.domain.length

    decay = model.theta * (ks * np.pi / length) ** 2
    q = 1 / ks**2

    expected = model.sigma**2 * q * (1 - np.exp(-2 * decay * dt)) / (2 * decay)

    np.testing.assert_allclose(
        integrator.transition_variances(dt),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_variances_nonnegative(integrator, dt):
    assert np.all(integrator.transition_variances(dt) >= 0)


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_variances_finite(integrator, dt):
    assert np.all(np.isfinite(integrator.transition_variances(dt)))


def test_transition_variances_increase_with_time(integrator):
    early = integrator.transition_variances(0.01)
    late = integrator.transition_variances(1.0)

    assert np.all(late >= early)


def test_transition_variances_stationary_upper_bound(
    integrator,
):
    model = integrator.model

    stationary = -(model.diffusion_eigenvalues**2) / (2 * model.drift_eigenvalues)

    result = integrator.transition_variances(DT)

    assert np.all(result <= stationary * (1 + RTOL))


@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_transition_variances_theta_dependence(theta):
    integrator = ExactLinear(make_model(theta=theta))
    ks = np.arange(1, NUM_MODES + 1)

    expected = SIGMA**2 * (1 - np.exp(-2 * theta * ks**2 * DT)) / (2 * theta * ks**4)

    np.testing.assert_allclose(
        integrator.transition_variances(DT),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("factor", [0, 0.25, 0.5, 2, 4])
def test_transition_variances_sigma_scaling(factor):
    integrator1 = ExactLinear(make_model(sigma=SIGMA))
    integrator2 = ExactLinear(make_model(sigma=factor * SIGMA))

    np.testing.assert_allclose(
        integrator2.transition_variances(DT),
        factor**2 * integrator1.transition_variances(DT),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("factor", [0, 0.25, 0.5, 2, 4])
def test_transition_variances_covariance_scaling(factor):
    integrator1 = ExactLinear(make_model(eigenvalues=lambda k: 1 / k**2))
    integrator2 = ExactLinear(make_model(eigenvalues=lambda k: factor / k**2))

    np.testing.assert_allclose(
        integrator2.transition_variances(DT),
        factor * integrator1.transition_variances(DT),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("dt", [1e-12, 1e-16, 1e-20])
def test_transition_variances_small_time(reference_integrator, dt):
    ks = np.arange(1, NUM_MODES + 1)
    expected = SIGMA**2 / ks**2

    np.testing.assert_allclose(
        reference_integrator.transition_variances(dt) / dt,
        expected,
        atol=0,
        rtol=1e-9,
    )


def test_transition_variances_stationary_limit(
    reference_integrator,
):
    ks = np.arange(1, NUM_MODES + 1)
    expected = SIGMA**2 / (2 * THETA * ks**4)

    np.testing.assert_allclose(
        reference_integrator.transition_variances(1e6),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_transition_variances_white_noise():
    integrator = ExactLinear(make_model(eigenvalues=lambda k: 1.0))
    ks = np.arange(1, NUM_MODES + 1)

    expected = SIGMA**2 * (1 - np.exp(-2 * THETA * ks**2 * DT)) / (2 * THETA * ks**2)

    np.testing.assert_allclose(
        integrator.transition_variances(DT),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_transition_variances_zero_sigma():
    integrator = ExactLinear(make_model(sigma=0))

    np.testing.assert_array_equal(
        integrator.transition_variances(DT),
        np.zeros(NUM_MODES),
    )


def test_transition_variances_zero_covariance():
    integrator = ExactLinear(make_model(eigenvalues=lambda k: 0.0))

    np.testing.assert_array_equal(
        integrator.transition_variances(DT),
        np.zeros(NUM_MODES),
    )


@pytest.mark.parametrize("rank", [1, 2, 3, 5])
def test_transition_variances_finite_rank(rank):
    integrator = ExactLinear(
        make_model(eigenvalues=lambda k: 1 / k**2 if k <= rank else 0.0)
    )

    result = integrator.transition_variances(DT)

    assert np.all(result[:rank] > 0)

    np.testing.assert_array_equal(
        result[rank:],
        np.zeros(NUM_MODES - rank),
    )


def test_transition_variances_nonconsecutive_zero_modes():
    integrator = ExactLinear(make_model(eigenvalues=lambda k: 1.0 if k % 2 else 0.0))

    result = integrator.transition_variances(DT)

    assert np.all(result[::2] > 0)

    np.testing.assert_array_equal(
        result[1::2],
        np.zeros(NUM_MODES // 2),
    )


def test_transition_variances_nonmonotonic_covariance():
    q = np.array([1.0, 3.0, 0.0, 2.0, 0.5])

    integrator = ExactLinear(make_model(eigenvalues=lambda k: q[k - 1]))

    ks = np.arange(1, NUM_MODES + 1)

    expected = (
        SIGMA**2 * q * (1 - np.exp(-2 * THETA * ks**2 * DT)) / (2 * THETA * ks**2)
    )

    np.testing.assert_allclose(
        integrator.transition_variances(DT),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_matrix_shape(integrator, dt):
    result = integrator.transition_matrix(dt)
    n = integrator.model.num_modes

    assert result.shape == (n, n)
    assert result.dtype == np.float64


def test_transition_matrix_known(reference_integrator):
    ks = np.arange(1, NUM_MODES + 1)
    expected = np.diag(np.exp(-THETA * ks**2 * DT))

    np.testing.assert_allclose(
        reference_integrator.transition_matrix(DT),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_transition_matrix_diagonal(integrator):
    result = integrator.transition_matrix(DT)

    np.testing.assert_array_equal(
        result,
        np.diag(np.diag(result)),
    )


def test_transition_matrix_symmetric(integrator):
    result = integrator.transition_matrix(DT)

    np.testing.assert_array_equal(result, result.T)


def test_transition_matrix_eigenvalue_consistency(integrator):
    np.testing.assert_array_equal(
        np.diag(integrator.transition_matrix(DT)),
        integrator.transition_eigenvalues(DT),
    )


def test_transition_matrix_positive_semidefinite(integrator):
    eigenvalues = np.linalg.eigvalsh(integrator.transition_matrix(DT))

    assert np.all(eigenvalues >= 0)


@pytest.mark.parametrize("dt", VALID_DT)
def test_transition_covariance_matrix_shape(integrator, dt):
    result = integrator.transition_covariance_matrix(dt)
    n = integrator.model.num_modes

    assert result.shape == (n, n)
    assert result.dtype == np.float64


def test_transition_covariance_matrix_known(
    reference_integrator,
):
    ks = np.arange(1, NUM_MODES + 1)

    expected = np.diag(
        SIGMA**2 * (1 - np.exp(-2 * THETA * ks**2 * DT)) / (2 * THETA * ks**4)
    )

    np.testing.assert_allclose(
        reference_integrator.transition_covariance_matrix(DT),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_transition_covariance_matrix_diagonal(integrator):
    result = integrator.transition_covariance_matrix(DT)

    np.testing.assert_array_equal(
        result,
        np.diag(np.diag(result)),
    )


def test_transition_covariance_matrix_symmetric(integrator):
    result = integrator.transition_covariance_matrix(DT)

    np.testing.assert_array_equal(result, result.T)


def test_transition_covariance_matrix_positive_semidefinite(
    integrator,
):
    eigenvalues = np.linalg.eigvalsh(integrator.transition_covariance_matrix(DT))

    assert np.all(eigenvalues >= 0)


def test_transition_covariance_matrix_variance_consistency(
    integrator,
):
    np.testing.assert_array_equal(
        np.diag(integrator.transition_covariance_matrix(DT)),
        integrator.transition_variances(DT),
    )


def test_transition_covariance_matrix_trace(integrator):
    expected = np.sum(integrator.transition_variances(DT))

    assert np.trace(integrator.transition_covariance_matrix(DT)) == pytest.approx(
        expected, abs=ATOL, rel=RTOL
    )


@pytest.mark.parametrize(
    "s, t",
    [
        (0.001, 0.01),
        (0.01, 0.1),
        (0.1, 0.1),
        (0.1, 0.5),
        (0.5, 1.0),
        (1.0, 10.0),
    ],
)
def test_transition_eigenvalues_semigroup(integrator, s, t):
    np.testing.assert_allclose(
        integrator.transition_eigenvalues(s + t),
        integrator.transition_eigenvalues(s) * integrator.transition_eigenvalues(t),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "s, t",
    [
        (0.001, 0.01),
        (0.01, 0.1),
        (0.1, 0.1),
        (0.1, 0.5),
        (0.5, 1.0),
        (1.0, 10.0),
    ],
)
def test_transition_matrix_semigroup(integrator, s, t):
    np.testing.assert_allclose(
        integrator.transition_matrix(s + t),
        integrator.transition_matrix(t) @ integrator.transition_matrix(s),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "s, t",
    [
        (0.001, 0.01),
        (0.01, 0.1),
        (0.1, 0.1),
        (0.1, 0.5),
        (0.5, 1.0),
        (1.0, 10.0),
    ],
)
def test_transition_variances_composition(integrator, s, t):
    expected = integrator.transition_eigenvalues(
        t
    ) ** 2 * integrator.transition_variances(s) + integrator.transition_variances(t)

    np.testing.assert_allclose(
        integrator.transition_variances(s + t),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "s, t",
    [
        (0.001, 0.01),
        (0.01, 0.1),
        (0.1, 0.1),
        (0.1, 0.5),
        (0.5, 1.0),
        (1.0, 10.0),
    ],
)
def test_transition_covariance_composition(integrator, s, t):
    f = integrator.transition_matrix(t)
    q_s = integrator.transition_covariance_matrix(s)
    q_t = integrator.transition_covariance_matrix(t)

    expected = f @ q_s @ f.T + q_t

    np.testing.assert_allclose(
        integrator.transition_covariance_matrix(s + t),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("num_steps", [2, 5, 10, 100])
def test_transition_repeated_composition(reference_integrator, num_steps):
    f = reference_integrator.transition_matrix(DT)
    q = reference_integrator.transition_covariance_matrix(DT)

    accumulated_f = np.eye(NUM_MODES)
    accumulated_q = np.zeros((NUM_MODES, NUM_MODES))

    for _ in range(num_steps):
        accumulated_f = f @ accumulated_f
        accumulated_q = f @ accumulated_q @ f.T + q

    np.testing.assert_allclose(
        accumulated_f,
        reference_integrator.transition_matrix(num_steps * DT),
        atol=ATOL,
        rtol=RTOL,
    )

    np.testing.assert_allclose(
        accumulated_q,
        reference_integrator.transition_covariance_matrix(num_steps * DT),
        atol=ATOL,
        rtol=RTOL,
    )


def test_stationary_covariance_invariance(integrator):
    model = integrator.model

    stationary = np.diag(
        -(model.diffusion_eigenvalues**2) / (2 * model.drift_eigenvalues)
    )

    f = integrator.transition_matrix(DT)
    q = integrator.transition_covariance_matrix(DT)

    np.testing.assert_allclose(
        f @ stationary @ f.T + q,
        stationary,
        atol=ATOL,
        rtol=RTOL,
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
def test_step_shape_and_dtype(integrator, coeffs):
    rng = np.random.default_rng(SEED)
    result = integrator.step(coeffs, DT, rng)

    assert isinstance(result, np.ndarray)
    assert result.shape == (integrator.model.num_modes,)
    assert result.dtype == np.float64


def test_step_known(reference_integrator):
    coeffs = np.array([1, -2, 3, -4, 5], dtype=np.float64)
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    ks = np.arange(1, NUM_MODES + 1)

    f = np.exp(-THETA * ks**2 * DT)

    v = SIGMA**2 * (1 - np.exp(-2 * THETA * ks**2 * DT)) / (2 * THETA * ks**4)

    expected = f * coeffs + np.sqrt(v) * rng2.standard_normal(NUM_MODES)

    np.testing.assert_allclose(
        reference_integrator.step(coeffs, DT, rng1),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_step_reproducibility(integrator):
    coeffs = np.arange(1, NUM_MODES + 1)
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    np.testing.assert_array_equal(
        integrator.step(coeffs, DT, rng1),
        integrator.step(coeffs, DT, rng2),
    )


def test_step_different_seeds(reference_integrator):
    coeffs = np.ones(NUM_MODES)
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED + 1)

    result1 = reference_integrator.step(coeffs, DT, rng1)
    result2 = reference_integrator.step(coeffs, DT, rng2)

    assert not np.array_equal(result1, result2)


def test_step_advances_rng(reference_integrator):
    coeffs = np.ones(NUM_MODES)
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    reference_integrator.step(coeffs, DT, rng1)
    rng2.standard_normal(NUM_MODES)

    np.testing.assert_array_equal(
        rng1.standard_normal(NUM_MODES),
        rng2.standard_normal(NUM_MODES),
    )


def test_step_consecutive_samples(reference_integrator):
    coeffs = np.zeros(NUM_MODES)
    rng = np.random.default_rng(SEED)

    result1 = reference_integrator.step(coeffs, DT, rng)
    result2 = reference_integrator.step(coeffs, DT, rng)

    assert not np.array_equal(result1, result2)


def test_step_no_input_modification(integrator):
    rng = np.random.default_rng(SEED)
    coeffs = rng.standard_normal(NUM_MODES)
    original = coeffs.copy()

    integrator.step(coeffs, DT, rng)

    np.testing.assert_array_equal(coeffs, original)


def test_step_no_memory_sharing(integrator):
    rng = np.random.default_rng(SEED)
    coeffs = np.ones(NUM_MODES)

    result = integrator.step(coeffs, DT, rng)

    assert not np.shares_memory(result, coeffs)


def test_step_deterministic_limit():
    integrator = ExactLinear(make_model(sigma=0))
    coeffs = np.arange(1, NUM_MODES + 1)
    ks = np.arange(1, NUM_MODES + 1)

    expected = np.exp(-THETA * ks**2 * DT) * coeffs

    np.testing.assert_allclose(
        integrator.step(coeffs, DT, np.random.default_rng(SEED)),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_step_deterministic_independent_of_seed():
    integrator = ExactLinear(make_model(sigma=0))
    coeffs = np.arange(1, NUM_MODES + 1)

    np.testing.assert_array_equal(
        integrator.step(coeffs, DT, np.random.default_rng(SEED)),
        integrator.step(coeffs, DT, np.random.default_rng(SEED + 1)),
    )


def test_step_deterministic_zero_state():
    integrator = ExactLinear(make_model(sigma=0))

    np.testing.assert_array_equal(
        integrator.step(
            np.zeros(NUM_MODES),
            DT,
            np.random.default_rng(SEED),
        ),
        np.zeros(NUM_MODES),
    )


def test_step_deterministic_semigroup():
    integrator = ExactLinear(make_model(sigma=0))
    coeffs = np.arange(1, NUM_MODES + 1)
    rng = np.random.default_rng(SEED)

    s = 0.1
    t = 0.5

    intermediate = integrator.step(coeffs, s, rng)
    composed = integrator.step(intermediate, t, rng)
    direct = integrator.step(coeffs, s + t, rng)

    np.testing.assert_allclose(
        composed,
        direct,
        atol=ATOL,
        rtol=RTOL,
    )


def test_step_zero_covariance():
    integrator = ExactLinear(make_model(eigenvalues=lambda k: 0.0))

    coeffs = np.arange(1, NUM_MODES + 1)

    expected = integrator.transition_eigenvalues(DT) * coeffs

    np.testing.assert_allclose(
        integrator.step(coeffs, DT, np.random.default_rng(SEED)),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_step_finite_rank_covariance():
    rank = 2

    integrator = ExactLinear(
        make_model(eigenvalues=lambda k: 1.0 if k <= rank else 0.0)
    )

    coeffs = np.arange(1, NUM_MODES + 1)
    expected = integrator.transition_eigenvalues(DT) * coeffs

    result = integrator.step(coeffs, DT, np.random.default_rng(SEED))

    np.testing.assert_array_equal(
        result[rank:],
        expected[rank:],
    )


def test_step_zero_initial_state(reference_integrator):
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    expected = np.sqrt(
        reference_integrator.transition_variances(DT)
    ) * rng2.standard_normal(NUM_MODES)

    np.testing.assert_array_equal(
        reference_integrator.step(np.zeros(NUM_MODES), DT, rng1),
        expected,
    )


def test_step_state_dependence(reference_integrator):
    x = np.arange(1, NUM_MODES + 1)
    y = -x

    result_x = reference_integrator.step(x, DT, np.random.default_rng(SEED))
    result_y = reference_integrator.step(y, DT, np.random.default_rng(SEED))

    expected = reference_integrator.transition_eigenvalues(DT) * (x - y)

    np.testing.assert_allclose(
        result_x - result_y,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_step_does_not_construct_dense_matrices(reference_integrator, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Dense matrix constructed")

    monkeypatch.setattr(ExactLinear, "transition_matrix", fail)
    monkeypatch.setattr(ExactLinear, "transition_covariance_matrix", fail)
    monkeypatch.setattr(
        LinearSpectralModel,
        "drift_matrix",
        property(fail),
    )
    monkeypatch.setattr(
        LinearSpectralModel,
        "diffusion_matrix",
        property(fail),
    )
    monkeypatch.setattr(
        LinearSpectralModel,
        "diffusion_covariance",
        property(fail),
    )

    result = reference_integrator.step(
        np.ones(NUM_MODES),
        DT,
        np.random.default_rng(SEED),
    )

    assert result.shape == (NUM_MODES,)


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
def test_integrator_dimension(num_modes):
    integrator = ExactLinear(make_model(num_modes=num_modes))

    assert integrator.transition_eigenvalues(DT).shape == (num_modes,)
    assert integrator.transition_variances(DT).shape == (num_modes,)
    assert integrator.transition_matrix(DT).shape == (num_modes, num_modes)
    assert integrator.transition_covariance_matrix(DT).shape == (num_modes, num_modes)

    result = integrator.step(
        np.ones(num_modes),
        DT,
        np.random.default_rng(SEED),
    )

    assert result.shape == (num_modes,)


def test_transition_truncation_consistency():
    small = ExactLinear(make_model(num_modes=5))
    large = ExactLinear(make_model(num_modes=10))

    np.testing.assert_array_equal(
        small.transition_eigenvalues(DT),
        large.transition_eigenvalues(DT)[:5],
    )

    np.testing.assert_array_equal(
        small.transition_variances(DT),
        large.transition_variances(DT)[:5],
    )

    np.testing.assert_array_equal(
        small.transition_matrix(DT),
        large.transition_matrix(DT)[:5, :5],
    )

    np.testing.assert_array_equal(
        small.transition_covariance_matrix(DT),
        large.transition_covariance_matrix(DT)[:5, :5],
    )


def test_step_truncation_consistency():
    small = ExactLinear(make_model(num_modes=5))
    large = ExactLinear(make_model(num_modes=10))

    coeffs = np.arange(1, 11, dtype=np.float64)

    result_small = small.step(coeffs[:5], DT, np.random.default_rng(SEED))
    result_large = large.step(coeffs, DT, np.random.default_rng(SEED))

    np.testing.assert_array_equal(
        result_small,
        result_large[:5],
    )


@pytest.mark.parametrize("method", TRANSITION_METHODS)
def test_returned_array_modification(reference_integrator, method):
    transition = getattr(reference_integrator, method)

    expected = transition(DT).copy()
    result = transition(DT)
    result[...] = 999

    np.testing.assert_array_equal(
        transition(DT),
        expected,
    )


@pytest.mark.parametrize("method", TRANSITION_METHODS)
@pytest.mark.parametrize("dt", INVALID_DT_TYPES)
def test_invalid_transition_dt_type(reference_integrator, method, dt):
    with pytest.raises(TypeError):
        getattr(reference_integrator, method)(dt)


@pytest.mark.parametrize("method", TRANSITION_METHODS)
@pytest.mark.parametrize("dt", INVALID_DT_VALUES)
def test_invalid_transition_dt_value(reference_integrator, method, dt):
    with pytest.raises(ValueError):
        getattr(reference_integrator, method)(dt)


@pytest.mark.parametrize("method", TRANSITION_METHODS)
@pytest.mark.parametrize(
    "dt",
    [
        1,
        0.5,
        np.int32(1),
        np.int64(1),
        np.float32(0.5),
        np.float64(0.5),
    ],
)
def test_valid_transition_dt_types(reference_integrator, method, dt):
    transition = getattr(reference_integrator, method)

    np.testing.assert_array_equal(
        transition(dt),
        transition(float(dt)),
    )


@pytest.mark.parametrize("dt", INVALID_DT_TYPES)
def test_step_invalid_dt_type(reference_integrator, dt):
    with pytest.raises(TypeError):
        reference_integrator.step(
            np.ones(NUM_MODES),
            dt,
            np.random.default_rng(SEED),
        )


@pytest.mark.parametrize("dt", INVALID_DT_VALUES)
def test_step_invalid_dt_value(reference_integrator, dt):
    with pytest.raises(ValueError):
        reference_integrator.step(
            np.ones(NUM_MODES),
            dt,
            np.random.default_rng(SEED),
        )


@pytest.mark.parametrize("coeffs", INVALID_COEFF_TYPES)
def test_step_invalid_coeffs_type(reference_integrator, coeffs):
    with pytest.raises(TypeError):
        reference_integrator.step(coeffs, DT, np.random.default_rng(SEED))


@pytest.mark.parametrize("coeffs", INVALID_COEFF_VALUES)
def test_step_invalid_coeffs_value(reference_integrator, coeffs):
    with pytest.raises(ValueError):
        reference_integrator.step(coeffs, DT, np.random.default_rng(SEED))


@pytest.mark.parametrize("rng", INVALID_RNG_TYPES)
def test_step_invalid_rng(reference_integrator, rng):
    with pytest.raises(TypeError):
        reference_integrator.step(np.ones(NUM_MODES), DT, rng)


@pytest.mark.parametrize(
    "bit_generator",
    [
        np.random.PCG64,
        np.random.MT19937,
        np.random.Philox,
        np.random.SFC64,
    ],
)
def test_step_valid_generators(reference_integrator, bit_generator):
    rng = np.random.Generator(bit_generator(SEED))

    result = reference_integrator.step(np.ones(NUM_MODES), DT, rng)

    assert result.shape == (NUM_MODES,)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("dt", INVALID_DT_TYPES + INVALID_DT_VALUES)
def test_invalid_dt_does_not_advance_rng(reference_integrator, dt):
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    with pytest.raises((TypeError, ValueError)):
        reference_integrator.step(np.ones(NUM_MODES), dt, rng1)

    np.testing.assert_array_equal(
        rng1.standard_normal(NUM_MODES),
        rng2.standard_normal(NUM_MODES),
    )


@pytest.mark.parametrize(
    "coeffs",
    INVALID_COEFF_TYPES + INVALID_COEFF_VALUES,
)
def test_invalid_coeffs_do_not_advance_rng(reference_integrator, coeffs):
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    with pytest.raises((TypeError, ValueError)):
        reference_integrator.step(coeffs, DT, rng1)

    np.testing.assert_array_equal(
        rng1.standard_normal(NUM_MODES),
        rng2.standard_normal(NUM_MODES),
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
def test_invalid_covariance_eigenvalue(value, exception):
    integrator = ExactLinear(make_model(eigenvalues=lambda k: 1.0 if k < 3 else value))

    with pytest.raises(exception):
        integrator.transition_variances(DT)

    with pytest.raises(exception):
        integrator.transition_covariance_matrix(DT)


def test_invalid_covariance_does_not_advance_rng():
    integrator = ExactLinear(make_model(eigenvalues=lambda k: 1.0 if k < 3 else -1.0))

    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    with pytest.raises(ValueError):
        integrator.step(np.ones(NUM_MODES), DT, rng1)

    np.testing.assert_array_equal(
        rng1.standard_normal(NUM_MODES),
        rng2.standard_normal(NUM_MODES),
    )


def test_transition_eigenvalues_do_not_evaluate_covariance():
    def eigenvalues(k):
        raise RuntimeError("Covariance evaluated unexpectedly")

    integrator = ExactLinear(make_model(eigenvalues=eigenvalues))

    assert integrator.transition_eigenvalues(DT).shape == (NUM_MODES,)

    assert integrator.transition_matrix(DT).shape == (NUM_MODES, NUM_MODES)


@pytest.mark.parametrize("dt", VALID_DT)
def test_all_zero_drift_eigenvalues(reference_integrator, monkeypatch, dt):
    monkeypatch.setattr(
        LinearSpectralModel,
        "drift_eigenvalues",
        property(lambda self: np.zeros(self.num_modes)),
    )

    g = reference_integrator.model.diffusion_eigenvalues

    with np.errstate(divide="raise", invalid="raise"):
        np.testing.assert_array_equal(
            reference_integrator.transition_eigenvalues(dt),
            np.ones(NUM_MODES),
        )

        np.testing.assert_allclose(
            reference_integrator.transition_variances(dt),
            g**2 * dt,
            atol=ATOL,
            rtol=RTOL,
        )


@pytest.mark.parametrize("dt", VALID_DT)
def test_mixed_zero_and_negative_drift_eigenvalues(
    reference_integrator, monkeypatch, dt
):
    a = np.array([0.0, -1.0, 0.0, -4.0, -9.0])

    monkeypatch.setattr(
        LinearSpectralModel,
        "drift_eigenvalues",
        property(lambda self: a.copy()),
    )

    g = reference_integrator.model.diffusion_eigenvalues

    expected = np.array(
        [
            g[0] ** 2 * dt,
            g[1] ** 2 * np.expm1(-2 * dt) / -2,
            g[2] ** 2 * dt,
            g[3] ** 2 * np.expm1(-8 * dt) / -8,
            g[4] ** 2 * np.expm1(-18 * dt) / -18,
        ]
    )

    with np.errstate(divide="raise", invalid="raise"):
        np.testing.assert_allclose(
            reference_integrator.transition_eigenvalues(dt),
            np.exp(a * dt),
            atol=ATOL,
            rtol=RTOL,
        )

        np.testing.assert_allclose(
            reference_integrator.transition_variances(dt),
            expected,
            atol=0,
            rtol=RTOL,
        )


def test_zero_drift_zero_diffusion(monkeypatch):
    integrator = ExactLinear(make_model(sigma=0))

    monkeypatch.setattr(
        LinearSpectralModel,
        "drift_eigenvalues",
        property(lambda self: np.zeros(self.num_modes)),
    )

    coeffs = np.arange(1, NUM_MODES + 1)

    with np.errstate(divide="raise", invalid="raise"):
        np.testing.assert_array_equal(
            integrator.transition_variances(DT),
            np.zeros(NUM_MODES),
        )

        np.testing.assert_array_equal(
            integrator.step(
                coeffs,
                DT,
                np.random.default_rng(SEED),
            ),
            coeffs,
        )


def test_zero_drift_step(reference_integrator, monkeypatch):
    monkeypatch.setattr(
        LinearSpectralModel,
        "drift_eigenvalues",
        property(lambda self: np.zeros(self.num_modes)),
    )

    coeffs = np.arange(1, NUM_MODES + 1)
    g = reference_integrator.model.diffusion_eigenvalues

    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    expected = coeffs + np.sqrt(DT) * g * rng2.standard_normal(NUM_MODES)

    np.testing.assert_allclose(
        reference_integrator.step(coeffs, DT, rng1),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_zero_drift_covariance_composition(reference_integrator, monkeypatch):
    monkeypatch.setattr(
        LinearSpectralModel,
        "drift_eigenvalues",
        property(lambda self: np.zeros(self.num_modes)),
    )

    s = 0.1
    t = 0.5

    np.testing.assert_allclose(
        reference_integrator.transition_covariance_matrix(s + t),
        reference_integrator.transition_covariance_matrix(s)
        + reference_integrator.transition_covariance_matrix(t),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.fixture(scope="module")
def conditional_samples():
    integrator = ExactLinear(make_model())
    rng = np.random.default_rng(SEED)

    coeffs = np.array([1, -2, 3, -4, 5], dtype=np.float64)

    samples = np.empty((NUM_SAMPLES, NUM_MODES))

    for i in range(NUM_SAMPLES):
        samples[i] = integrator.step(coeffs, DT, rng)

    ks = np.arange(1, NUM_MODES + 1)

    mean = np.exp(-THETA * ks**2 * DT) * coeffs

    variances = SIGMA**2 * (1 - np.exp(-2 * THETA * ks**2 * DT)) / (2 * THETA * ks**4)

    return samples, mean, variances


def test_step_conditional_mean(conditional_samples):
    samples, expected_mean, variances = conditional_samples

    empirical_mean = np.mean(samples, axis=0)

    standard_error = np.sqrt(variances / NUM_SAMPLES)

    assert np.all(np.abs(empirical_mean - expected_mean) <= NUM_STD * standard_error)


def test_step_conditional_variance(conditional_samples):
    samples, _, expected_variances = conditional_samples

    empirical_variances = np.var(samples, axis=0, ddof=1)

    standard_error = expected_variances * np.sqrt(2 / (NUM_SAMPLES - 1))

    assert np.all(
        np.abs(empirical_variances - expected_variances) <= NUM_STD * standard_error
    )


def test_step_conditional_covariance(conditional_samples):
    samples, _, variances = conditional_samples

    empirical_covariance = np.cov(samples, rowvar=False, ddof=1)

    expected_covariance = np.diag(variances)

    standard_error = np.sqrt(np.outer(variances, variances) / (NUM_SAMPLES - 1))

    diagonal = np.diag_indices(NUM_MODES)
    standard_error[diagonal] *= np.sqrt(2)

    assert np.all(
        np.abs(empirical_covariance - expected_covariance) <= NUM_STD * standard_error
    )


def test_step_cross_mode_correlations(conditional_samples):
    samples, _, _ = conditional_samples

    correlation = np.corrcoef(samples, rowvar=False)

    off_diagonal = ~np.eye(NUM_MODES, dtype=bool)

    tolerance = NUM_STD / np.sqrt(NUM_SAMPLES - 1)

    assert np.all(np.abs(correlation[off_diagonal]) <= tolerance)


def test_step_standardized_third_moment(conditional_samples):
    samples, mean, variances = conditional_samples

    standardized = (samples - mean) / np.sqrt(variances)

    third_moment = np.mean(standardized**3, axis=0)

    tolerance = NUM_STD * np.sqrt(15 / NUM_SAMPLES)

    assert np.all(np.abs(third_moment) <= tolerance)


def test_step_standardized_fourth_moment(conditional_samples):
    samples, mean, variances = conditional_samples

    standardized = (samples - mean) / np.sqrt(variances)

    fourth_moment = np.mean(standardized**4, axis=0)

    tolerance = NUM_STD * np.sqrt(96 / NUM_SAMPLES)

    assert np.all(np.abs(fourth_moment - 3) <= tolerance)


def test_step_temporal_innovation_independence(
    reference_integrator,
):
    rng = np.random.default_rng(SEED)
    coeffs = np.ones(NUM_MODES)

    f = reference_integrator.transition_eigenvalues(DT)
    v = reference_integrator.transition_variances(DT)

    innovations = np.empty((NUM_SAMPLES, NUM_MODES))

    for i in range(NUM_SAMPLES):
        next_coeffs = reference_integrator.step(coeffs, DT, rng)

        innovations[i] = (next_coeffs - f * coeffs) / np.sqrt(v)

        coeffs = next_coeffs

    for lag in [1, 2, 5, 10]:
        left = innovations[:-lag]
        right = innovations[lag:]

        left = left - np.mean(left, axis=0)
        right = right - np.mean(right, axis=0)

        correlations = (
            left.T
            @ right
            / np.sqrt(
                np.outer(
                    np.sum(left**2, axis=0),
                    np.sum(right**2, axis=0),
                )
            )
        )

        tolerance = NUM_STD / np.sqrt(NUM_SAMPLES - lag)

        assert np.all(np.abs(correlations) <= tolerance)


def test_step_stationary_distribution(reference_integrator):
    rng = np.random.default_rng(SEED)

    ks = np.arange(1, NUM_MODES + 1)
    stationary_variances = SIGMA**2 / (2 * THETA * ks**4)

    samples = np.empty((NUM_SAMPLES, NUM_MODES))

    for i in range(NUM_SAMPLES):
        initial = np.sqrt(stationary_variances) * rng.standard_normal(NUM_MODES)

        samples[i] = reference_integrator.step(initial, DT, rng)

    empirical_mean = np.mean(samples, axis=0)
    empirical_variance = np.var(samples, axis=0, ddof=1)

    mean_error = np.sqrt(stationary_variances / NUM_SAMPLES)

    variance_error = stationary_variances * np.sqrt(2 / (NUM_SAMPLES - 1))

    assert np.all(np.abs(empirical_mean) <= NUM_STD * mean_error)

    assert np.all(
        np.abs(empirical_variance - stationary_variances) <= NUM_STD * variance_error
    )


def test_step_multistep_conditional_distribution(
    reference_integrator,
):
    rng = np.random.default_rng(SEED)

    initial = np.arange(1, NUM_MODES + 1, dtype=np.float64)

    s = 0.1
    t = 0.5
    total_time = s + t

    samples = np.empty((NUM_SAMPLES, NUM_MODES))

    for i in range(NUM_SAMPLES):
        intermediate = reference_integrator.step(initial, s, rng)

        samples[i] = reference_integrator.step(intermediate, t, rng)

    ks = np.arange(1, NUM_MODES + 1)

    expected_mean = np.exp(-THETA * ks**2 * total_time) * initial

    expected_variances = (
        SIGMA**2 * (1 - np.exp(-2 * THETA * ks**2 * total_time)) / (2 * THETA * ks**4)
    )

    empirical_mean = np.mean(samples, axis=0)
    empirical_variances = np.var(samples, axis=0, ddof=1)

    mean_error = np.sqrt(expected_variances / NUM_SAMPLES)

    variance_error = expected_variances * np.sqrt(2 / (NUM_SAMPLES - 1))

    assert np.all(np.abs(empirical_mean - expected_mean) <= NUM_STD * mean_error)

    assert np.all(
        np.abs(empirical_variances - expected_variances) <= NUM_STD * variance_error
    )
