import numpy as np
import pytest

from spdepy.basis import SineBasis
from spdepy.covariance import SpectralCovariance
from spdepy.domains import Interval
from spdepy.noise import SpectralWienerNoise

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

INVALID_INT_VALUES = [0, -1, -10]

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

INVALID_RNGS = [
    None,
    42,
    "42",
    np.random.RandomState(42),
    np.random.PCG64(42),
    np.random,
]

ATOL = 1e-10
RTOL = 1e-10
SEED = 42
NUM_SAMPLES = 100_000
NUM_STD = 6


@pytest.fixture(params=VALID_INTERVALS)
def basis(request):
    a, b = request.param
    return SineBasis(Interval(a, b))


@pytest.fixture
def covariance(basis):
    return SpectralCovariance(basis, lambda k: 1 / k**2)


@pytest.fixture
def noise(covariance):
    return SpectralWienerNoise(covariance)


@pytest.fixture
def statistical_noise():
    basis = SineBasis(Interval(0, np.pi))
    covariance = SpectralCovariance(basis, lambda k: 1 / k**2)
    return SpectralWienerNoise(covariance)


def test_construction(covariance):
    noise = SpectralWienerNoise(covariance)

    assert noise.covariance is covariance
    assert noise.basis is covariance.basis


@pytest.mark.parametrize(
    "covariance",
    [
        None,
        1,
        "covariance",
        lambda k: 1.0,
        np.eye(3),
        SineBasis(Interval(0, 1)),
    ],
)
def test_invalid_covariance(covariance):
    with pytest.raises(TypeError):
        SpectralWienerNoise(covariance)


def test_covariance_read_only(noise, covariance):
    with pytest.raises(AttributeError):
        noise.covariance = covariance


def test_basis_read_only(noise, basis):
    with pytest.raises(AttributeError):
        noise.basis = basis


def test_basis_identity(noise, covariance):
    assert noise.basis is covariance.basis


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10, 100])
@pytest.mark.parametrize("dt", [0.001, 0.1, 1.0, 2.0])
def test_increment_shape(noise, num_modes, dt):
    rng = np.random.default_rng(SEED)
    result = noise.increment(dt, num_modes, rng)

    assert isinstance(result, np.ndarray)
    assert result.shape == (num_modes,)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("num_modes", [1, 2, 5, 10])
@pytest.mark.parametrize("num_steps", [1, 2, 10, 100])
def test_increments_shape(noise, num_modes, num_steps):
    rng = np.random.default_rng(SEED)
    result = noise.increments(0.1, num_modes, num_steps, rng)

    assert isinstance(result, np.ndarray)
    assert result.shape == (num_steps, num_modes)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "dt",
    [
        1,
        0.1,
        np.int64(1),
        np.float32(0.1),
        np.float64(0.1),
    ],
)
def test_valid_dt_types(noise, dt):
    rng = np.random.default_rng(SEED)

    assert noise.increment(dt, 5, rng).shape == (5,)
    assert noise.increments(dt, 5, 10, rng).shape == (10, 5)


def test_numpy_integer_dimensions(noise):
    rng = np.random.default_rng(SEED)

    assert noise.increment(0.1, np.int64(5), rng).shape == (5,)

    assert noise.increments(0.1, np.int64(5), np.int64(10), rng).shape == (10, 5)


@pytest.mark.parametrize("dt", [0.001, 0.1, 1.0, 2.0])
@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_increment_known_samples(noise, dt, num_modes):
    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    ks = np.arange(1, num_modes + 1, dtype=np.float64)
    expected = np.sqrt(dt) / ks * reference_rng.standard_normal(num_modes)

    np.testing.assert_allclose(
        noise.increment(dt, num_modes, rng),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("dt", [0.001, 0.1, 1.0, 2.0])
@pytest.mark.parametrize("num_modes", [1, 5, 10])
@pytest.mark.parametrize("num_steps", [1, 5, 10])
def test_increments_known_samples(noise, dt, num_modes, num_steps):
    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    ks = np.arange(1, num_modes + 1, dtype=np.float64)
    expected = np.sqrt(dt) / ks * reference_rng.standard_normal((num_steps, num_modes))

    np.testing.assert_allclose(
        noise.increments(dt, num_modes, num_steps, rng),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("seed", [0, 1, 42, 123])
def test_increment_reproducibility(noise, seed):
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    np.testing.assert_array_equal(
        noise.increment(0.1, 10, rng1),
        noise.increment(0.1, 10, rng2),
    )


@pytest.mark.parametrize("seed", [0, 1, 42, 123])
def test_increments_reproducibility(noise, seed):
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    np.testing.assert_array_equal(
        noise.increments(0.1, 10, 20, rng1),
        noise.increments(0.1, 10, 20, rng2),
    )


def test_increment_different_seeds(noise):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(123)

    assert not np.array_equal(
        noise.increment(0.1, 10, rng1),
        noise.increment(0.1, 10, rng2),
    )


def test_increments_different_seeds(noise):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(123)

    assert not np.array_equal(
        noise.increments(0.1, 10, 20, rng1),
        noise.increments(0.1, 10, 20, rng2),
    )


def test_increment_advances_rng(noise):
    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    noise.increment(0.1, 10, rng)
    reference_rng.standard_normal(10)

    np.testing.assert_array_equal(
        rng.standard_normal(10),
        reference_rng.standard_normal(10),
    )


def test_increments_advances_rng(noise):
    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    noise.increments(0.1, 10, 20, rng)
    reference_rng.standard_normal((20, 10))

    np.testing.assert_array_equal(
        rng.standard_normal(10),
        reference_rng.standard_normal(10),
    )


def test_successive_increments_differ(noise):
    rng = np.random.default_rng(SEED)

    first = noise.increment(0.1, 10, rng)
    second = noise.increment(0.1, 10, rng)

    assert not np.array_equal(first, second)


def test_successive_batches_differ(noise):
    rng = np.random.default_rng(SEED)

    first = noise.increments(0.1, 10, 20, rng)
    second = noise.increments(0.1, 10, 20, rng)

    assert not np.array_equal(first, second)


@pytest.mark.parametrize("num_modes", [1, 5, 10])
@pytest.mark.parametrize("num_steps", [1, 5, 10])
def test_batch_single_increment_consistency(noise, num_modes, num_steps):
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    batched = noise.increments(0.1, num_modes, num_steps, rng1)

    individual = np.stack(
        [noise.increment(0.1, num_modes, rng2) for _ in range(num_steps)]
    )

    np.testing.assert_array_equal(batched, individual)


def test_single_step_batch(noise):
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    np.testing.assert_array_equal(
        noise.increment(0.1, 10, rng1),
        noise.increments(0.1, 10, 1, rng2)[0],
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_white_noise_increment(basis, num_modes):
    covariance = SpectralCovariance(basis, lambda k: 1.0)
    noise = SpectralWienerNoise(covariance)

    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    expected = np.sqrt(0.1) * reference_rng.standard_normal(num_modes)

    np.testing.assert_array_equal(
        noise.increment(0.1, num_modes, rng),
        expected,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_white_noise_increments(basis, num_modes):
    covariance = SpectralCovariance(basis, lambda k: 1.0)
    noise = SpectralWienerNoise(covariance)

    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    expected = np.sqrt(0.1) * reference_rng.standard_normal((20, num_modes))

    np.testing.assert_array_equal(
        noise.increments(0.1, num_modes, 20, rng),
        expected,
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_zero_covariance_increment(basis, num_modes):
    covariance = SpectralCovariance(basis, lambda k: 0.0)
    noise = SpectralWienerNoise(covariance)
    rng = np.random.default_rng(SEED)

    np.testing.assert_array_equal(
        noise.increment(0.1, num_modes, rng),
        np.zeros(num_modes),
    )


@pytest.mark.parametrize("num_modes", [1, 5, 10])
def test_zero_covariance_increments(basis, num_modes):
    covariance = SpectralCovariance(basis, lambda k: 0.0)
    noise = SpectralWienerNoise(covariance)
    rng = np.random.default_rng(SEED)

    np.testing.assert_array_equal(
        noise.increments(0.1, num_modes, 20, rng),
        np.zeros((20, num_modes)),
    )


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_finite_rank_increment(basis, rank):
    covariance = SpectralCovariance(
        basis,
        lambda k: 1 / k**2 if k <= rank else 0.0,
    )
    noise = SpectralWienerNoise(covariance)

    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    expected = (
        np.sqrt(0.1)
        * np.sqrt(covariance.eigenvalues(10))
        * reference_rng.standard_normal(10)
    )

    result = noise.increment(0.1, 10, rng)

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(
        result[rank:],
        np.zeros(10 - rank),
    )


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_finite_rank_increments(basis, rank):
    covariance = SpectralCovariance(
        basis,
        lambda k: 1 / k**2 if k <= rank else 0.0,
    )
    noise = SpectralWienerNoise(covariance)
    rng = np.random.default_rng(SEED)

    result = noise.increments(0.1, 10, 20, rng)

    np.testing.assert_array_equal(
        result[:, rank:],
        np.zeros((20, 10 - rank)),
    )

    assert np.all(np.var(result[:, :rank], axis=0) > 0)


def test_nonconsecutive_zero_modes(basis):
    covariance = SpectralCovariance(
        basis,
        lambda k: 1.0 if k % 2 else 0.0,
    )
    noise = SpectralWienerNoise(covariance)
    rng = np.random.default_rng(SEED)

    result = noise.increments(0.1, 10, 20, rng)

    np.testing.assert_array_equal(
        result[:, 1::2],
        np.zeros((20, 5)),
    )

    assert np.all(np.var(result[:, ::2], axis=0) > 0)


def test_nonmonotonic_covariance(basis):
    values = np.array([1.0, 3.0, 0.0, 2.0, 0.5])
    covariance = SpectralCovariance(basis, lambda k: values[k - 1])
    noise = SpectralWienerNoise(covariance)

    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    expected = np.sqrt(0.1) * np.sqrt(values) * reference_rng.standard_normal((20, 5))

    np.testing.assert_array_equal(
        noise.increments(0.1, 5, 20, rng),
        expected,
    )


@pytest.mark.parametrize("factor", [0.25, 0.5, 2, 4, 10])
def test_increment_time_scaling(noise, factor):
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    first = noise.increment(0.1, 10, rng1)
    second = noise.increment(0.1 * factor, 10, rng2)

    np.testing.assert_allclose(
        second,
        np.sqrt(factor) * first,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("factor", [0.25, 0.5, 2, 4, 10])
def test_increments_time_scaling(noise, factor):
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    first = noise.increments(0.1, 10, 20, rng1)
    second = noise.increments(0.1 * factor, 10, 20, rng2)

    np.testing.assert_allclose(
        second,
        np.sqrt(factor) * first,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize("factor", [0, 0.25, 0.5, 2, 4])
def test_covariance_scaling(basis, factor):
    covariance1 = SpectralCovariance(basis, lambda k: 1 / k**2)
    covariance2 = SpectralCovariance(basis, lambda k: factor / k**2)

    noise1 = SpectralWienerNoise(covariance1)
    noise2 = SpectralWienerNoise(covariance2)

    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)

    first = noise1.increments(0.1, 10, 20, rng1)
    second = noise2.increments(0.1, 10, 20, rng2)

    np.testing.assert_allclose(
        second,
        np.sqrt(factor) * first,
        atol=ATOL,
        rtol=RTOL,
    )


def test_increment_covariance_evaluation_count(basis):
    calls = []

    def eigenvalues(k):
        calls.append(k)
        return 1 / k**2

    covariance = SpectralCovariance(basis, eigenvalues)
    noise = SpectralWienerNoise(covariance)
    rng = np.random.default_rng(SEED)

    noise.increment(0.1, 5, rng)

    assert calls == [1, 2, 3, 4, 5]


def test_increments_covariance_evaluation_count(basis):
    calls = []

    def eigenvalues(k):
        calls.append(k)
        return 1 / k**2

    covariance = SpectralCovariance(basis, eigenvalues)
    noise = SpectralWienerNoise(covariance)
    rng = np.random.default_rng(SEED)

    noise.increments(0.1, 5, 100, rng)

    assert calls == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("dt", [0.001, 0.1, 1.0])
def test_empirical_mean(statistical_noise, dt):
    rng = np.random.default_rng(SEED)
    num_modes = 5

    samples = statistical_noise.increments(dt, num_modes, NUM_SAMPLES, rng)

    variances = dt / np.arange(1, num_modes + 1) ** 2
    standard_errors = np.sqrt(variances / NUM_SAMPLES)
    means = np.mean(samples, axis=0)

    assert np.all(np.abs(means) <= NUM_STD * standard_errors)


@pytest.mark.parametrize("dt", [0.001, 0.1, 1.0])
def test_empirical_variance(statistical_noise, dt):
    rng = np.random.default_rng(SEED)
    num_modes = 5

    samples = statistical_noise.increments(dt, num_modes, NUM_SAMPLES, rng)

    expected = dt / np.arange(1, num_modes + 1) ** 2
    empirical = np.var(samples, axis=0, ddof=1)

    standard_errors = expected * np.sqrt(2 / (NUM_SAMPLES - 1))

    assert np.all(np.abs(empirical - expected) <= NUM_STD * standard_errors)


def test_empirical_covariance_matrix(statistical_noise):
    rng = np.random.default_rng(SEED)
    dt = 0.1
    num_modes = 5

    samples = statistical_noise.increments(dt, num_modes, NUM_SAMPLES, rng)

    variances = dt / np.arange(1, num_modes + 1) ** 2
    expected = np.diag(variances)
    empirical = np.cov(samples, rowvar=False, ddof=1)

    standard_errors = np.sqrt(
        (np.outer(variances, variances) + expected**2) / (NUM_SAMPLES - 1)
    )

    assert np.all(np.abs(empirical - expected) <= NUM_STD * standard_errors)


def test_cross_mode_uncorrelated(statistical_noise):
    rng = np.random.default_rng(SEED)

    samples = statistical_noise.increments(0.1, 5, NUM_SAMPLES, rng)

    correlations = np.corrcoef(samples, rowvar=False)
    off_diagonal = correlations[np.triu_indices(5, k=1)]

    tolerance = NUM_STD / np.sqrt(NUM_SAMPLES - 3)

    assert np.all(np.abs(off_diagonal) <= tolerance)


@pytest.mark.parametrize("lag", [1, 2, 5, 10])
def test_temporal_autocorrelation(statistical_noise, lag):
    rng = np.random.default_rng(SEED)

    samples = statistical_noise.increments(0.1, 5, NUM_SAMPLES, rng)

    for k in range(samples.shape[1]):
        correlation = np.corrcoef(
            samples[:-lag, k],
            samples[lag:, k],
        )[0, 1]

        tolerance = NUM_STD / np.sqrt(NUM_SAMPLES - lag)

        assert abs(correlation) <= tolerance


def test_cross_mode_temporal_correlation(statistical_noise):
    rng = np.random.default_rng(SEED)
    num_modes = 5

    samples = statistical_noise.increments(0.1, num_modes, NUM_SAMPLES, rng)

    correlations = np.corrcoef(
        samples[:-1].T,
        samples[1:].T,
    )

    cross_correlations = correlations[:num_modes, num_modes:]

    tolerance = NUM_STD / np.sqrt(NUM_SAMPLES - 1)

    assert np.all(np.abs(cross_correlations) <= tolerance)


def test_gaussian_third_moment(statistical_noise):
    rng = np.random.default_rng(SEED)
    dt = 0.1
    num_modes = 5

    samples = statistical_noise.increments(dt, num_modes, NUM_SAMPLES, rng)

    variances = dt / np.arange(1, num_modes + 1) ** 2
    standardized = samples / np.sqrt(variances)

    empirical = np.mean(standardized**3, axis=0)
    standard_error = np.sqrt(15 / NUM_SAMPLES)

    assert np.all(np.abs(empirical) <= NUM_STD * standard_error)


def test_gaussian_fourth_moment(statistical_noise):
    rng = np.random.default_rng(SEED)
    dt = 0.1
    num_modes = 5

    samples = statistical_noise.increments(dt, num_modes, NUM_SAMPLES, rng)

    variances = dt / np.arange(1, num_modes + 1) ** 2
    standardized = samples / np.sqrt(variances)

    empirical = np.mean(standardized**4, axis=0)
    standard_error = np.sqrt(96 / NUM_SAMPLES)

    assert np.all(np.abs(empirical - 3) <= NUM_STD * standard_error)


@pytest.mark.parametrize("group_size", [2, 4, 10])
def test_brownian_increment_aggregation(statistical_noise, group_size):
    rng = np.random.default_rng(SEED)
    dt = 0.01
    num_modes = 5

    fine = statistical_noise.increments(
        dt,
        num_modes,
        NUM_SAMPLES * group_size,
        rng,
    )

    coarse = fine.reshape(NUM_SAMPLES, group_size, num_modes).sum(axis=1)

    expected_variances = group_size * dt / np.arange(1, num_modes + 1) ** 2

    means = np.mean(coarse, axis=0)
    variances = np.var(coarse, axis=0, ddof=1)

    mean_errors = np.sqrt(expected_variances / NUM_SAMPLES)

    variance_errors = expected_variances * np.sqrt(2 / (NUM_SAMPLES - 1))

    assert np.all(np.abs(means) <= NUM_STD * mean_errors)

    assert np.all(np.abs(variances - expected_variances) <= NUM_STD * variance_errors)


@pytest.mark.parametrize("method", ["increment", "increments"])
@pytest.mark.parametrize("dt", INVALID_DT_TYPES)
def test_invalid_dt_type(statistical_noise, method, dt):
    rng = np.random.default_rng(SEED)

    with pytest.raises(TypeError):
        if method == "increment":
            statistical_noise.increment(dt, 5, rng)
        else:
            statistical_noise.increments(dt, 5, 10, rng)


@pytest.mark.parametrize("method", ["increment", "increments"])
@pytest.mark.parametrize("dt", INVALID_DT_VALUES)
def test_invalid_dt_value(statistical_noise, method, dt):
    rng = np.random.default_rng(SEED)

    with pytest.raises(ValueError):
        if method == "increment":
            statistical_noise.increment(dt, 5, rng)
        else:
            statistical_noise.increments(dt, 5, 10, rng)


@pytest.mark.parametrize("method", ["increment", "increments"])
@pytest.mark.parametrize("num_modes", INVALID_INT_TYPES)
def test_invalid_num_modes_type(statistical_noise, method, num_modes):
    rng = np.random.default_rng(SEED)

    with pytest.raises(TypeError):
        if method == "increment":
            statistical_noise.increment(0.1, num_modes, rng)
        else:
            statistical_noise.increments(0.1, num_modes, 10, rng)


@pytest.mark.parametrize("method", ["increment", "increments"])
@pytest.mark.parametrize("num_modes", INVALID_INT_VALUES)
def test_invalid_num_modes_value(statistical_noise, method, num_modes):
    rng = np.random.default_rng(SEED)

    with pytest.raises(ValueError):
        if method == "increment":
            statistical_noise.increment(0.1, num_modes, rng)
        else:
            statistical_noise.increments(0.1, num_modes, 10, rng)


@pytest.mark.parametrize("num_steps", INVALID_INT_TYPES)
def test_invalid_num_steps_type(statistical_noise, num_steps):
    rng = np.random.default_rng(SEED)

    with pytest.raises(TypeError):
        statistical_noise.increments(0.1, 5, num_steps, rng)


@pytest.mark.parametrize("num_steps", INVALID_INT_VALUES)
def test_invalid_num_steps_value(statistical_noise, num_steps):
    rng = np.random.default_rng(SEED)

    with pytest.raises(ValueError):
        statistical_noise.increments(0.1, 5, num_steps, rng)


@pytest.mark.parametrize("rng", INVALID_RNGS)
def test_increment_invalid_rng(statistical_noise, rng):
    with pytest.raises(TypeError):
        statistical_noise.increment(0.1, 5, rng)


@pytest.mark.parametrize("rng", INVALID_RNGS)
def test_increments_invalid_rng(statistical_noise, rng):
    with pytest.raises(TypeError):
        statistical_noise.increments(0.1, 5, 10, rng)


@pytest.mark.parametrize(
    "dt, num_modes, exception",
    [
        (0, 5, ValueError),
        (-1, 5, ValueError),
        (np.nan, 5, ValueError),
        (True, 5, TypeError),
        (0.1, 0, ValueError),
        (0.1, -1, ValueError),
        (0.1, 1.5, TypeError),
        (0.1, True, TypeError),
    ],
)
def test_increment_invalid_input_preserves_rng(
    statistical_noise, dt, num_modes, exception
):
    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    with pytest.raises(exception):
        statistical_noise.increment(dt, num_modes, rng)

    np.testing.assert_array_equal(
        rng.standard_normal(10),
        reference_rng.standard_normal(10),
    )


@pytest.mark.parametrize(
    "dt, num_modes, num_steps, exception",
    [
        (0, 5, 10, ValueError),
        (-1, 5, 10, ValueError),
        (np.nan, 5, 10, ValueError),
        (True, 5, 10, TypeError),
        (0.1, 0, 10, ValueError),
        (0.1, -1, 10, ValueError),
        (0.1, 1.5, 10, TypeError),
        (0.1, True, 10, TypeError),
        (0.1, 5, 0, ValueError),
        (0.1, 5, -1, ValueError),
        (0.1, 5, 1.5, TypeError),
        (0.1, 5, True, TypeError),
    ],
)
def test_increments_invalid_input_preserves_rng(
    statistical_noise,
    dt,
    num_modes,
    num_steps,
    exception,
):
    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    with pytest.raises(exception):
        statistical_noise.increments(dt, num_modes, num_steps, rng)

    np.testing.assert_array_equal(
        rng.standard_normal(10),
        reference_rng.standard_normal(10),
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
@pytest.mark.parametrize("method", ["increment", "increments"])
def test_invalid_covariance_preserves_rng(value, exception, method):
    basis = SineBasis(Interval(0, np.pi))
    covariance = SpectralCovariance(
        basis,
        lambda k: 1.0 if k < 3 else value,
    )
    noise = SpectralWienerNoise(covariance)

    rng = np.random.default_rng(SEED)
    reference_rng = np.random.default_rng(SEED)

    with pytest.raises(exception):
        if method == "increment":
            noise.increment(0.1, 5, rng)
        else:
            noise.increments(0.1, 5, 10, rng)

    np.testing.assert_array_equal(
        rng.standard_normal(10),
        reference_rng.standard_normal(10),
    )


def test_partial_covariance_validity():
    basis = SineBasis(Interval(0, np.pi))
    covariance = SpectralCovariance(
        basis,
        lambda k: 1.0 if k <= 3 else -1.0,
    )
    noise = SpectralWienerNoise(covariance)
    rng = np.random.default_rng(SEED)

    assert noise.increment(0.1, 3, rng).shape == (3,)
    assert noise.increments(0.1, 3, 10, rng).shape == (10, 3)

    with pytest.raises(ValueError):
        noise.increment(0.1, 4, rng)

    with pytest.raises(ValueError):
        noise.increments(0.1, 4, 10, rng)
