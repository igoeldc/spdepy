from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from spdepy.basis import SineBasis
from spdepy.boundaries import Dirichlet
from spdepy.covariance import SpectralCovariance
from spdepy.domains import Interval
from spdepy.models import LinearSpectralModel
from spdepy.noise import SpectralWienerNoise
from spdepy.operators import Laplacian
from spdepy.simulation import Simulator
from spdepy.temporal import ExactLinear


def make_simulator(
    a=0.0,
    b=np.pi,
    num_modes=5,
    theta=1.0,
    sigma=1.0,
    eigenvalues=lambda k: 1.0,
):
    domain = Interval(a, b)
    basis = SineBasis(domain)
    operator = Laplacian(domain, Dirichlet())
    covariance = SpectralCovariance(basis, eigenvalues)
    noise = SpectralWienerNoise(covariance)
    model = LinearSpectralModel(operator, noise, num_modes, theta, sigma)
    return Simulator(ExactLinear(model))


@pytest.fixture
def simulator():
    return make_simulator()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def x0(simulator):
    return np.arange(1, simulator.num_modes + 1, dtype=np.float64)


def test_integrator_property(simulator):
    assert isinstance(simulator.integrator, ExactLinear)


def test_model_property(simulator):
    assert simulator.model is simulator.integrator.model


def test_num_modes_property(simulator):
    assert simulator.num_modes == simulator.model.num_modes


@pytest.mark.parametrize("attribute", ["integrator", "model", "num_modes"])
def test_read_only_properties(simulator, attribute):
    with pytest.raises(AttributeError):
        setattr(simulator, attribute, None)


@pytest.mark.parametrize(
    "integrator",
    [None, 1, 1.0, True, "integrator", [], {}, object()],
)
def test_invalid_integrator(integrator):
    with pytest.raises(TypeError):
        Simulator(integrator)


def test_integrator_identity(simulator):
    integrator = simulator.integrator
    other = Simulator(integrator)
    assert other.integrator is integrator


@pytest.mark.parametrize(
    "times",
    [
        [0.0],
        [1.0],
        [-1.0],
        [0.0, 1.0],
        [0.0, 0.1, 0.2, 1.0],
        [-2.0, -1.0, 0.0, 2.0],
        (0.0, 0.25, 0.5, 1.0),
        np.linspace(0.0, 1.0, 101),
        np.array([0, 1, 2], dtype=np.int32),
        np.array([0, 1, 2], dtype=np.float32),
        [0.0, 1e-20, 1e-10, 1.0],
    ],
)
def test_valid_explicit_times(simulator, x0, rng, times):
    actual_times, trajectory = simulator.simulate(x0, rng, times=times)

    assert_array_equal(actual_times, np.asarray(times, dtype=np.float64))
    assert actual_times.dtype == np.float64
    assert trajectory.dtype == np.float64
    assert trajectory.shape == (len(times), simulator.num_modes)
    assert_array_equal(trajectory[0], x0)


@pytest.mark.parametrize(
    "times",
    [
        [],
        (),
        np.array([]),
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0, 0.5],
        [0.0, 1.0, 1.0],
        [np.nan],
        [np.inf],
        [-np.inf],
        [0.0, np.nan],
        [0.0, np.inf],
        [[0.0, 1.0]],
        [[0.0], [1.0]],
        1.0,
        np.array(1.0),
    ],
)
def test_invalid_explicit_times(simulator, x0, rng, times):
    with pytest.raises(ValueError):
        simulator.simulate(x0, rng, times=times)


@pytest.mark.parametrize(
    "times",
    [
        "times",
        [True, False],
        [0.0, True],
        [0.0, "1"],
        [0.0, 1j],
        np.array([0, 1], dtype=object),
        np.array([False, True]),
        np.array([0.0, 1.0], dtype=complex),
    ],
)
def test_invalid_explicit_time_types(simulator, x0, rng, times):
    with pytest.raises(TypeError):
        simulator.simulate(x0, rng, times=times)


def test_nonfinite_consecutive_difference(simulator, x0, rng):
    largest = np.finfo(np.float64).max

    with np.errstate(over="ignore"), pytest.raises(ValueError):
        simulator.simulate(x0, rng, times=[-largest, largest])


@pytest.mark.parametrize("time", [-10.0, 0.0, 10.0])
def test_single_time(simulator, x0, rng, time):
    times, trajectory = simulator.simulate(x0, rng, times=[time])

    assert_array_equal(times, [time])
    assert trajectory.shape == (1, simulator.num_modes)
    assert_array_equal(trajectory[0], x0)


def test_single_time_does_not_consume_rng(simulator, x0):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    simulator.simulate(x0, rng1, times=[0.0])

    assert_array_equal(
        rng1.standard_normal(100),
        rng2.standard_normal(100),
    )


@pytest.mark.parametrize(
    "t_span, dt, expected",
    [
        ((0.0, 1.0), 0.25, [0, 0.25, 0.5, 0.75, 1]),
        ((0.0, 1.0), 0.3, [0, 0.3, 0.6, 0.9, 1]),
        ((0.0, 1.0), 1.0, [0, 1]),
        ((0.0, 1.0), 2.0, [0, 1]),
        ((-1.0, 1.0), 0.75, [-1, -0.25, 0.5, 1]),
        ((-2.0, -1.0), 0.25, [-2, -1.75, -1.5, -1.25, -1]),
        ((2.0, 3.0), 0.4, [2, 2.4, 2.8, 3]),
        ((0.0, 0.001), 0.00025, [0, 0.00025, 0.0005, 0.00075, 0.001]),
    ],
)
@pytest.mark.parametrize("use_interval", [False, True])
def test_times_from_dt(simulator, x0, rng, t_span, dt, expected, use_interval):
    a, b = t_span

    if use_interval:
        t_span = Interval(a, b)

    times, trajectory = simulator.simulate(x0, rng, t_span=t_span, dt=dt)

    assert_allclose(times, expected, rtol=1e-14, atol=1e-15)
    assert times[0] == a
    assert times[-1] == b
    assert np.all(np.diff(times) > 0)
    assert trajectory.shape == (len(expected), simulator.num_modes)


def test_dt_preserves_small_positive_final_step(simulator, x0, rng):
    times, _ = simulator.simulate(x0, rng, t_span=(0.0, 0.9), dt=0.3)

    last_full_step = 3 * 0.3

    assert len(times) == 5
    assert times[-2] == last_full_step
    assert times[-1] == 0.9
    assert times[-1] > times[-2]


def test_dt_exact_binary_multiple(simulator, x0, rng):
    times, _ = simulator.simulate(x0, rng, t_span=(0.0, 1.0), dt=0.125)

    assert_array_equal(times, np.arange(9) * 0.125)
    assert_array_equal(np.diff(times), np.full(8, 0.125))


def test_dt_endpoint_rounding_overshoot(simulator, x0, rng):
    a = -0.3
    b = 0.1
    dt = 0.4

    assert a + dt > b

    times, _ = simulator.simulate(x0, rng, t_span=(a, b), dt=dt)

    assert_array_equal(times, [a, b])


@pytest.mark.parametrize("dt", [0, -1, -0.01, np.nan, np.inf, -np.inf])
def test_invalid_dt_values(simulator, x0, rng, dt):
    with pytest.raises(ValueError):
        simulator.simulate(x0, rng, t_span=(0, 1), dt=dt)


@pytest.mark.parametrize("dt", [True, False, "0.1", 1j, [], [0.1], {}])
def test_invalid_dt_types(simulator, x0, rng, dt):
    with pytest.raises(TypeError):
        simulator.simulate(x0, rng, t_span=(0, 1), dt=dt)


@pytest.mark.parametrize(
    "t_span",
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (-2.0, -1.0),
        (2.0, 5.0),
        (0.0, 0.001),
        Interval(0.0, 1.0),
        Interval(-1.0, 1.0),
    ],
)
@pytest.mark.parametrize("num_steps", [1, 2, 3, 10, 100])
def test_times_from_num_steps(simulator, x0, rng, t_span, num_steps):
    a, b = t_span.bounds if isinstance(t_span, Interval) else t_span

    times, trajectory = simulator.simulate(x0, rng, t_span=t_span, num_steps=num_steps)

    assert len(times) == num_steps + 1
    assert times[0] == a
    assert times[-1] == b

    assert_array_equal(times, np.linspace(a, b, num_steps + 1))

    assert_allclose(
        np.diff(times),
        (b - a) / num_steps,
        rtol=1e-12,
        atol=0,
    )

    assert trajectory.shape == (num_steps + 1, simulator.num_modes)


@pytest.mark.parametrize("num_steps", [0, -1, -100])
def test_invalid_num_steps_values(simulator, x0, rng, num_steps):
    with pytest.raises(ValueError):
        simulator.simulate(x0, rng, t_span=(0, 1), num_steps=num_steps)


@pytest.mark.parametrize(
    "num_steps",
    [True, False, 1.0, 2.5, "10", np.nan, np.inf, [], 1j],
)
def test_invalid_num_steps_types(simulator, x0, rng, num_steps):
    with pytest.raises(TypeError):
        simulator.simulate(x0, rng, t_span=(0, 1), num_steps=num_steps)


@pytest.mark.parametrize(
    "t_span",
    [
        (),
        (0,),
        (0, 1, 2),
        (0, 0),
        (1, 0),
        (np.nan, 1),
        (0, np.nan),
        (-np.inf, 1),
        (0, np.inf),
    ],
)
@pytest.mark.parametrize("specification", [{"dt": 0.1}, {"num_steps": 10}])
def test_invalid_t_span_values(simulator, x0, rng, t_span, specification):
    with pytest.raises(ValueError):
        simulator.simulate(x0, rng, t_span=t_span, **specification)


@pytest.mark.parametrize(
    "t_span",
    [
        1,
        "interval",
        [0, 1],
        np.array([0, 1]),
        {},
        (True, 1),
        (0, False),
        ("0", 1),
        (0, 1j),
    ],
)
@pytest.mark.parametrize("specification", [{"dt": 0.1}, {"num_steps": 10}])
def test_invalid_t_span_types(simulator, x0, rng, t_span, specification):
    with pytest.raises(TypeError):
        simulator.simulate(x0, rng, t_span=t_span, **specification)


@pytest.mark.parametrize(
    "has_times, has_t_span, has_dt, has_num_steps",
    list(product([False, True], repeat=4)),
)
def test_all_time_specification_combinations(
    simulator,
    x0,
    rng,
    has_times,
    has_t_span,
    has_dt,
    has_num_steps,
):
    kwargs = {
        "times": [0.0, 0.5, 1.0] if has_times else None,
        "t_span": (0.0, 1.0) if has_t_span else None,
        "dt": 0.5 if has_dt else None,
        "num_steps": 2 if has_num_steps else None,
    }

    valid = (has_times and not has_t_span and not has_dt and not has_num_steps) or (
        not has_times and has_t_span and (has_dt != has_num_steps)
    )

    if valid:
        times, trajectory = simulator.simulate(x0, rng, **kwargs)

        assert_array_equal(times, [0.0, 0.5, 1.0])
        assert trajectory.shape == (3, simulator.num_modes)
    else:
        with pytest.raises(ValueError):
            simulator.simulate(x0, rng, **kwargs)


def test_unrepresentable_num_steps_grid(simulator, x0, rng):
    a = 1.0
    b = np.nextafter(a, np.inf)

    with pytest.raises(ValueError):
        simulator.simulate(x0, rng, t_span=(a, b), num_steps=2)


def test_unrepresentable_dt_grid(simulator, x0, rng):
    a = 1.0
    b = np.nextafter(a, np.inf)
    dt = (b - a) / 2

    with pytest.raises(ValueError):
        simulator.simulate(x0, rng, t_span=(a, b), dt=dt)


@pytest.mark.parametrize(
    "initial_condition",
    [
        [1, 2, 3, 4, 5],
        (1, 2, 3, 4, 5),
        np.arange(1, 6, dtype=np.int32),
        np.arange(1, 6, dtype=np.float32),
        np.arange(1, 6, dtype=np.float64),
        np.zeros(5),
        -np.ones(5),
    ],
)
def test_valid_initial_coefficients(simulator, rng, initial_condition):
    _, trajectory = simulator.simulate(initial_condition, rng, times=[0.0, 1.0])

    assert_array_equal(trajectory[0], initial_condition)


@pytest.mark.parametrize(
    "initial_condition",
    [
        [],
        [1],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5, 6],
        [[1, 2, 3, 4, 5]],
        [[1], [2], [3], [4], [5]],
        1.0,
        np.array(1.0),
        [1, 2, 3, 4, np.nan],
        [1, 2, 3, 4, np.inf],
        [1, 2, 3, 4, -np.inf],
    ],
)
def test_invalid_initial_coefficient_values(simulator, rng, initial_condition):
    with pytest.raises(ValueError):
        simulator.simulate(initial_condition, rng, times=[0.0, 1.0])


@pytest.mark.parametrize(
    "initial_condition",
    [
        None,
        True,
        "initial condition",
        [1, 2, 3, 4, True],
        [1, 2, 3, 4, "5"],
        [1, 2, 3, 4, 1j],
        np.ones(5, dtype=complex),
        np.ones(5, dtype=bool),
        np.ones(5, dtype=object),
    ],
)
def test_invalid_initial_coefficient_types(simulator, rng, initial_condition):
    with pytest.raises(TypeError):
        simulator.simulate(initial_condition, rng, times=[0.0, 1.0])


@pytest.mark.parametrize(
    "a, b",
    [
        (0.0, np.pi),
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 5.0),
        (-6.0, -2.0),
        (0.0, 0.001),
    ],
)
@pytest.mark.parametrize("mode", [1, 2, 3, 4, 5])
def test_callable_initial_single_mode(a, b, mode):
    simulator = make_simulator(a=a, b=b)
    basis = simulator.model.eigenbasis

    def initial_condition(x: float) -> float:
        return float(basis.evaluate(mode, x))

    expected = np.zeros(simulator.num_modes)
    expected[mode - 1] = 1.0

    _, trajectory = simulator.simulate(
        initial_condition,
        np.random.default_rng(42),
        times=[0.0],
    )

    assert_allclose(trajectory[0], expected, rtol=1e-10, atol=1e-10)


def test_callable_initial_linear_combination(simulator, x0, rng):
    basis = simulator.model.eigenbasis

    def initial_condition(x):
        return sum(
            coefficient * basis.evaluate(k, x)
            for k, coefficient in enumerate(x0, start=1)
        )

    _, trajectory = simulator.simulate(initial_condition, rng, times=[0.0])

    assert_allclose(trajectory[0], x0, rtol=1e-10, atol=1e-10)


def test_callable_initial_zero(simulator, rng):
    _, trajectory = simulator.simulate(lambda x: 0.0, rng, times=[0.0])

    assert_array_equal(trajectory[0], np.zeros(simulator.num_modes))


def test_callable_initial_constant(simulator, rng):
    _, trajectory = simulator.simulate(lambda x: 1.0, rng, times=[0.0])

    k = np.arange(1, simulator.num_modes + 1)
    expected = np.sqrt(2 / np.pi) * (1 - (-1) ** k) / k

    assert_allclose(trajectory[0], expected, rtol=1e-10, atol=1e-10)


def test_callable_initial_outside_truncation(simulator, rng):
    basis = simulator.model.eigenbasis
    mode = simulator.num_modes + 1

    _, trajectory = simulator.simulate(
        lambda x: basis.evaluate(mode, x),
        rng,
        times=[0.0],
    )

    assert_allclose(trajectory[0], 0.0, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    "initial_condition",
    [
        lambda x: "invalid",
        lambda x: 1j,
        lambda x: [1.0, 2.0],
    ],
)
def test_invalid_callable_return_type(simulator, rng, initial_condition):
    with pytest.raises(TypeError):
        simulator.simulate(initial_condition, rng, times=[0.0])


def test_callable_exception_propagates(simulator, rng):
    def initial_condition(x):
        raise RuntimeError("initial condition failed")

    with pytest.raises(RuntimeError, match="initial condition failed"):
        simulator.simulate(initial_condition, rng, times=[0.0])


@pytest.mark.parametrize(
    "invalid_projection",
    [
        np.array([1.0]),
        np.ones((1, 5)),
        np.full(5, np.nan),
        np.full(5, np.inf),
    ],
)
def test_callable_projection_is_validated(
    simulator, rng, monkeypatch, invalid_projection
):
    monkeypatch.setattr(
        SineBasis,
        "project",
        lambda self, f, num_modes: invalid_projection,
    )

    with pytest.raises(ValueError):
        simulator.simulate(lambda x: x, rng, times=[0.0])


def test_callable_projected_once(simulator, rng, monkeypatch):
    calls = []
    original = SineBasis.project

    def tracked_project(self, f, num_modes):
        calls.append(num_modes)
        return original(self, f, num_modes)

    monkeypatch.setattr(SineBasis, "project", tracked_project)

    simulator.simulate(np.sin, rng, t_span=(0, 1), num_steps=10)

    assert calls == [simulator.num_modes]


def test_coefficients_do_not_trigger_projection(simulator, x0, rng, monkeypatch):
    def forbidden_projection(*args, **kwargs):
        pytest.fail("Projection should not be called")

    monkeypatch.setattr(SineBasis, "project", forbidden_projection)

    simulator.simulate(x0, rng, times=[0.0, 1.0])


@pytest.mark.parametrize(
    "a, b",
    [
        (0.0, np.pi),
        (0.0, 1.0),
        (-1.0, 1.0),
        (1.0, 5.0),
        (-6.0, -2.0),
        (0.0, 0.001),
    ],
)
@pytest.mark.parametrize("num_modes", [1, 3, 10])
@pytest.mark.parametrize("theta", [0.1, 1.0, 3.0])
@pytest.mark.parametrize(
    "times",
    [
        [0.0],
        [0.0, 0.01, 0.1, 1.0],
        [-1.0, -0.5, 0.0, 0.5],
        [2.0, 2.1, 2.5, 3.0],
        np.linspace(0.0, 1.0, 21),
    ],
)
def test_deterministic_analytical_solution(a, b, num_modes, theta, times):
    simulator = make_simulator(
        a=a,
        b=b,
        num_modes=num_modes,
        theta=theta,
        sigma=0.0,
    )

    x0 = np.arange(1, num_modes + 1, dtype=np.float64)

    times, trajectory = simulator.simulate(x0, np.random.default_rng(42), times=times)

    k = np.arange(1, num_modes + 1)
    drift = -theta * (k * np.pi / (b - a)) ** 2

    expected = x0[None, :] * np.exp((times - times[0])[:, None] * drift[None, :])

    assert_allclose(trajectory, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("num_modes", [1, 5, 20])
@pytest.mark.parametrize("theta", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.0, 0.1, 1.0, 3.0])
@pytest.mark.parametrize(
    "times",
    [
        [0.0, 1.0],
        [0.0, 0.01, 0.1, 0.5, 1.0],
        [-1.0, -0.5, 0.0, 1.0],
        np.linspace(0.0, 1.0, 11),
    ],
)
def test_simulation_matches_manual_integration(num_modes, theta, sigma, times):
    simulator = make_simulator(
        num_modes=num_modes,
        theta=theta,
        sigma=sigma,
        eigenvalues=lambda k: 1 / k**2,
    )

    x0 = np.arange(1, num_modes + 1, dtype=np.float64)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    actual_times, actual = simulator.simulate(x0, rng1, times=times)

    expected = np.empty_like(actual)
    expected[0] = x0

    for i, dt in enumerate(np.diff(actual_times), start=1):
        expected[i] = simulator.integrator.step(expected[i - 1], dt, rng2)

    assert_array_equal(actual, expected)

    assert_array_equal(
        rng1.standard_normal(100),
        rng2.standard_normal(100),
    )


@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_reproducibility(simulator, x0, seed):
    times1, trajectory1 = simulator.simulate(
        x0,
        np.random.default_rng(seed),
        t_span=(0, 1),
        num_steps=20,
    )

    times2, trajectory2 = simulator.simulate(
        x0,
        np.random.default_rng(seed),
        t_span=(0, 1),
        num_steps=20,
    )

    assert_array_equal(times1, times2)
    assert_array_equal(trajectory1, trajectory2)


def test_different_seeds(simulator, x0):
    _, trajectory1 = simulator.simulate(
        x0,
        np.random.default_rng(42),
        times=[0.0, 0.5, 1.0],
    )

    _, trajectory2 = simulator.simulate(
        x0,
        np.random.default_rng(43),
        times=[0.0, 0.5, 1.0],
    )

    assert_array_equal(trajectory1[0], trajectory2[0])
    assert not np.array_equal(trajectory1[1:], trajectory2[1:])


def test_repeated_simulations_advance_rng(simulator, x0, rng):
    _, trajectory1 = simulator.simulate(x0, rng, times=[0.0, 0.5, 1.0])

    _, trajectory2 = simulator.simulate(x0, rng, times=[0.0, 0.5, 1.0])

    assert not np.array_equal(trajectory1[1:], trajectory2[1:])


def test_equivalent_time_specifications(simulator, x0):
    specifications = [
        {"times": np.linspace(0, 1, 9)},
        {"t_span": (0, 1), "dt": 0.125},
        {"t_span": (0, 1), "num_steps": 8},
        {"t_span": Interval(0, 1), "dt": 0.125},
        {"t_span": Interval(0, 1), "num_steps": 8},
    ]

    expected_times, expected_trajectory = simulator.simulate(
        x0,
        np.random.default_rng(42),
        **specifications[0],
    )

    for specification in specifications[1:]:
        times, trajectory = simulator.simulate(
            x0,
            np.random.default_rng(42),
            **specification,
        )

        assert_array_equal(times, expected_times)
        assert_array_equal(trajectory, expected_trajectory)


def test_callable_and_coefficient_equivalence(simulator):
    basis = simulator.model.eigenbasis

    def initial_condition(x):
        return np.sin(x) + 2 * np.sin(3 * x)

    coefficients = basis.project(initial_condition, simulator.num_modes)

    times1, trajectory1 = simulator.simulate(
        initial_condition,
        np.random.default_rng(42),
        t_span=(0, 1),
        num_steps=10,
    )

    times2, trajectory2 = simulator.simulate(
        coefficients,
        np.random.default_rng(42),
        t_span=(0, 1),
        num_steps=10,
    )

    assert_array_equal(times1, times2)
    assert_array_equal(trajectory1, trajectory2)


def test_time_translation_invariance(simulator, x0):
    _, trajectory1 = simulator.simulate(
        x0,
        np.random.default_rng(42),
        times=[0.0, 0.25, 0.5, 1.0],
    )

    _, trajectory2 = simulator.simulate(
        x0,
        np.random.default_rng(42),
        times=[-2.0, -1.75, -1.5, -1.0],
    )

    assert_array_equal(trajectory1, trajectory2)


def test_simulation_continuation(simulator, x0):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    _, full = simulator.simulate(x0, rng1, times=[0, 0.25, 0.5, 0.75, 1])

    _, first = simulator.simulate(x0, rng2, times=[0, 0.25, 0.5])

    _, second = simulator.simulate(first[-1], rng2, times=[0.5, 0.75, 1])

    combined = np.concatenate([first, second[1:]], axis=0)

    assert_array_equal(full, combined)


def test_zero_initial_and_zero_noise():
    simulator = make_simulator(sigma=0.0)

    _, trajectory = simulator.simulate(
        np.zeros(simulator.num_modes),
        np.random.default_rng(42),
        t_span=(0, 1),
        num_steps=20,
    )

    assert_array_equal(trajectory, np.zeros_like(trajectory))


def test_zero_covariance():
    simulator = make_simulator(sigma=1.0, eigenvalues=lambda k: 0.0)

    x0 = np.ones(simulator.num_modes)

    times, trajectory = simulator.simulate(
        x0,
        np.random.default_rng(42),
        t_span=(0, 1),
        num_steps=10,
    )

    k = np.arange(1, simulator.num_modes + 1)
    expected = np.exp(-times[:, None] * k[None, :] ** 2)

    assert_allclose(trajectory, expected, rtol=1e-12, atol=1e-14)


def test_finite_rank_noise():
    simulator = make_simulator(eigenvalues=lambda k: 1.0 if k <= 2 else 0.0)

    _, trajectory = simulator.simulate(
        np.zeros(simulator.num_modes),
        np.random.default_rng(42),
        t_span=(0, 1),
        num_steps=20,
    )

    assert np.any(trajectory[1:, :2] != 0)
    assert_array_equal(
        trajectory[:, 2:],
        np.zeros_like(trajectory[:, 2:]),
    )


def test_input_coefficients_not_modified(simulator, x0, rng):
    original = x0.copy()

    simulator.simulate(x0, rng, t_span=(0, 1), num_steps=10)

    assert_array_equal(x0, original)


def test_input_times_not_modified(simulator, x0, rng):
    times = np.linspace(0, 1, 11)
    original = times.copy()

    simulator.simulate(x0, rng, times=times)

    assert_array_equal(times, original)


def test_trajectory_does_not_alias_initial_condition(simulator, x0, rng):
    original = x0.copy()

    _, trajectory = simulator.simulate(x0, rng, times=[0.0, 1.0])

    assert not np.shares_memory(x0, trajectory)

    trajectory[0] = 0.0

    assert_array_equal(x0, original)


def test_read_only_inputs(simulator, x0, rng):
    times = np.linspace(0, 1, 11)

    x0.setflags(write=False)
    times.setflags(write=False)

    actual_times, trajectory = simulator.simulate(x0, rng, times=times)

    assert_array_equal(actual_times, times)
    assert_array_equal(trajectory[0], x0)


def test_independent_trajectory_arrays(simulator, x0, rng):
    _, trajectory1 = simulator.simulate(x0, rng, times=[0.0, 1.0])

    _, trajectory2 = simulator.simulate(x0, rng, times=[0.0, 1.0])

    assert not np.shares_memory(trajectory1, trajectory2)


@pytest.mark.parametrize(
    "invalid_rng",
    [
        None,
        42,
        True,
        "42",
        np.random.RandomState(42),
        np.random.PCG64(42),
    ],
)
def test_invalid_rng(simulator, x0, invalid_rng):
    with pytest.raises(TypeError):
        simulator.simulate(x0, invalid_rng, times=[0.0, 1.0])


@pytest.mark.parametrize(
    "specification",
    [
        {},
        {"times": []},
        {"times": [0, 0]},
        {"times": [1, 0]},
        {"times": [0, np.nan]},
        {"times": [0, 1], "dt": 0.1},
        {"times": [0, 1], "t_span": (0, 1)},
        {"t_span": (0, 1)},
        {"t_span": (0, 1), "dt": 0},
        {"t_span": (0, 1), "num_steps": 0},
        {"t_span": (0, 1), "dt": 0.1, "num_steps": 10},
        {"t_span": (1, 0), "dt": 0.1},
        {"dt": 0.1},
        {"num_steps": 10},
    ],
)
def test_invalid_time_specification_preserves_rng(simulator, x0, specification):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    with pytest.raises(ValueError):
        simulator.simulate(x0, rng1, **specification)

    assert_array_equal(
        rng1.standard_normal(100),
        rng2.standard_normal(100),
    )


@pytest.mark.parametrize(
    "initial_condition",
    [
        [],
        [1, 2],
        np.full(5, np.nan),
        np.full(5, np.inf),
    ],
)
def test_invalid_initial_condition_preserves_rng(simulator, initial_condition):
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    with pytest.raises(ValueError):
        simulator.simulate(initial_condition, rng1, times=[0.0, 1.0])

    assert_array_equal(
        rng1.standard_normal(100),
        rng2.standard_normal(100),
    )


def test_invalid_times_prevent_projection(simulator, rng, monkeypatch):
    def forbidden_projection(*args, **kwargs):
        pytest.fail("Projection should not be called")

    monkeypatch.setattr(SineBasis, "project", forbidden_projection)

    with pytest.raises(ValueError):
        simulator.simulate(np.sin, rng, times=[1.0, 0.0])


def test_invalid_rng_prevents_projection(simulator, monkeypatch):
    def forbidden_projection(*args, **kwargs):
        pytest.fail("Projection should not be called")

    monkeypatch.setattr(SineBasis, "project", forbidden_projection)

    with pytest.raises(TypeError):
        simulator.simulate(np.sin, None, times=[0.0, 1.0])


def test_step_call_count_and_arguments(simulator, x0, rng, monkeypatch):
    calls = []
    original_step = ExactLinear.step

    def tracked_step(self, coeffs, dt, rng):
        calls.append((coeffs.copy(), dt, rng))
        return original_step(self, coeffs, dt, rng)

    monkeypatch.setattr(ExactLinear, "step", tracked_step)

    times, trajectory = simulator.simulate(x0, rng, times=[0.0, 0.1, 0.4, 1.0])

    assert len(calls) == len(times) - 1

    for i, (coeffs, dt, actual_rng) in enumerate(calls):
        assert_array_equal(coeffs, trajectory[i])
        assert dt == times[i + 1] - times[i]
        assert actual_rng is rng


def test_single_time_does_not_call_integrator(simulator, x0, rng, monkeypatch):
    def forbidden_step(*args, **kwargs):
        pytest.fail("Integrator should not be called")

    monkeypatch.setattr(ExactLinear, "step", forbidden_step)

    simulator.simulate(x0, rng, times=[0.0])


def test_time_parameters_are_keyword_only(simulator, x0, rng):
    with pytest.raises(TypeError):
        simulator.simulate(x0, rng, [0.0, 1.0])


def test_rng_is_required(simulator, x0):
    with pytest.raises(TypeError):
        simulator.simulate(x0, times=[0.0, 1.0])


def test_initial_condition_is_required(simulator, rng):
    with pytest.raises(TypeError):
        simulator.simulate(rng=rng, times=[0.0, 1.0])
