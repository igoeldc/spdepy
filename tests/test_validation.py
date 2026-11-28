from fractions import Fraction
from numbers import Integral, Real

import numpy as np
import pytest

from spdepy._validation import (
    valid_dims,
    valid_nonneg_real,
    valid_not_bool,
    valid_not_empty,
    valid_number,
    valid_pos_int,
    valid_pos_real,
    valid_real,
    valid_real_array,
    valid_rng,
    valid_shape,
    valid_type,
)

BOOL_VALUES = [
    True,
    False,
    np.bool_(True),
    np.bool_(False),
]

INVALID_REAL_TYPES = [
    *BOOL_VALUES,
    None,
    "1",
    "1.5",
    1 + 0j,
    1 + 2j,
    np.complex128(1),
    [1],
    (1,),
    np.array(1),
    np.array([1]),
    {},
    object(),
]

NONFINITE_VALUES = [
    np.nan,
    np.inf,
    -np.inf,
    np.float32(np.nan),
    np.float32(np.inf),
    np.float64(-np.inf),
]

VALID_POSITIVE_INTEGERS = [
    1,
    2,
    100,
    np.int8(1),
    np.int16(2),
    np.int32(3),
    np.int64(4),
    np.uint8(5),
    np.uint16(6),
    np.uint32(7),
    np.uint64(8),
]

VALID_POSITIVE_REALS = [
    1,
    0.5,
    100,
    1e-20,
    1e20,
    np.int32(1),
    np.int64(2),
    np.uint64(3),
    np.float16(0.5),
    np.float32(0.5),
    np.float64(0.5),
    Fraction(1, 3),
]

VALID_ZERO_REALS = [
    0,
    0.0,
    -0.0,
    np.int32(0),
    np.int64(0),
    np.uint64(0),
    np.float32(0),
    np.float64(-0.0),
]

VALID_NEGATIVE_REALS = [
    -1,
    -0.5,
    -100,
    -1e-20,
    -1e20,
    np.int32(-1),
    np.int64(-2),
    np.float16(-0.5),
    np.float32(-0.5),
    np.float64(-0.5),
    Fraction(-1, 3),
]

VALID_REALS = [
    *VALID_POSITIVE_REALS,
    *VALID_ZERO_REALS,
    *VALID_NEGATIVE_REALS,
]

REAL_VALIDATORS = [
    valid_real,
    valid_pos_real,
    valid_nonneg_real,
]

ARRAY_VALIDATORS = [
    valid_dims,
    valid_shape,
    valid_not_empty,
]


@pytest.mark.parametrize("value", BOOL_VALUES)
def test_valid_not_bool_rejects_boolean(value):
    with pytest.raises(TypeError):
        valid_not_bool("value", value)


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        -1,
        1.5,
        np.int64(1),
        np.float64(1),
        None,
        "value",
        [],
        {},
        np.array([True, False]),
    ],
)
def test_valid_not_bool_accepts_nonboolean(value):
    assert valid_not_bool("value", value) is None


@pytest.mark.parametrize(
    "value, num_type",
    [
        (1, Integral),
        (-1, Integral),
        (0, Integral),
        (np.int32(1), Integral),
        (np.int64(1), Integral),
        (np.uint64(1), Integral),
        (1, Real),
        (1.5, Real),
        (-1.5, Real),
        (np.float32(1.5), Real),
        (np.float64(1.5), Real),
        (Fraction(1, 3), Real),
        (1, int),
        (1.5, float),
    ],
)
def test_valid_number_accepts_matching_type(value, num_type):
    assert valid_number("value", value, num_type) is None


@pytest.mark.parametrize(
    "value, num_type",
    [
        (1.5, Integral),
        (1.0, Integral),
        (np.float64(1), Integral),
        ("1", Integral),
        (None, Integral),
        (1 + 0j, Real),
        ("1", Real),
        ([1], Real),
        (np.array([1]), Real),
        (None, Real),
        (1.5, int),
        (1, float),
    ],
)
def test_valid_number_rejects_nonmatching_type(value, num_type):
    with pytest.raises(TypeError):
        valid_number("value", value, num_type)


@pytest.mark.parametrize("value", BOOL_VALUES)
def test_valid_number_does_not_independently_reject_bool(value):
    if isinstance(value, Integral):
        assert valid_number("value", value, Integral) is None
    else:
        with pytest.raises(TypeError):
            valid_number("value", value, Integral)


@pytest.mark.parametrize("value", VALID_POSITIVE_INTEGERS)
def test_valid_pos_int(value):
    result = valid_pos_int("value", value)

    assert type(result) is int
    assert result == int(value)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
        np.int8(0),
        np.int16(-1),
        np.int32(-2),
        np.int64(-3),
        np.uint64(0),
    ],
)
def test_valid_pos_int_rejects_nonpositive(value):
    with pytest.raises(ValueError):
        valid_pos_int("value", value)


@pytest.mark.parametrize(
    "value",
    [
        *INVALID_REAL_TYPES,
        1.0,
        1.5,
        0.0,
        -1.0,
        np.float32(1),
        np.float64(1),
        Fraction(1, 1),
        Fraction(1, 2),
        *NONFINITE_VALUES,
    ],
)
def test_valid_pos_int_rejects_invalid_type(value):
    with pytest.raises(TypeError):
        valid_pos_int("value", value)


@pytest.mark.parametrize("value", VALID_REALS)
def test_valid_real(value):
    result = valid_real("value", value)

    assert type(result) is float
    assert result == float(value)


@pytest.mark.parametrize("validator", REAL_VALIDATORS)
@pytest.mark.parametrize("value", INVALID_REAL_TYPES)
def test_real_validators_reject_invalid_types(validator, value):
    with pytest.raises(TypeError):
        validator("value", value)


@pytest.mark.parametrize("validator", REAL_VALIDATORS)
@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_real_validators_reject_nonfinite_values(validator, value):
    with pytest.raises(ValueError):
        validator("value", value)


@pytest.mark.parametrize("value", VALID_POSITIVE_REALS)
def test_valid_pos_real(value):
    result = valid_pos_real("value", value)

    assert type(result) is float
    assert result == float(value)


@pytest.mark.parametrize(
    "value",
    VALID_ZERO_REALS + VALID_NEGATIVE_REALS,
)
def test_valid_pos_real_rejects_nonpositive(value):
    with pytest.raises(ValueError):
        valid_pos_real("value", value)


@pytest.mark.parametrize(
    "value",
    VALID_POSITIVE_REALS + VALID_ZERO_REALS,
)
def test_valid_nonneg_real(value):
    result = valid_nonneg_real("value", value)

    assert type(result) is float
    assert result == float(value)


@pytest.mark.parametrize("value", VALID_NEGATIVE_REALS)
def test_valid_nonneg_real_rejects_negative(value):
    with pytest.raises(ValueError):
        valid_nonneg_real("value", value)


@pytest.mark.parametrize(
    "value",
    [
        -0.0,
        np.float32(-0.0),
        np.float64(-0.0),
    ],
)
def test_valid_real_preserves_negative_zero(value):
    result = valid_real("value", value)

    assert result == 0
    assert np.signbit(result)


@pytest.mark.parametrize(
    "value",
    [
        -0.0,
        np.float32(-0.0),
        np.float64(-0.0),
    ],
)
def test_valid_nonneg_real_accepts_negative_zero(value):
    result = valid_nonneg_real("value", value)

    assert result == 0


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (1, int),
        (1.5, float),
        ("value", str),
        ([], list),
        ((), tuple),
        ({}, dict),
        (np.array([1]), np.ndarray),
        (np.random.default_rng(42), np.random.Generator),
        (None, type(None)),
        (True, bool),
        (True, int),
        (np.int64(1), Integral),
        (np.float64(1), Real),
        (Fraction(1, 3), Real),
    ],
)
def test_valid_type(value, expected_type):
    result = valid_type("value", value, expected_type)

    assert result is value


@pytest.mark.parametrize(
    "value, expected_type",
    [
        (None, int),
        (1.5, int),
        (1, float),
        ("1", int),
        ([], tuple),
        ((), list),
        ([1], np.ndarray),
        (np.array([1]), list),
        (np.random.RandomState(42), np.random.Generator),
    ],
)
def test_valid_type_rejects_invalid_type(value, expected_type):
    with pytest.raises(TypeError):
        valid_type("value", value, expected_type)


def test_valid_type_accepts_subclass():
    class Parent:
        pass

    class Child(Parent):
        pass

    value = Child()

    assert valid_type("value", value, Parent) is value


def test_valid_type_accepts_exact_type():
    class Example:
        pass

    value = Example()

    assert valid_type("value", value, Example) is value


@pytest.mark.parametrize(
    "bit_generator",
    [
        np.random.PCG64,
        np.random.PCG64DXSM,
        np.random.MT19937,
        np.random.Philox,
        np.random.SFC64,
    ],
)
def test_valid_rng(bit_generator):
    rng = np.random.Generator(bit_generator(42))

    assert valid_rng(rng) is rng


@pytest.mark.parametrize(
    "rng",
    [
        None,
        42,
        "42",
        True,
        np.random.RandomState(42),
        np.random.PCG64(42),
        np.random.MT19937(42),
        np.random,
        np.random.default_rng,
        object(),
    ],
)
def test_valid_rng_rejects_invalid_type(rng):
    with pytest.raises(TypeError):
        valid_rng(rng)


def test_valid_rng_does_not_advance_state():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    valid_rng(rng1)

    np.testing.assert_array_equal(
        rng1.standard_normal(100),
        rng2.standard_normal(100),
    )


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (1,),
        (5,),
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (2, 3),
        (2, 3, 4),
        (2, 0, 4),
        (1, 2, 3, 4),
    ],
)
def test_valid_dims(shape):
    value = np.empty(shape)

    result = valid_dims("value", value, len(shape))

    assert result is value


@pytest.mark.parametrize(
    "shape, expected_dims",
    [
        ((), 1),
        ((), 2),
        ((1,), 0),
        ((5,), 2),
        ((2, 3), 0),
        ((2, 3), 1),
        ((2, 3), 3),
        ((2, 3, 4), 2),
        ((2, 3, 4), 4),
        ((0,), 2),
        ((3, 0), 1),
    ],
)
def test_valid_dims_rejects_wrong_dimensions(shape, expected_dims):
    value = np.empty(shape)

    with pytest.raises(ValueError):
        valid_dims("value", value, expected_dims)


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        1.5,
        "value",
        [],
        [1, 2, 3],
        (1, 2, 3),
        {},
        np.float64(1),
    ],
)
def test_valid_dims_rejects_nonarray(value):
    with pytest.raises(TypeError):
        valid_dims("value", value, 1)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (1,),
        (5,),
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (2, 3),
        (2, 3, 4),
        (2, 0, 4),
        (1, 2, 3, 4),
    ],
)
def test_valid_shape(shape):
    value = np.empty(shape)

    result = valid_shape("value", value, shape)

    assert result is value


@pytest.mark.parametrize(
    "actual_shape, expected_shape",
    [
        ((), (1,)),
        ((1,), ()),
        ((3,), (4,)),
        ((3,), (1, 3)),
        ((3,), (3, 1)),
        ((2, 3), (3, 2)),
        ((2, 3), (6,)),
        ((2, 3), (2, 4)),
        ((2, 3, 4), (2, 3)),
        ((2, 3, 4), (2, 4, 3)),
        ((0,), (1,)),
        ((0,), (0, 0)),
        ((3, 0), (0, 3)),
    ],
)
def test_valid_shape_rejects_wrong_shape(actual_shape, expected_shape):
    value = np.empty(actual_shape)

    with pytest.raises(ValueError):
        valid_shape("value", value, expected_shape)


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        1.5,
        "value",
        [],
        [1, 2, 3],
        (1, 2, 3),
        {},
        np.float64(1),
    ],
)
def test_valid_shape_rejects_nonarray(value):
    with pytest.raises(TypeError):
        valid_shape("value", value, (3,))


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (1,),
        (5,),
        (1, 1),
        (2, 3),
        (2, 3, 4),
        (1, 2, 3, 4),
    ],
)
def test_valid_not_empty(shape):
    value = np.empty(shape)

    assert valid_not_empty("value", value) is value


@pytest.mark.parametrize(
    "shape",
    [
        (0,),
        (0, 0),
        (0, 3),
        (3, 0),
        (0, 2, 3),
        (2, 0, 3),
        (2, 3, 0),
        (1, 2, 0, 4),
    ],
)
def test_valid_not_empty_rejects_empty(shape):
    value = np.empty(shape)

    with pytest.raises(ValueError):
        valid_not_empty("value", value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        1.5,
        "value",
        [],
        [1, 2, 3],
        (1, 2, 3),
        {},
        np.float64(1),
    ],
)
def test_valid_not_empty_rejects_nonarray(value):
    with pytest.raises(TypeError):
        valid_not_empty("value", value)


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.int64,
        np.float64,
        np.complex128,
        np.str_,
        object,
    ],
)
def test_structural_validators_are_dtype_independent(dtype):
    value = np.ones((2, 3), dtype=dtype)

    assert valid_dims("value", value, 2) is value
    assert valid_shape("value", value, (2, 3)) is value
    assert valid_not_empty("value", value) is value


def test_structural_validators_do_not_validate_finiteness():
    value = np.array([np.nan, np.inf, -np.inf])

    assert valid_dims("value", value, 1) is value
    assert valid_shape("value", value, (3,)) is value
    assert valid_not_empty("value", value) is value


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, np.array(1.0)),
        (1.5, np.array(1.5)),
        (np.int64(1), np.array(1.0)),
        (np.float32(0.5), np.array(0.5)),
        (Fraction(1, 2), np.array(0.5)),
        ([1, 2, 3], np.array([1.0, 2.0, 3.0])),
        ((1, 2, 3), np.array([1.0, 2.0, 3.0])),
        (range(3), np.array([0.0, 1.0, 2.0])),
        (
            [1, 2.5, np.float32(3.5)],
            np.array([1.0, 2.5, 3.5]),
        ),
        (
            [Fraction(1, 2), Fraction(3, 4)],
            np.array([0.5, 0.75]),
        ),
        (
            [[1, 2], [3, 4]],
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        ),
        (
            ((1, 2), (3, 4)),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        ),
        (
            [[[1, 2], [3, 4]]],
            np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        ),
        ([], np.empty((0,))),
        ((), np.empty((0,))),
        ([[], []], np.empty((2, 0))),
    ],
)
def test_valid_real_array_conversion(value, expected):
    result = valid_real_array("value", value)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == expected.shape

    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (1,),
        (5,),
        (2, 3),
        (3, 0),
        (2, 3, 4),
    ],
)
def test_valid_real_array_numeric_dtypes(dtype, shape):
    value = np.ones(shape, dtype=dtype)

    result = valid_real_array("value", value)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape == shape

    np.testing.assert_array_equal(
        result,
        value.astype(np.float64),
    )


@pytest.mark.parametrize(
    "dtype",
    [
        np.bool_,
        np.complex64,
        np.complex128,
        np.str_,
        np.bytes_,
        object,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (3,),
        (2, 3),
    ],
)
def test_valid_real_array_rejects_invalid_dtype(dtype, shape):
    value = np.ones(shape, dtype=dtype)

    with pytest.raises(TypeError):
        valid_real_array("value", value)


@pytest.mark.parametrize(
    "value",
    [
        np.array(["2026-01-01"], dtype="datetime64[D]"),
        np.array([1, 2], dtype="timedelta64[D]"),
        np.zeros(3, dtype=[("x", "f8"), ("y", "f8")]),
    ],
)
def test_valid_real_array_rejects_other_dtypes(value):
    with pytest.raises(TypeError):
        valid_real_array("value", value)


@pytest.mark.parametrize(
    "value",
    [
        *BOOL_VALUES,
        None,
        "1",
        b"1",
        1 + 0j,
        1 + 2j,
        {},
        object(),
        [True, False],
        [1, True],
        [True, 1],
        [1, np.bool_(False)],
        [1, "2"],
        ["1", 2],
        [1, 2 + 0j],
        [1, None],
        [1, {}],
        [[1, 2], [3, True]],
        [[1, 2], [3, "4"]],
        [[1, 2], [3, None]],
        np.array([1, 2, 3], dtype=object),
    ],
)
def test_valid_real_array_rejects_invalid_contents(value):
    with pytest.raises(TypeError):
        valid_real_array("value", value)


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_valid_real_array_rejects_nonfinite_scalar(value):
    with pytest.raises(ValueError):
        valid_real_array("value", value)


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_valid_real_array_rejects_nonfinite_sequence(value):
    with pytest.raises(ValueError):
        valid_real_array("value", [1, value, 3])


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_valid_real_array_rejects_nonfinite_ndarray(value):
    array = np.array([1, value, 3], dtype=np.float64)

    with pytest.raises(ValueError):
        valid_real_array("value", array)


@pytest.mark.parametrize("value", NONFINITE_VALUES)
def test_valid_real_array_rejects_nonfinite_multidimensional(value):
    array = np.array(
        [[1, 2], [3, value]],
        dtype=np.float64,
    )

    with pytest.raises(ValueError):
        valid_real_array("value", array)


@pytest.mark.parametrize(
    "value",
    [
        [[1], [2, 3]],
        [1, [2, 3]],
        [[1, 2], [3]],
    ],
)
def test_valid_real_array_rejects_ragged_sequences(value):
    with pytest.raises((TypeError, ValueError)):
        valid_real_array("value", value)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (1,),
        (5,),
        (2, 3),
        (3, 0),
        (2, 3, 4),
        (1, 2, 3, 4),
    ],
)
def test_valid_real_array_preserves_shape(shape):
    value = np.zeros(shape)

    result = valid_real_array("value", value)

    assert result.shape == shape


@pytest.mark.parametrize(
    "dtype",
    [
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_valid_real_array_does_not_modify_input(dtype):
    value = np.arange(12, dtype=dtype).reshape(3, 4)
    original = value.copy()

    valid_real_array("value", value)

    assert value.dtype == dtype

    np.testing.assert_array_equal(value, original)


def test_valid_real_array_does_not_modify_list():
    value = [[1, 2], [3, 4]]
    original = [row.copy() for row in value]

    valid_real_array("value", value)

    assert value == original


def test_valid_real_array_accepts_read_only_array():
    value = np.arange(5, dtype=np.float64)
    value.flags.writeable = False

    result = valid_real_array("value", value)

    np.testing.assert_array_equal(result, value)

    assert not value.flags.writeable


def test_valid_real_array_accepts_noncontiguous_array():
    value = np.arange(20, dtype=np.float64)[::2]

    assert not value.flags.c_contiguous

    result = valid_real_array("value", value)

    np.testing.assert_array_equal(result, value)


def test_valid_real_array_accepts_transposed_array():
    value = np.arange(12, dtype=np.float64).reshape(3, 4).T

    result = valid_real_array("value", value)

    assert result.shape == (4, 3)

    np.testing.assert_array_equal(result, value)


def test_valid_real_array_accepts_negative_strides():
    value = np.arange(5, dtype=np.float64)[::-1]

    result = valid_real_array("value", value)

    np.testing.assert_array_equal(result, value)


def test_valid_real_array_accepts_fortran_order():
    value = np.asfortranarray(np.arange(12).reshape(3, 4))

    result = valid_real_array("value", value)

    assert result.dtype == np.float64

    np.testing.assert_array_equal(result, value)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (1,),
        (5,),
        (2, 3),
        (3, 0),
        (2, 3, 4),
    ],
)
def test_valid_real_array_ndim_constraint(shape):
    value = np.ones(shape)

    result = valid_real_array(
        "value",
        value,
        ndim=len(shape),
    )

    assert result.ndim == len(shape)
    assert result.shape == shape


@pytest.mark.parametrize(
    "value, ndim",
    [
        (1, 0),
        (1.5, 0),
        ([], 1),
        ([1, 2, 3], 1),
        ((1, 2, 3), 1),
        ([[1, 2], [3, 4]], 2),
        ([[[1, 2], [3, 4]]], 3),
    ],
)
def test_valid_real_array_ndim_after_conversion(value, ndim):
    result = valid_real_array(
        "value",
        value,
        ndim=ndim,
    )

    assert result.ndim == ndim


@pytest.mark.parametrize(
    "value, ndim",
    [
        (1, 1),
        (1, 2),
        ([], 0),
        ([], 2),
        ([1, 2, 3], 0),
        ([1, 2, 3], 2),
        ([[1, 2], [3, 4]], 1),
        ([[1, 2], [3, 4]], 3),
        (np.empty((3, 0)), 1),
    ],
)
def test_valid_real_array_rejects_wrong_ndim(value, ndim):
    with pytest.raises(ValueError):
        valid_real_array("value", value, ndim=ndim)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (1,),
        (5,),
        (0, 3),
        (3, 0),
        (2, 3),
        (2, 3, 4),
    ],
)
def test_valid_real_array_shape_constraint(shape):
    value = np.ones(shape)

    result = valid_real_array(
        "value",
        value,
        shape=shape,
    )

    assert result.shape == shape


@pytest.mark.parametrize(
    "value, shape",
    [
        (1, ()),
        (1.5, ()),
        ([], (0,)),
        ([1, 2, 3], (3,)),
        ((1, 2, 3), (3,)),
        ([[1, 2], [3, 4]], (2, 2)),
        ([[[1, 2], [3, 4]]], (1, 2, 2)),
        ([[], []], (2, 0)),
    ],
)
def test_valid_real_array_shape_after_conversion(value, shape):
    result = valid_real_array(
        "value",
        value,
        shape=shape,
    )

    assert result.shape == shape


@pytest.mark.parametrize(
    "value, shape",
    [
        (1, (1,)),
        ([1], ()),
        ([], (1,)),
        ([1, 2, 3], (2,)),
        ([1, 2, 3], (4,)),
        ([1, 2, 3], (1, 3)),
        ([1, 2, 3], (3, 1)),
        ([[1, 2], [3, 4]], (4,)),
        ([[1, 2], [3, 4]], (1, 4)),
        ([[1, 2], [3, 4]], (2, 3)),
        (np.empty((3, 0)), (0, 3)),
    ],
)
def test_valid_real_array_rejects_wrong_shape(value, shape):
    with pytest.raises(ValueError):
        valid_real_array("value", value, shape=shape)


@pytest.mark.parametrize(
    "value",
    [
        [],
        (),
        [[], []],
        np.empty((0,)),
        np.empty((0, 0)),
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.empty((2, 0, 4)),
    ],
)
def test_valid_real_array_allows_empty_by_default(value):
    result = valid_real_array("value", value)

    assert result.size == 0


@pytest.mark.parametrize(
    "value",
    [
        [],
        (),
        [[], []],
        np.empty((0,)),
        np.empty((0, 0)),
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.empty((2, 0, 4)),
    ],
)
def test_valid_real_array_explicitly_allows_empty(value):
    result = valid_real_array(
        "value",
        value,
        allow_empty=True,
    )

    assert result.size == 0


@pytest.mark.parametrize(
    "value",
    [
        [],
        (),
        [[], []],
        np.empty((0,)),
        np.empty((0, 0)),
        np.empty((0, 3)),
        np.empty((3, 0)),
        np.empty((2, 0, 4)),
    ],
)
def test_valid_real_array_rejects_empty(value):
    with pytest.raises(ValueError):
        valid_real_array(
            "value",
            value,
            allow_empty=False,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        1.5,
        [0],
        [1, 2, 3],
        [[1, 2], [3, 4]],
        np.zeros(()),
        np.zeros((1,)),
        np.zeros((5,)),
        np.zeros((2, 3)),
    ],
)
def test_valid_real_array_accepts_nonempty(value):
    result = valid_real_array(
        "value",
        value,
        allow_empty=False,
    )

    assert result.size > 0


@pytest.mark.parametrize(
    "value, ndim, shape",
    [
        (1, 0, ()),
        ([1], 1, (1,)),
        ([1, 2, 3], 1, (3,)),
        ([[1, 2], [3, 4]], 2, (2, 2)),
        (np.ones((2, 3, 4)), 3, (2, 3, 4)),
    ],
)
def test_valid_real_array_combined_constraints(value, ndim, shape):
    result = valid_real_array(
        "value",
        value,
        ndim=ndim,
        shape=shape,
        allow_empty=False,
    )

    assert result.ndim == ndim
    assert result.shape == shape
    assert result.size > 0
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "value, ndim, shape",
    [
        ([], 1, (0,)),
        ([[], []], 2, (2, 0)),
        (np.empty((3, 0)), 2, (3, 0)),
        (np.empty((2, 0, 4)), 3, (2, 0, 4)),
    ],
)
def test_valid_real_array_combined_empty_constraints(value, ndim, shape):
    result = valid_real_array(
        "value",
        value,
        ndim=ndim,
        shape=shape,
        allow_empty=True,
    )

    assert result.ndim == ndim
    assert result.shape == shape
    assert result.size == 0


@pytest.mark.parametrize(
    "value, kwargs",
    [
        (
            [1, 2, 3],
            {"ndim": 2, "shape": (3,)},
        ),
        (
            [1, 2, 3],
            {"ndim": 1, "shape": (4,)},
        ),
        (
            [1, 2, 3],
            {"ndim": 2, "shape": (4,)},
        ),
        (
            [],
            {"ndim": 1, "shape": (0,), "allow_empty": False},
        ),
        (
            np.empty((3, 0)),
            {
                "ndim": 2,
                "shape": (3, 0),
                "allow_empty": False,
            },
        ),
        (
            1,
            {"ndim": 1, "shape": ()},
        ),
        (
            1,
            {"ndim": 0, "shape": (1,)},
        ),
    ],
)
def test_valid_real_array_rejects_combined_constraints(value, kwargs):
    with pytest.raises(ValueError):
        valid_real_array("value", value, **kwargs)


@pytest.mark.parametrize(
    "value",
    [
        1,
        [1, 2, 3],
        [[1, 2], [3, 4]],
        np.empty((0,)),
        np.empty((3, 0)),
    ],
)
def test_valid_real_array_explicit_default_constraints(value):
    default = valid_real_array("value", value)

    explicit = valid_real_array(
        "value",
        value,
        ndim=None,
        shape=None,
        allow_empty=True,
    )

    np.testing.assert_array_equal(default, explicit)


@pytest.mark.parametrize(
    "value, kwargs",
    [
        ([1, True], {"ndim": 1}),
        ([1, True], {"shape": (2,)}),
        ([1, "2"], {"allow_empty": False}),
        (
            [1, 2 + 0j],
            {
                "ndim": 1,
                "shape": (2,),
                "allow_empty": False,
            },
        ),
    ],
)
def test_valid_real_array_constraints_do_not_bypass_type_validation(value, kwargs):
    with pytest.raises(TypeError):
        valid_real_array("value", value, **kwargs)


@pytest.mark.parametrize(
    "value, kwargs",
    [
        ([1, np.nan], {"ndim": 1}),
        ([1, np.inf], {"shape": (2,)}),
        ([1, -np.inf], {"allow_empty": False}),
        (
            [1, np.nan],
            {
                "ndim": 1,
                "shape": (2,),
                "allow_empty": False,
            },
        ),
    ],
)
def test_valid_real_array_constraints_do_not_bypass_finiteness(value, kwargs):
    with pytest.raises(ValueError):
        valid_real_array("value", value, **kwargs)


@pytest.mark.parametrize(
    "args",
    [
        ("value", [1, 2, 3], 1),
        ("value", [1, 2, 3], 1, (3,)),
        ("value", [1, 2, 3], 1, (3,), False),
    ],
)
def test_valid_real_array_constraints_are_keyword_only(args):
    with pytest.raises(TypeError):
        valid_real_array(*args)


@pytest.mark.parametrize(
    "validator, args, exception",
    [
        (valid_not_bool, (True,), TypeError),
        (valid_number, ("1", Real), TypeError),
        (valid_pos_int, (0,), ValueError),
        (valid_pos_int, (1.5,), TypeError),
        (valid_real, ("1",), TypeError),
        (valid_real, (np.nan,), ValueError),
        (valid_nonneg_real, (-1,), ValueError),
        (valid_pos_real, (0,), ValueError),
        (valid_type, (1, str), TypeError),
        (valid_dims, (np.ones(3), 2), ValueError),
        (valid_shape, (np.ones(3), (4,)), ValueError),
        (valid_not_empty, (np.empty(0),), ValueError),
        (valid_real_array, ([1, True],), TypeError),
        (valid_real_array, ([1, np.nan],), ValueError),
    ],
)
def test_validation_errors_include_parameter_name(validator, args, exception):
    with pytest.raises(exception, match="example_parameter"):
        validator("example_parameter", *args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ndim": 2},
        {"shape": (3,)},
    ],
)
def test_array_constraint_errors_include_parameter_name(kwargs):
    with pytest.raises(ValueError, match="coeffs"):
        valid_real_array("coeffs", [1, 2], **kwargs)


def test_empty_array_error_includes_parameter_name():
    with pytest.raises(ValueError, match="coeffs"):
        valid_real_array(
            "coeffs",
            [],
            allow_empty=False,
        )


def test_rng_error_includes_parameter_name():
    with pytest.raises(TypeError, match="rng"):
        valid_rng(None)
