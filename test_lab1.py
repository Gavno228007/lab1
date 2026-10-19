import numpy as np

import python_basics as pb
import numpy_tasks as nt
from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)
from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
)


# ==========================================
# Часть 1. Python Basics
# ==========================================

def test_count_vowels():
    assert pb.count_vowels(TextInput(value="Hello World")) == 3
    assert pb.count_vowels(TextInput(value="AEIOUaeiou")) == 10
    assert pb.count_vowels(TextInput(value="bcdfghjklmnpqrstvwxyz")) == 0
    assert pb.count_vowels(TextInput(value="")) == 0


def test_has_unique_characters():
    assert pb.has_unique_characters(TextInput(value="abcdef")) is True
    assert pb.has_unique_characters(TextInput(value="aba")) is False
    assert pb.has_unique_characters(TextInput(value="")) is True
    assert pb.has_unique_characters(TextInput(value="Aa")) is True


def test_count_one_bits():
    assert pb.count_one_bits(PositiveIntegerInput(value=7)) == 3   # 111
    assert pb.count_one_bits(PositiveIntegerInput(value=8)) == 1   # 1000
    assert pb.count_one_bits(PositiveIntegerInput(value=1)) == 1   # 1


def test_multiplicative_persistence():
    assert pb.multiplicative_persistence(PositiveIntegerInput(value=39)) == 3
    assert pb.multiplicative_persistence(PositiveIntegerInput(value=4)) == 0
    assert pb.multiplicative_persistence(PositiveIntegerInput(value=999)) == 4


def test_mse():
    assert pb.mse(VectorPairInput(predicted=[1, 2, 3], expected=[1, 2, 3])) == 0.0
    assert pb.mse(VectorPairInput(predicted=[1, 2], expected=[2, 4])) == 2.5
    assert pb.mse(VectorPairInput(predicted=[], expected=[])) == 0.0


def test_prime_factorization():
    assert pb.prime_factorization(PositiveIntegerInput(value=86240)) == "(2**5)(5)(7**2)(11)"
    assert pb.prime_factorization(PositiveIntegerInput(value=13)) == "(13)"
    assert pb.prime_factorization(PositiveIntegerInput(value=1)) == "(1)"


def test_pyramid():
    assert pb.pyramid(PositiveIntegerInput(value=14)) == 3  # 1 + 4 + 9 = 14
    assert pb.pyramid(PositiveIntegerInput(value=1)) == 1
    assert pb.pyramid(PositiveIntegerInput(value=10)) == "It is impossible"


def test_is_balanced_number():
    assert pb.is_balanced_number(PositiveIntegerInput(value=1234006)) is True
    assert pb.is_balanced_number(PositiveIntegerInput(value=123456)) is False
    assert pb.is_balanced_number(PositiveIntegerInput(value=7)) is True
    assert pb.is_balanced_number(PositiveIntegerInput(value=99)) is True


# ==========================================
# Часть 2. NumPy Tasks
# ==========================================

def test_sum_prod():
    mats = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 0.0], [1.0, 2.0]],
    ])
    vecs = np.array([
        [[1.0], [1.0]],
        [[2.0], [3.0]],
    ])
    # A1*x1 = [[3], [7]], A2*x2 = [[4], [8]] -> Sum = [[7], [15]]
    res = nt.sum_prod(MatrixVectorBatchInput(matrices=mats, vectors=vecs))
    np.testing.assert_allclose(res, [[7.0], [15.0]])


def test_binarize():
    m = np.array([[1.0, 5.0], [10.0, 2.0]])
    res = nt.binarize(BinarizeInput(matrix=m, threshold=4.0))
    np.testing.assert_array_equal(res, [[0, 1], [1, 0]])


def test_unique_rows_and_columns():
    m = np.array([
        [1.0, 2.0, 1.0],
        [3.0, 3.0, 3.0],
    ])
    assert nt.unique_rows(MatrixInput(matrix=m)) == [[1.0, 2.0], [3.0]]
    assert nt.unique_columns(MatrixInput(matrix=m)) == [[1.0, 3.0], [2.0, 3.0], [1.0, 3.0]]


def test_matrix_statistics():
    stats = nt.matrix_statistics(RandomMatrixInput(rows=10, columns=5, mean=0.0, std=1.0, seed=42))
    assert stats.matrix.shape == (10, 5)
    assert len(stats.row_means) == 10
    assert len(stats.column_means) == 5


def test_chess():
    res = nt.chess(ChessInput(rows=3, columns=4, first=0, second=1))
    expected = [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    np.testing.assert_array_equal(res, expected)


def test_draw_rectangle():
    res = nt.draw_rectangle(
        RectangleInput(
            image_height=6,
            image_width=6,
            height=2,
            width=2,
            shape_color=(255, 0, 0),
            background_color=(0, 0, 0),
        )
    )
    assert res.shape == (6, 6, 3)
    expected_block = np.zeros((2, 2, 3), dtype=np.uint8)
    expected_block[:, :] = (255, 0, 0)
    np.testing.assert_array_equal(res[2:4, 2:4], expected_block)
    np.testing.assert_array_equal(res[0, 0], (0, 0, 0))

def test_draw_ellipse():
    res = nt.draw_ellipse(EllipseInput(
        image_height=7,
        image_width=7,
        semi_axis_y=2.0,
        semi_axis_x=2.0,
        shape_color=(255, 255, 255),
        background_color=(0, 0, 0),
    ))
    assert res.shape == (7, 7, 3)
    np.testing.assert_array_equal(res[3, 3], (255, 255, 255))
    np.testing.assert_array_equal(res[0, 0], (0, 0, 0))


def test_analyze_time_series():
    series = np.array([1.0, 3.0, 2.0, 5.0, 1.0])
    stats = nt.analyze_time_series(TimeSeriesInput(values=series, window=3))
    assert stats.local_maxima_indices == [1, 3]
    assert stats.local_minima_indices == [2]
    assert len(stats.moving_average) == len(series) - 3 + 1


def test_one_hot():
    res = nt.one_hot(OneHotInput(labels=np.array([0, 2, 3, 0]), class_count=None))
    expected = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    np.testing.assert_array_equal(res, expected)