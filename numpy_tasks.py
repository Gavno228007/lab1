"""Заготовки задач на NumPy."""

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput,
    ChessInput,
    EllipseInput,
    MatrixInput,
    MatrixStatistics,
    MatrixVectorBatchInput,
    OneHotInput,
    RandomMatrixInput,
    RectangleInput,
    TimeSeriesInput,
    TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    return np.sum(data.matrices @ data.vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    return np.where(data.matrix > data.threshold, 1, 0)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    return [np.unique(row).tolist() for row in data.matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    return [np.unique(col).tolist() for col in data.matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)
    matrix = rng.normal(
        loc=data.mean, scale=data.std, size=(data.rows, data.columns)
    )
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    r, c = np.indices((data.rows, data.columns))
    return np.where((r + c) % 2 == 0, data.first, data.second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    image = np.full(
        (data.image_height, data.image_width, 3),
        data.background_color,
        dtype=np.uint8,
    )
    start_y = (data.image_height - data.height) // 2
    start_x = (data.image_width - data.width) // 2
    image[
        start_y : start_y + data.height,
        start_x : start_x + data.width,
    ] = data.shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    image = np.full(
        (data.image_height, data.image_width, 3),
        data.background_color,
        dtype=np.uint8,
    )
    y0 = (data.image_height - 1) / 2
    x0 = (data.image_width - 1) / 2
    y, x = np.ogrid[: data.image_height, : data.image_width]
    mask = (
        ((x - x0) ** 2) / (data.semi_axis_x**2)
        + ((y - y0) ** 2) / (data.semi_axis_y**2)
        <= 1
    )
    image[mask] = data.shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = data.values
    local_maxima = (
        np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0]
        + 1
    )
    local_minima = (
        np.where((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0]
        + 1
    )
    moving_average = np.convolve(
        values, np.ones(data.window) / data.window, mode="valid"
    )
    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=local_maxima.tolist(),
        local_minima_indices=local_minima.tolist(),
        moving_average=moving_average.tolist(),
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = data.labels
    class_count = data.class_count
    if not class_count:
        class_count = np.max(labels) + 1
    n = len(labels)
    encoded = np.zeros((n, class_count), dtype=int)
    encoded[np.arange(n), labels] = 1
    return encoded