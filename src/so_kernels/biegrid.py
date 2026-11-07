"""Deterministic BIEGrid operators used by the historical RSRS regressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import dblquad


Variant = Literal[
    "real_symmetric",
    "real_nonsymmetric",
    "complex_symmetric",
    "complex_nonsymmetric",
]


def _regular_square(cells_per_axis: int) -> NDArray[np.float64]:
    coordinates = np.arange(cells_per_axis + 1, dtype=np.float64) / cells_per_axis
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))


def _laplace_2d(dx: NDArray, dy: NDArray) -> NDArray:
    radius = np.hypot(dx, dy)
    with np.errstate(divide="ignore"):
        values = -(0.5 / np.pi) * np.log(radius)
    return np.where(radius == 0.0, 0.0, values)


class _TransposeView:
    def __init__(self, parent: "BIEGridOperator"):
        self.parent = parent
        self.shape = parent.shape
        self.dtype = parent.dtype

    def matvec(self, values):
        return self.parent.transpose_matmat(np.asarray(values)).reshape(-1)

    def matmat(self, values):
        return self.parent.transpose_matmat(values)

    def __matmul__(self, values):
        values = np.asarray(values)
        return self.matvec(values) if values.ndim == 1 else self.matmat(values)


@dataclass
class BIEGridOperator:
    """FFT-applied 2-D Laplace BIEGrid plus a deterministic rank-one update.

    The complex transpose is the algebraic transpose, not the conjugate
    transpose. Sample generation should pass :attr:`transpose_operator`
    explicitly for nonsymmetric RSRS factorisations.
    """

    cells_per_axis: int = 22
    variant: Variant = "real_symmetric"
    perturbation_scale: float = 1.0e-2
    perturbation_seed: int = 12345

    def __post_init__(self) -> None:
        if self.cells_per_axis < 4:
            raise ValueError("cells_per_axis must be at least four")
        allowed = {
            "real_symmetric",
            "real_nonsymmetric",
            "complex_symmetric",
            "complex_nonsymmetric",
        }
        if self.variant not in allowed:
            raise ValueError(f"unknown BIEGrid variant {self.variant!r}")

        self.points = _regular_square(self.cells_per_axis)
        self.dimension = self.points.shape[0]
        self.shape = (self.dimension, self.dimension)
        self.dtype = np.dtype(
            np.complex128 if self.variant.startswith("complex") else np.float64
        )
        self._build_fft_kernel()
        self._build_perturbation()
        self.transpose_operator = _TransposeView(self)

    @property
    def symmetric(self) -> bool:
        return self.variant.endswith("symmetric") and not self.variant.endswith(
            "nonsymmetric"
        )

    def _build_fft_kernel(self) -> None:
        n = self.cells_per_axis
        grid = self.points[:, :2]
        base = (_laplace_2d(grid[:, 0], grid[:, 1]) / self.dimension).reshape(
            n + 1, n + 1
        )
        embedded = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.float64)
        embedded[: n + 1, : n + 1] = base
        embedded[: n + 1, n + 1 :] = base[:, 1 : n + 1]
        embedded[n + 1 :, : n + 1] = base[1 : n + 1, :]
        embedded[n + 1 :, n + 1 :] = base[1 : n + 1, 1 : n + 1]
        embedded[:, n + 1 :] = np.fliplr(embedded[:, n + 1 :])
        embedded[n + 1 :, :] = np.flipud(embedded[n + 1 :, :])
        self._kernel_fft = np.fft.fft2(embedded)

        h = 1.0 / n
        diagonal = 4.0 * dblquad(
            lambda y, x: -(0.5 / np.pi) * np.log(np.hypot(x, y)),
            0.0,
            h / 2.0,
            lambda _x: 0.0,
            lambda _x: h / 2.0,
        )[0]
        self._diagonal = float(diagonal)

    def _build_perturbation(self) -> None:
        rng = np.random.default_rng(self.perturbation_seed)
        if self.variant.startswith("complex"):
            left = rng.standard_normal(self.dimension) + 1j * rng.standard_normal(
                self.dimension
            )
            right = rng.standard_normal(self.dimension) + 1j * rng.standard_normal(
                self.dimension
            )
            alpha = self.perturbation_scale * (1.0 + 2.0j) / np.sqrt(5.0)
        else:
            left = rng.standard_normal(self.dimension)
            right = rng.standard_normal(self.dimension)
            alpha = self.perturbation_scale
        self._left = np.asarray(left / np.linalg.norm(left), dtype=self.dtype)
        self._right = np.asarray(right / np.linalg.norm(right), dtype=self.dtype)
        self._alpha = self.dtype.type(alpha)

    def _base_matmat(self, values: NDArray) -> NDArray:
        values = np.asarray(values, dtype=self.dtype)
        is_vector = values.ndim == 1
        matrix = values.reshape(self.dimension, 1) if is_vector else values
        n = self.cells_per_axis
        blocks = matrix.reshape(n + 1, n + 1, matrix.shape[1])
        transformed = self._kernel_fft[:, :, None] * np.fft.fft2(
            blocks, s=(2 * n + 1, 2 * n + 1), axes=(0, 1)
        )
        result = np.fft.ifft2(transformed, axes=(0, 1))[: n + 1, : n + 1]
        result = result.reshape(self.dimension, matrix.shape[1])
        if not np.issubdtype(self.dtype, np.complexfloating):
            result = result.real
        result = result + self._diagonal * matrix
        return result.reshape(-1) if is_vector else result

    def matmat(self, values):
        values = np.asarray(values, dtype=self.dtype)
        is_vector = values.ndim == 1
        matrix = values.reshape(self.dimension, 1) if is_vector else values
        result = self._base_matmat(matrix)
        if self.symmetric:
            coefficient = self._left @ matrix
            result += self._alpha * np.outer(self._left, coefficient)
        else:
            coefficient = np.conjugate(self._right) @ matrix
            result += self._alpha * np.outer(self._left, coefficient)
        return result.reshape(-1) if is_vector else result

    def matvec(self, values):
        return self.matmat(values)

    def transpose_matmat(self, values):
        values = np.asarray(values, dtype=self.dtype)
        is_vector = values.ndim == 1
        matrix = values.reshape(self.dimension, 1) if is_vector else values
        result = self._base_matmat(matrix)
        if self.symmetric:
            coefficient = self._left @ matrix
            result += self._alpha * np.outer(self._left, coefficient)
        else:
            coefficient = self._left @ matrix
            result += self._alpha * np.outer(np.conjugate(self._right), coefficient)
        return result.reshape(-1) if is_vector else result

    def dense(self) -> NDArray:
        return self.matmat(np.eye(self.dimension, dtype=self.dtype))

    def __matmul__(self, values):
        values = np.asarray(values)
        return self.matvec(values) if values.ndim == 1 else self.matmat(values)


def biegrid_real_symmetric(**kwargs) -> BIEGridOperator:
    return BIEGridOperator(variant="real_symmetric", **kwargs)


def biegrid_real_nonsymmetric(**kwargs) -> BIEGridOperator:
    return BIEGridOperator(variant="real_nonsymmetric", **kwargs)


def biegrid_complex_symmetric(**kwargs) -> BIEGridOperator:
    return BIEGridOperator(variant="complex_symmetric", **kwargs)


def biegrid_complex_nonsymmetric(**kwargs) -> BIEGridOperator:
    return BIEGridOperator(variant="complex_nonsymmetric", **kwargs)
