"""Matrix-free composition of a Bempp Laplace single-layer operator."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _shape(operator: Any, name: str) -> tuple[int, int]:
    try:
        rows, columns = operator.shape
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(f"{name} must expose a two-dimensional shape") from exc
    return int(rows), int(columns)


class LaplaceSingleLayerFmm:
    """The composition ``Q @ (K - C) @ P + S`` used by Bempp-rs."""

    def __init__(
        self,
        space_to_point: Any,
        kernel: Any,
        near_correction: Any,
        point_to_space: Any,
        singular: Any,
        *,
        dtype: Any | None = None,
    ) -> None:
        p_shape = _shape(space_to_point, "space_to_point")
        k_shape = _shape(kernel, "kernel")
        c_shape = _shape(near_correction, "near_correction")
        q_shape = _shape(point_to_space, "point_to_space")
        s_shape = _shape(singular, "singular")

        if k_shape != c_shape:
            raise ValueError("kernel and near_correction shapes differ")
        if p_shape[0] != k_shape[1]:
            raise ValueError("space_to_point range does not match kernel domain")
        if q_shape[1] != k_shape[0]:
            raise ValueError("kernel range does not match point_to_space domain")
        expected = (q_shape[0], p_shape[1])
        if s_shape != expected:
            raise ValueError(f"singular shape {s_shape} does not match operator {expected}")

        self.space_to_point = space_to_point
        self.kernel = kernel
        self.near_correction = near_correction
        self.point_to_space = point_to_space
        self.singular = singular
        self.shape = expected
        self.dtype = np.dtype(getattr(kernel, "dtype", np.float64) if dtype is None else dtype)

    @staticmethod
    def _apply(operator: Any, values: NDArray) -> NDArray:
        if values.ndim == 1 and hasattr(operator, "matvec"):
            return np.asarray(operator.matvec(values))
        if values.ndim == 2 and hasattr(operator, "matmat"):
            return np.asarray(operator.matmat(values))
        return np.asarray(operator @ values)

    def _apply_values(self, values: NDArray) -> NDArray:
        charges = self._apply(self.space_to_point, values)
        regular = self._apply(self.kernel, charges) - self._apply(
            self.near_correction, charges
        )
        return self._apply(self.point_to_space, regular) + self._apply(self.singular, values)

    def matvec(self, vector: ArrayLike) -> NDArray:
        values = np.asarray(vector, dtype=self.dtype)
        if values.shape != (self.shape[1],):
            raise ValueError(f"vector must have shape ({self.shape[1]},), got {values.shape}")
        return self._apply_values(values)

    def matmat(self, vectors: ArrayLike) -> NDArray:
        values = np.asarray(vectors, dtype=self.dtype)
        if values.ndim != 2 or values.shape[0] != self.shape[1]:
            raise ValueError(
                f"vectors must have shape ({self.shape[1]}, n_vectors), got {values.shape}"
            )
        return self._apply_values(values)

    def __matmul__(self, value: ArrayLike) -> NDArray:
        values = np.asarray(value)
        if values.ndim == 1:
            return self.matvec(values)
        if values.ndim == 2:
            return self.matmat(values)
        raise ValueError("right operand must be a vector or matrix")

    def aslinearoperator(self):
        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            matmat=self.matmat,
            dtype=self.dtype,
        )

