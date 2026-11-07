"""High-level façade for complete operators implemented inside Bempp-rs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BemppRsLaplaceSingleLayer:
    """Persistent matrix-free Laplace single-layer operator from Bempp-rs.

    The grid, function space, near correction, singular quadrature and KiFMM
    evaluator all remain owned by one Rust worker. Only vectors and result
    vectors cross the Python boundary.
    """

    def __init__(
        self,
        refinement_level: int,
        *,
        function_space_degree: int = 0,
        quadrature_degree: int = 6,
        local_tree_depth: int = 1,
        global_tree_depth: int = 6,
        expansion_order: int = 6,
        _native_operator: Any | None = None,
    ) -> None:
        if _native_operator is None:
            try:
                from bempp_rs_operators import LaplaceSingleLayer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "The high-level Bempp-rs operator extension is not installed. "
                    "Run so-kernels/scripts/install-backends.sh with this Python."
                ) from exc
            _native_operator = LaplaceSingleLayer(
                refinement_level,
                function_space_degree=function_space_degree,
                quadrature_degree=quadrature_degree,
                local_tree_depth=local_tree_depth,
                global_tree_depth=global_tree_depth,
                expansion_order=expansion_order,
            )
        self._operator = _native_operator
        self.shape = tuple(int(value) for value in self._operator.shape)
        self.dtype = np.dtype(np.float64)

    @property
    def dimension(self) -> int:
        return self.shape[0]

    def matvec(self, vector: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(vector, dtype=self.dtype)
        if values.shape != (self.shape[1],):
            raise ValueError(f"vector must have shape ({self.shape[1]},), got {values.shape}")
        return np.asarray(self._operator.matvec(values))

    def matmat(self, vectors: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(vectors, dtype=self.dtype)
        if values.ndim != 2 or values.shape[0] != self.shape[1]:
            raise ValueError(
                f"vectors must have shape ({self.shape[1]}, n_vectors), got {values.shape}"
            )
        return np.asarray(self._operator.matmat(values))

    def transpose_matvec(self, vector: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(vector, dtype=self.dtype)
        if values.shape != (self.shape[0],):
            raise ValueError(f"vector must have shape ({self.shape[0]},), got {values.shape}")
        return np.asarray(self._operator.transpose_matvec(values))

    def transpose_matmat(self, vectors: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(vectors, dtype=self.dtype)
        if values.ndim != 2 or values.shape[0] != self.shape[0]:
            raise ValueError(
                f"vectors must have shape ({self.shape[0]}, n_vectors), got {values.shape}"
            )
        return np.asarray(self._operator.transpose_matmat(values))

    def __matmul__(self, value: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(value)
        if values.ndim == 1:
            return self.matvec(values)
        if values.ndim == 2:
            return self.matmat(values)
        raise ValueError("right operand must be a vector or matrix")

    def aslinearoperator(self):
        """Return a SciPy ``LinearOperator`` without copying the native state."""

        from scipy.sparse.linalg import LinearOperator

        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            matmat=self.matmat,
            rmatvec=self.transpose_matvec,
            rmatmat=self.transpose_matmat,
            dtype=self.dtype,
        )
