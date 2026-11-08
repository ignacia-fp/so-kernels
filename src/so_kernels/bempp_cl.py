"""Bempp-cl operators exposed through the common so-kernels interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _apply_block(operator: Any, values: NDArray, rows: int) -> NDArray:
    """Apply a Bempp weak form while preserving a block's column ordering."""

    try:
        result = operator @ values
    except (TypeError, ValueError):
        result = operator * values
    output = np.asarray(result)
    columns = 1 if values.ndim == 1 else values.shape[1]
    expected = rows * columns
    if output.size != expected:
        raise RuntimeError(
            f"Bempp-cl returned {output.size} entries, expected {expected}"
        )
    if values.ndim == 1:
        return output.reshape(rows)
    if output.shape != (rows, columns):
        output = output.reshape((rows, columns), order="F")
    return output


class BemppClDiscreteOperator:
    """A Bempp-cl weak form with an explicit algebraic transpose.

    ``transpose_operator`` must represent :math:`A^T`, not :math:`A^*`.
    Keeping it explicit prevents complex RSRS Z samples from accidentally
    using the Hermitian adjoint.
    """

    def __init__(
        self,
        operator: Any,
        transpose_operator: Any,
        *,
        dimension: int,
        dtype: Any,
        points: ArrayLike | None = None,
        symmetric: bool = False,
    ) -> None:
        self._operator = operator
        self._transpose_operator = transpose_operator
        self.shape = (int(dimension), int(dimension))
        self.dtype = np.dtype(dtype)
        self.symmetric = bool(symmetric)
        if points is not None:
            coordinates = np.asarray(points, dtype=np.float64)
            if coordinates.shape != (self.shape[0], 3):
                raise ValueError(
                    f"points must have shape ({self.shape[0]}, 3), got {coordinates.shape}"
                )
            self.points = coordinates

    @property
    def dimension(self) -> int:
        return self.shape[0]

    @property
    def transpose_operator(self) -> "_BemppClTranspose":
        """Return the explicit non-conjugating transpose as an operator."""

        return _BemppClTranspose(self)

    def matvec(self, vector: ArrayLike) -> NDArray:
        values = np.asarray(vector, dtype=self.dtype)
        if values.shape != (self.dimension,):
            raise ValueError(
                f"vector must have shape ({self.dimension},), got {values.shape}"
            )
        return _apply_block(self._operator, values, self.dimension)

    def matmat(self, vectors: ArrayLike) -> NDArray:
        values = np.asarray(vectors, dtype=self.dtype)
        if values.ndim != 2 or values.shape[0] != self.dimension:
            raise ValueError(
                f"vectors must have shape ({self.dimension}, n_vectors), got {values.shape}"
            )
        return _apply_block(self._operator, values, self.dimension)

    def transpose_matvec(self, vector: ArrayLike) -> NDArray:
        values = np.asarray(vector, dtype=self.dtype)
        if values.shape != (self.dimension,):
            raise ValueError(
                f"vector must have shape ({self.dimension},), got {values.shape}"
            )
        return _apply_block(self._transpose_operator, values, self.dimension)

    def transpose_matmat(self, vectors: ArrayLike) -> NDArray:
        values = np.asarray(vectors, dtype=self.dtype)
        if values.ndim != 2 or values.shape[0] != self.dimension:
            raise ValueError(
                f"vectors must have shape ({self.dimension}, n_vectors), got {values.shape}"
            )
        return _apply_block(self._transpose_operator, values, self.dimension)

    def __matmul__(self, value: ArrayLike) -> NDArray:
        values = np.asarray(value)
        if values.ndim == 1:
            return self.matvec(values)
        if values.ndim == 2:
            return self.matmat(values)
        raise ValueError("right operand must be a vector or matrix")

    def aslinearoperator(self):
        from scipy.sparse.linalg import LinearOperator

        # SciPy's rmatvec convention is the adjoint, so it must not be wired to
        # our algebraic transpose for complex operators.
        return LinearOperator(
            self.shape,
            matvec=self.matvec,
            matmat=self.matmat,
            dtype=self.dtype,
        )


class _BemppClTranspose:
    def __init__(self, parent: BemppClDiscreteOperator) -> None:
        self._parent = parent
        self.shape = parent.shape
        self.dtype = parent.dtype

    def matvec(self, vector: ArrayLike) -> NDArray:
        return self._parent.transpose_matvec(vector)

    def matmat(self, vectors: ArrayLike) -> NDArray:
        return self._parent.transpose_matmat(vectors)

    def __matmul__(self, value: ArrayLike) -> NDArray:
        values = np.asarray(value)
        if values.ndim == 1:
            return self.matvec(values)
        return self.matmat(values)


def _configure_bempp(
    *, precision: str, expansion_order: int, regular_quadrature: int
) -> Any:
    from .backends import load_bempp_cl

    if precision not in {"single", "double"}:
        raise ValueError("precision must be 'single' or 'double'")
    api = load_bempp_cl().api
    api.DEFAULT_PRECISION = precision
    api.GLOBAL_PARAMETERS.fmm.expansion_order = int(expansion_order)
    api.GLOBAL_PARAMETERS.quadrature.regular = int(regular_quadrature)
    return api


def _load_grid(api: Any, grid: Any) -> Any:
    if isinstance(grid, (str, Path)):
        return api.import_grid(str(grid))
    return grid


def _grid_centroids(grid: Any, dimension: int) -> NDArray[np.float64]:
    points = np.asarray(grid.centroids, dtype=np.float64)
    if points.shape == (dimension, 3):
        return points
    if points.shape == (3, dimension):
        return points.T
    raise ValueError(
        f"grid centroids must have shape ({dimension}, 3) or (3, {dimension}), "
        f"got {points.shape}"
    )


def bempp_cl_laplace_single_layer(
    grid: Any,
    *,
    assembler: str = "exafmm",
    precision: str = "double",
    expansion_order: int = 8,
    regular_quadrature: int = 2,
) -> BemppClDiscreteOperator:
    """Build the DP0 Laplace single-layer operator used by rsrs-exps."""

    api = _configure_bempp(
        precision=precision,
        expansion_order=expansion_order,
        regular_quadrature=regular_quadrature,
    )
    grid = _load_grid(api, grid)
    space = api.function_space(grid, "DP", 0)
    operator = api.operators.boundary.laplace.single_layer(
        space, space, space, assembler=assembler
    ).weak_form()
    dtype = np.float32 if precision == "single" else np.float64
    return BemppClDiscreteOperator(
        operator,
        operator.T,
        dimension=space.global_dof_count,
        dtype=dtype,
        points=_grid_centroids(grid, space.global_dof_count),
        symmetric=True,
    )


def bempp_cl_helmholtz_combined(
    grid: Any,
    kappa: float,
    *,
    assembler: str = "exafmm",
    precision: str = "double",
    expansion_order: int = 8,
    regular_quadrature: int = 2,
) -> BemppClDiscreteOperator:
    """Build ``0.5 I + D' - i k S`` and its algebraic transpose.

    This matches ``BemppClHelmholtzCombined`` in rsrs-exps. In particular,
    ``A.T = 0.5 I + (D').T - i k S.T``: the scalar is not conjugated.
    """

    api = _configure_bempp(
        precision=precision,
        expansion_order=expansion_order,
        regular_quadrature=regular_quadrature,
    )
    grid = _load_grid(api, grid)
    space = api.function_space(grid, "DP", 0)
    identity = api.operators.boundary.sparse.identity(space, space, space).weak_form()
    single_layer = api.operators.boundary.helmholtz.single_layer(
        space, space, space, kappa, assembler=assembler
    ).weak_form()
    adjoint_double_layer = api.operators.boundary.helmholtz.adjoint_double_layer(
        space, space, space, kappa, assembler=assembler
    ).weak_form()
    operator = 0.5 * identity + adjoint_double_layer - 1j * kappa * single_layer
    transpose = (
        0.5 * identity
        + adjoint_double_layer.T
        - 1j * kappa * single_layer.T
    )
    dtype = np.complex64 if precision == "single" else np.complex128
    return BemppClDiscreteOperator(
        operator,
        transpose,
        dimension=space.global_dof_count,
        dtype=dtype,
        points=_grid_centroids(grid, space.global_dof_count),
    )
