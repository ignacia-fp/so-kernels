"""NumPy-friendly reusable KiFMM-rs Laplace particle operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backends import load_kifmm


def _backend() -> Any:
    return load_kifmm()


def _points(value: ArrayLike, name: str, dtype: np.dtype) -> NDArray:
    points = np.asarray(value, dtype=dtype)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {points.shape}")
    return np.ascontiguousarray(points.T).reshape(-1)


@dataclass(frozen=True)
class _Config:
    expansion_order: int
    n_crit: int | None
    depth: int | None
    block_size: int
    diagonal_shift: float


class LaplaceParticleFmm:
    """Reusable single-node Laplace particle FMM.

    Each process must construct its own instance. The underlying native tree is
    reused for successive vectors within that process.
    """

    def __init__(
        self,
        sources: ArrayLike,
        targets: ArrayLike,
        *,
        expansion_order: int = 6,
        n_crit: int | None = 150,
        depth: int | None = None,
        block_size: int = 32,
        diagonal_shift: float = 0.0,
        dtype: Any = np.float64,
    ) -> None:
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise TypeError("LaplaceParticleFmm supports float32 and float64")
        if expansion_order < 2:
            raise ValueError("expansion_order must be at least 2")
        if (n_crit is None) == (depth is None):
            raise ValueError("set exactly one of n_crit and depth")

        self._sources = _points(sources, "sources", self.dtype)
        self._targets = _points(targets, "targets", self.dtype)
        self._shape = (self._targets.size // 3, self._sources.size // 3)
        if self._shape[0] == self._shape[1] and np.array_equal(
            self._sources, self._targets
        ):
            self.points = self._targets.reshape(3, -1).T.copy()
            self.symmetric = True
        else:
            self.symmetric = False
        self._config = _Config(
            expansion_order, n_crit, depth, block_size, float(diagonal_shift)
        )
        self._fmm: Any | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def _construct(self, charges: NDArray) -> None:
        backend = _backend()
        kernel = backend.LaplaceKernel(self.dtype, backend.EvalType.Value)
        tree = backend.SingleNodeTree(
            self._sources,
            self._targets,
            charges,
            n_crit=self._config.n_crit,
            depth=self._config.depth,
            prune_empty=True,
        )
        translation = backend.FftFieldTranslation(kernel, block_size=self._config.block_size)
        expansion_order = np.asarray([self._config.expansion_order], dtype=np.uintp)
        self._fmm = backend.KiFmm(expansion_order, tree, translation)

    def _apply_one(self, vector: NDArray) -> NDArray:
        charges = np.ascontiguousarray(vector, dtype=self.dtype)
        if self._fmm is None:
            self._construct(charges)
        else:
            self._fmm.clear()
            self._fmm.attach_charges_unordered(charges)
        self._fmm.evaluate()
        result = np.array(self._fmm.all_potentials_u[0, :, 0], copy=True)
        if self._config.diagonal_shift:
            if self.shape[0] != self.shape[1] or not np.array_equal(
                self._sources, self._targets
            ):
                raise ValueError(
                    "diagonal_shift requires identical source and target points"
                )
            result += self._config.diagonal_shift * vector
        return result

    def matvec(self, vector: ArrayLike) -> NDArray:
        values = np.asarray(vector, dtype=self.dtype)
        if values.shape != (self.shape[1],):
            raise ValueError(f"vector must have shape ({self.shape[1]},), got {values.shape}")
        return self._apply_one(values)

    def matmat(self, vectors: ArrayLike) -> NDArray:
        values = np.asarray(vectors, dtype=self.dtype)
        if values.ndim != 2 or values.shape[0] != self.shape[1]:
            raise ValueError(
                f"vectors must have shape ({self.shape[1]}, n_vectors), got {values.shape}"
            )
        result = np.empty((self.shape[0], values.shape[1]), dtype=self.dtype)
        for column in range(values.shape[1]):
            result[:, column] = self._apply_one(values[:, column])
        return result

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


def laplace_particle_fmm(
    sources: ArrayLike,
    targets: ArrayLike,
    **kwargs: Any,
) -> LaplaceParticleFmm:
    return LaplaceParticleFmm(sources, targets, **kwargs)
