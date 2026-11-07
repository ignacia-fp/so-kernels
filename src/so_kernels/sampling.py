"""Generation and validation of row-oriented RSRS samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


def _apply(operator: Any, vectors: NDArray) -> NDArray:
    if hasattr(operator, "matmat"):
        return np.asarray(operator.matmat(vectors))
    return np.asarray(operator @ vectors)


@dataclass(frozen=True)
class GeneratedSamples:
    y_test: NDArray
    y_sketch: NDArray
    z_test: NDArray | None = None
    z_sketch: NDArray | None = None
    seed: int | None = None

    @property
    def sample_count(self) -> int:
        return int(self.y_test.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.y_test.shape[1])


def validate_samples(samples: GeneratedSamples, *, require_z: bool = False) -> None:
    if samples.y_test.ndim != 2 or samples.y_test.shape != samples.y_sketch.shape:
        raise ValueError("y_test and y_sketch must be equally shaped matrices")
    if samples.y_test.dtype != samples.y_sketch.dtype:
        raise TypeError("y_test and y_sketch must have matching dtypes")
    if (samples.z_test is None) != (samples.z_sketch is None):
        raise ValueError("z_test and z_sketch must both be present or absent")
    if require_z and samples.z_test is None:
        raise ValueError("nonsymmetric sampling requires z samples")
    if samples.z_test is not None:
        if samples.z_test.shape != samples.y_test.shape:
            raise ValueError("y and z sample families must have matching shapes")
        if samples.z_sketch.shape != samples.z_test.shape:
            raise ValueError("z_test and z_sketch must have matching shapes")
        if samples.z_test.dtype != samples.y_test.dtype:
            raise TypeError("all sample arrays must have matching dtypes")


def generate_samples(
    operator: Any,
    *,
    sample_count: int,
    seed: int,
    transpose_operator: Any | None = None,
    dtype: Any = np.float64,
) -> GeneratedSamples:
    """Generate Gaussian RSRS samples.

    `transpose_operator` must implement the non-conjugating transpose. It is
    intentionally not inferred from SciPy's `rmatvec`, which represents the
    conjugate transpose for complex operators.
    """

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rows, columns = map(int, operator.shape)
    if rows != columns:
        raise ValueError("RSRS currently requires a square operator")

    scalar_dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    if np.issubdtype(scalar_dtype, np.complexfloating):
        omega_y = (
            rng.standard_normal((columns, sample_count))
            + 1j * rng.standard_normal((columns, sample_count))
        ).astype(scalar_dtype)
    else:
        omega_y = rng.standard_normal((columns, sample_count)).astype(scalar_dtype)
    y = _apply(operator, omega_y)

    z_test = None
    z_sketch = None
    if transpose_operator is not None:
        if np.issubdtype(scalar_dtype, np.complexfloating):
            omega_z = (
                rng.standard_normal((columns, sample_count))
                + 1j * rng.standard_normal((columns, sample_count))
            ).astype(scalar_dtype)
        else:
            omega_z = rng.standard_normal((columns, sample_count)).astype(scalar_dtype)
        z = _apply(transpose_operator, omega_z)
        z_test = np.ascontiguousarray(omega_z.T)
        z_sketch = np.ascontiguousarray(z.T)

    samples = GeneratedSamples(
        y_test=np.ascontiguousarray(omega_y.T),
        y_sketch=np.ascontiguousarray(y.T),
        z_test=z_test,
        z_sketch=z_sketch,
        seed=seed,
    )
    validate_samples(samples)
    return samples

