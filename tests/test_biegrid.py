import numpy as np
import pytest

from so_kernels.biegrid import BIEGridOperator
from so_kernels.sampling import generate_samples


@pytest.mark.parametrize(
    "variant",
    [
        "real_symmetric",
        "real_nonsymmetric",
        "complex_symmetric",
        "complex_nonsymmetric",
    ],
)
def test_biegrid_apply_transpose_and_samples(variant):
    operator = BIEGridOperator(cells_per_axis=6, variant=variant)
    dense = operator.dense()
    rng = np.random.default_rng(7)
    values = rng.standard_normal((operator.dimension, 3)).astype(operator.dtype)
    if np.issubdtype(operator.dtype, np.complexfloating):
        values += 1j * rng.standard_normal(values.shape)

    np.testing.assert_allclose(operator.matmat(values), dense @ values, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        operator.transpose_operator.matmat(values), dense.T @ values, rtol=2e-13, atol=2e-13
    )

    samples = generate_samples(
        operator,
        transpose_operator=None if operator.symmetric else operator.transpose_operator,
        sample_count=12,
        seed=123,
        dtype=operator.dtype,
    )
    np.testing.assert_allclose(samples.y_sketch, (dense @ samples.y_test.T).T)
    if operator.symmetric:
        assert samples.z_test is None
    else:
        np.testing.assert_allclose(samples.z_sketch, (dense.T @ samples.z_test.T).T)


def test_complex_symmetric_is_transpose_symmetric_but_not_hermitian():
    matrix = BIEGridOperator(cells_per_axis=6, variant="complex_symmetric").dense()
    np.testing.assert_allclose(matrix, matrix.T, rtol=2e-13, atol=2e-13)
    assert np.linalg.norm(matrix - matrix.conj().T) / np.linalg.norm(matrix) > 1e-6
