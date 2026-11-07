import numpy as np

from so_kernels.bempp_cl import BemppClDiscreteOperator


class FlatteningOperator:
    """Mimic a backend returning a column-major flattened block."""

    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)

    def __matmul__(self, values):
        result = self.matrix @ values
        if result.ndim == 2:
            return result.reshape(-1, order="F")
        return result


def test_bempp_cl_adapter_preserves_block_order_and_transpose():
    matrix = np.array([[1 + 2j, 3 - 1j], [4j, 2 + 5j]])
    operator = BemppClDiscreteOperator(
        FlatteningOperator(matrix),
        FlatteningOperator(matrix.T),
        dimension=2,
        dtype=np.complex128,
    )
    vectors = np.array([[1 - 1j, 2 + 1j], [3 + 2j, -1j]])

    np.testing.assert_allclose(operator @ vectors, matrix @ vectors)
    np.testing.assert_allclose(operator.transpose_operator @ vectors, matrix.T @ vectors)
    assert not np.allclose(matrix.T @ vectors, matrix.conj().T @ vectors)
