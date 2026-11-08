import numpy as np

from so_kernels.native import BemppRsLaplaceSingleLayer


class FakeNativeOperator:
    # A refinement-zero regular sphere has eight triangular DP0 cells.
    shape = (8, 8)

    def matvec(self, values):
        return 2.0 * values

    def matmat(self, values):
        return 2.0 * values

    def transpose_matvec(self, values):
        return 3.0 * values

    def transpose_matmat(self, values):
        return 3.0 * values


def test_native_facade_forwards_vector_and_matrix_operations():
    operator = BemppRsLaplaceSingleLayer(0, _native_operator=FakeNativeOperator())
    vector = np.arange(8.0)
    matrix = np.arange(16.0).reshape(8, 2)

    np.testing.assert_allclose(operator @ vector, 2.0 * vector)
    np.testing.assert_allclose(operator @ matrix, 2.0 * matrix)
    np.testing.assert_allclose(operator.transpose_matvec(vector), 3.0 * vector)
    np.testing.assert_allclose(operator.transpose_matmat(matrix), 3.0 * matrix)


def test_native_facade_checks_shapes():
    operator = BemppRsLaplaceSingleLayer(0, _native_operator=FakeNativeOperator())

    with np.testing.assert_raises(ValueError):
        operator.matvec(np.zeros(7))
    with np.testing.assert_raises(ValueError):
        operator.matmat(np.zeros((7, 4)))
