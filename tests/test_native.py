import numpy as np

from so_kernels.native import BemppRsLaplaceSingleLayer


class FakeNativeOperator:
    shape = (3, 3)

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
    vector = np.arange(3.0)
    matrix = np.arange(6.0).reshape(3, 2)

    np.testing.assert_allclose(operator @ vector, 2.0 * vector)
    np.testing.assert_allclose(operator @ matrix, 2.0 * matrix)
    np.testing.assert_allclose(operator.transpose_matvec(vector), 3.0 * vector)
    np.testing.assert_allclose(operator.transpose_matmat(matrix), 3.0 * matrix)


def test_native_facade_checks_shapes():
    operator = BemppRsLaplaceSingleLayer(0, _native_operator=FakeNativeOperator())

    with np.testing.assert_raises(ValueError):
        operator.matvec(np.zeros(2))
    with np.testing.assert_raises(ValueError):
        operator.matmat(np.zeros((2, 4)))
