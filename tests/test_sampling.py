import numpy as np

from so_kernels.sampling import generate_samples


class MatrixOperator:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix)
        self.shape = self.matrix.shape

    def matmat(self, values):
        return self.matrix @ values


def test_complex_samples_use_explicit_nonconjugating_transpose():
    matrix = np.array([[1 + 2j, 3], [4j, 5 - 1j]])
    samples = generate_samples(
        MatrixOperator(matrix),
        transpose_operator=MatrixOperator(matrix.T),
        sample_count=4,
        seed=17,
        dtype=np.complex128,
    )
    np.testing.assert_allclose(samples.y_sketch, (matrix @ samples.y_test.T).T)
    np.testing.assert_allclose(samples.z_sketch, (matrix.T @ samples.z_test.T).T)


def test_seed_reproduces_samples_and_different_seed_changes_them():
    operator = MatrixOperator(np.eye(3))
    first = generate_samples(operator, sample_count=5, seed=3)
    repeated = generate_samples(operator, sample_count=5, seed=3)
    different = generate_samples(operator, sample_count=5, seed=4)
    np.testing.assert_array_equal(first.y_test, repeated.y_test)
    assert not np.array_equal(first.y_test, different.y_test)

