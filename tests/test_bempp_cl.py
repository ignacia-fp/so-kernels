import numpy as np
from types import SimpleNamespace

from so_kernels.bempp_cl import BemppClDiscreteOperator, _configure_bempp


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


def test_bempp_adapter_exposes_operator_supplied_right_hand_sides():
    matrix = np.eye(2)
    operator = BemppClDiscreteOperator(
        matrix,
        matrix,
        dimension=2,
        dtype=np.complex128,
        right_hand_side_provider=lambda count: np.arange(2 * count).reshape(2, count),
    )
    np.testing.assert_array_equal(
        operator.right_hand_sides(count=3), np.arange(6).reshape(2, 3)
    )


def test_bempp_parameters_are_explicitly_configurable(monkeypatch):
    fmm = SimpleNamespace()
    quadrature = SimpleNamespace()
    api = SimpleNamespace(
        DEFAULT_PRECISION=None,
        GLOBAL_PARAMETERS=SimpleNamespace(fmm=fmm, quadrature=quadrature),
    )
    monkeypatch.setattr(
        "so_kernels.backends.load_bempp_cl",
        lambda: SimpleNamespace(api=api),
    )

    configured = _configure_bempp(
        precision="single",
        expansion_order=8,
        ncrit=256,
        near_field_representation="sparse",
        fmm_backend="exafmm",
        fmm_depth=7,
        fmm_debug=True,
        fmm_dense_evaluation=True,
        regular_quadrature=6,
        singular_quadrature=8,
    )

    assert configured is api
    assert api.DEFAULT_PRECISION == "single"
    assert fmm.expansion_order == 8
    assert fmm.ncrit == 256
    assert fmm.near_field_representation == "sparse"
    assert fmm.backend == "exafmm"
    assert fmm.depth == 7
    assert fmm.debug is True
    assert fmm.dense_evaluation is True
    assert quadrature.regular == 6
    assert quadrature.singular == 8
