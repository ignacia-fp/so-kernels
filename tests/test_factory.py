import pytest

from so_kernels import OperatorSpec, available_operators, create_operator


def test_biegrid_operator_spec_round_trip():
    spec = OperatorSpec("biegrid_complex_nonsymmetric", {"cells_per_axis": 6})
    restored = OperatorSpec.from_dict(spec.to_dict())
    operator = create_operator(restored)
    assert operator.shape == (49, 49)
    assert operator.dtype.name == "complex128"
    assert not operator.symmetric


def test_unknown_operator_reports_choices():
    with pytest.raises(ValueError, match="unknown operator type"):
        create_operator("missing")
    assert "biegrid_real_symmetric" in available_operators()
