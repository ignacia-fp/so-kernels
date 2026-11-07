"""Serializable operator specifications and the public operator factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .bempp_cl import bempp_cl_helmholtz_combined, bempp_cl_laplace_single_layer
from .biegrid import BIEGridOperator
from .laplace import LaplaceParticleFmm
from .native import BemppRsLaplaceSingleLayer


@dataclass(frozen=True)
class OperatorSpec:
    """Pickle- and JSON-friendly description of an operator construction."""

    operator_type: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"operator_type": self.operator_type, "parameters": dict(self.parameters)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "OperatorSpec":
        return cls(str(values["operator_type"]), dict(values.get("parameters", {})))


def _biegrid(variant: str) -> Callable[..., BIEGridOperator]:
    def construct(**parameters: Any) -> BIEGridOperator:
        return BIEGridOperator(variant=variant, **parameters)

    return construct


_OPERATORS: dict[str, Callable[..., Any]] = {
    "biegrid_real_symmetric": _biegrid("real_symmetric"),
    "biegrid_real_nonsymmetric": _biegrid("real_nonsymmetric"),
    "biegrid_complex_symmetric": _biegrid("complex_symmetric"),
    "biegrid_complex_nonsymmetric": _biegrid("complex_nonsymmetric"),
    "bempp_rs_laplace_single_layer": BemppRsLaplaceSingleLayer,
    "bempp_cl_laplace_single_layer": bempp_cl_laplace_single_layer,
    "bempp_cl_helmholtz_combined": bempp_cl_helmholtz_combined,
    "laplace_particle_fmm": LaplaceParticleFmm,
}


def available_operators() -> tuple[str, ...]:
    return tuple(sorted(_OPERATORS))


def create_operator(spec: OperatorSpec | Mapping[str, Any] | str, **parameters: Any) -> Any:
    """Construct one process-local operator from a named specification."""

    if isinstance(spec, str):
        spec = OperatorSpec(spec, parameters)
    elif not isinstance(spec, OperatorSpec):
        spec = OperatorSpec.from_dict(spec)
    elif parameters:
        raise TypeError("parameters must be stored in OperatorSpec when a spec is supplied")
    try:
        constructor = _OPERATORS[spec.operator_type]
    except KeyError as exc:
        choices = ", ".join(available_operators())
        raise ValueError(f"unknown operator type {spec.operator_type!r}; choose from {choices}") from exc
    return constructor(**dict(spec.parameters))
