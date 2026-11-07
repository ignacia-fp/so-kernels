"""Matrix-free Bempp/KiFMM operators and RSRS sample generation."""

from .backends import (
    BackendModules,
    backend_status,
    load_backends,
    load_bempp,
    load_bempp_cl,
    load_kifmm,
    load_operators,
)
from .bempp_cl import (
    BemppClDiscreteOperator,
    bempp_cl_helmholtz_combined,
    bempp_cl_laplace_single_layer,
)
from .biegrid import (
    BIEGridOperator,
    biegrid_complex_nonsymmetric,
    biegrid_complex_symmetric,
    biegrid_real_nonsymmetric,
    biegrid_real_symmetric,
)
from .laplace import LaplaceParticleFmm, laplace_particle_fmm
from .factory import OperatorSpec, available_operators, create_operator
from .native import BemppRsLaplaceSingleLayer
from .sampling import GeneratedSamples, generate_samples, validate_samples
from .single_layer import LaplaceSingleLayerFmm

__all__ = [
    "BackendModules",
    "BemppRsLaplaceSingleLayer",
    "BemppClDiscreteOperator",
    "BIEGridOperator",
    "GeneratedSamples",
    "LaplaceParticleFmm",
    "LaplaceSingleLayerFmm",
    "OperatorSpec",
    "generate_samples",
    "backend_status",
    "laplace_particle_fmm",
    "available_operators",
    "create_operator",
    "validate_samples",
    "load_backends",
    "load_bempp",
    "load_bempp_cl",
    "load_kifmm",
    "load_operators",
    "bempp_cl_helmholtz_combined",
    "bempp_cl_laplace_single_layer",
    "biegrid_real_symmetric",
    "biegrid_real_nonsymmetric",
    "biegrid_complex_symmetric",
    "biegrid_complex_nonsymmetric",
]
