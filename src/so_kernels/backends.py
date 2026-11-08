"""Discovery and diagnostics for the native Bempp-rs and KiFMM wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class BackendModules:
    """The native Python modules used by :mod:`so_kernels`."""

    bempp: ModuleType
    kifmm: ModuleType
    operators: ModuleType
    bempp_cl: ModuleType


def load_bempp() -> ModuleType:
    """Load the Python wrapper installed by the Bempp-rs repository."""

    try:
        return import_module("bempp")
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise ImportError(
            "Bempp-rs is not installed in this Python environment. Run "
            "so-kernels/scripts/install-backends.sh with this interpreter."
        ) from exc


def load_kifmm() -> ModuleType:
    """Load the existing KiFMM-rs Python wrapper."""

    try:
        return import_module("kifmm_py")
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise ImportError(
            "kifmm_py is not installed in this Python environment. Run "
            "so-kernels/scripts/install-backends.sh with this interpreter."
        ) from exc


def load_operators() -> ModuleType:
    """Load the high-level Bempp-rs operator extension."""

    try:
        return import_module("bempp_rs_operators")
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise ImportError(
            "bempp_rs_operators is not installed in this Python environment. Run "
            "so-kernels/scripts/install-backends.sh with this interpreter."
        ) from exc


def load_bempp_cl() -> ModuleType:
    """Load the Bempp-cl fork used by rsrs-exps."""

    try:
        module = import_module("bempp_cl")
        # Bempp-cl does not guarantee that importing its namespace eagerly
        # attaches the public ``api`` package.
        if not hasattr(module, "api"):
            module.api = import_module("bempp_cl.api")
        return module
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise ImportError(
            "bempp_cl is not installed in this Python environment. Run "
            "so-kernels/scripts/install-backends.sh with this interpreter."
        ) from exc


def load_backends() -> BackendModules:
    """Load and return both native backend modules."""

    return BackendModules(
        bempp=load_bempp(),
        kifmm=load_kifmm(),
        operators=load_operators(),
        bempp_cl=load_bempp_cl(),
    )


def backend_status() -> dict[str, str | None]:
    """Return import/version information without raising on missing backends."""

    status: dict[str, str | None] = {}
    for distribution, module_name in (
        ("bempp-rs", "bempp"),
        ("kifmm_py", "kifmm_py"),
        ("bempp-rs-operators", "bempp_rs_operators"),
        ("bempp-cl", "bempp_cl"),
    ):
        try:
            module = import_module(module_name)
        except ImportError:
            status[distribution] = None
        else:
            status[distribution] = str(getattr(module, "__version__", "installed"))
    return status
