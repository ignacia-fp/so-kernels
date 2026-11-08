# so-kernels

`so-kernels` is the Python-facing operator and sampling package used by the
RSRS workflow. It composes two independent native backends:

- [Bempp-rs](https://github.com/ignacia-fp/bempp-rs-for-rsrs) for boundary
  spaces, quadrature maps, singular terms, and near-field corrections;
- [KiFMM-rs](https://github.com/ignacia-fp/kifmm-for-rsrs) for reusable
  particle FMM evaluations.
- [Bempp-cl-rsrs](https://github.com/ignacia-fp/bempp-cl-rsrs) for the
  established Python/ExaFMM operators used by `rsrs-exps`.

The backends remain separate repositories. This package owns their stable
Python API, matrix-free operator composition, sample generation, validation,
and cache-facing conventions.

## Local development

The expected sibling layout is:

```text
WIP/
  bempp-rs/
  kifmm-for-rsrs/
  so-kernels/
  rsrs/
  rsrs-exps/
```

Using a Python 3.10 environment, run:

```bash
bash scripts/install-backends.sh \
    --python /path/to/venv/bin/python \
    --editable
```

This builds the existing low-level wrappers, the persistent high-level
Bempp-rs operator extension, and this façade in the same target environment.
The script verifies that `bempp`, `bempp_rs_operators`, `kifmm_py`,
`bempp_cl`, and `so_kernels` can all be imported by the selected
interpreter. By default it installs the exact Bempp-cl submodule checkout in
`rsrs-exps/external/bempp-cl`; use `--bempp-cl PATH` to select another clone.

The façade does not duplicate either native binding. It imports them freely:

```python
from so_kernels import load_backends

native = load_backends()
native.bempp
native.kifmm
native.operators
native.bempp_cl
```

The complete Bempp-rs Laplace single-layer operator can be constructed and
used without moving its grids, spaces, near corrections or FMM state into
Python:

```python
from so_kernels import BemppRsLaplaceSingleLayer

A = BemppRsLaplaceSingleLayer(
    refinement_level=5,
    function_space_degree=0,
    expansion_order=8,
)
y = A @ x
Y = A @ omega
Z = A.transpose_matmat(eta)
scipy_operator = A.aslinearoperator()
```

The matching Bempp-cl operators use the same interface:

```python
from so_kernels import (
    bempp_cl_helmholtz_combined,
    bempp_cl_laplace_single_layer,
)

laplace = bempp_cl_laplace_single_layer(
    "grid.msh", assembler="exafmm", expansion_order=8
)
combined = bempp_cl_helmholtz_combined(
    "grid.msh",
    kappa=2.0,
    assembler="exafmm",
    precision="double",
    expansion_order=8,
    ncrit=400,
    near_field_representation="evaluate",
)

Y = combined @ omega_y
Z = combined.transpose_operator @ omega_z
```

For the complex combined operator, `transpose_operator` is the explicit
non-conjugating transpose required by RSRS, not SciPy's adjoint convention.

Bempp-cl constructor defaults match Bempp-cl itself: double precision,
expansion order 5, `ncrit=400`, near fields evaluated directly, FMM depth 4,
and regular/singular quadrature order 4. All are JSON-serializable operator
parameters and can be overridden per experiment.

The direct KiFMM particle constructor retains the defaults used by this
project: expansion order 6, `n_crit=150`, FFT block size 32, automatic depth,
and empty-tree pruning. It exposes `expansion_order`, `n_crit` or `depth`,
`block_size`, `prune_empty`, and `dtype` as operator parameters.

The native object owns one persistent Rust worker. Arrays cross the boundary,
but the operator and KiFMM evaluator are built once and reused. The initial
native constructor supports real-valued Laplace single-layer operators on a
regular sphere; arbitrary meshes and complex Helmholtz operators are the next
backend extensions.

A normal published/Git installation can instead use the `backends` extra.

## Sample convention

One sample is stored per row:

```text
y_test[i, :]   = omega_i
y_sketch[i, :] = A @ omega_i
z_test[i, :]   = eta_i
z_sketch[i, :] = A.T @ eta_i
```

`A.T` is the non-conjugating transpose. For complex nonsymmetric operators the
transpose operator must therefore be supplied explicitly; this package never
silently substitutes the Hermitian adjoint.
