#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/install-backends.sh [options]

Install the low-level Bempp-rs wrapper, the persistent high-level Bempp-rs
operator extension, KiFMM Python wrapper, and so-kernels façade inside one
explicitly selected Python environment.

Options:
  --python PATH       Python interpreter to target (default: python3)
  --workspace PATH    Parent containing bempp-rs, kifmm-for-rsrs, and
                      so-kernels (default: parent of this repository)
  --bempp-cl PATH     Bempp-cl checkout to install (default:
                      WORKSPACE/rsrs-exps/external/bempp-cl)
  --editable          Install all three repositories in editable mode
  --no-build-isolation
                      Pass --no-build-isolation to native package installs
  -h, --help          Show this help
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$script_dir")"
workspace_dir="$(dirname "$repo_dir")"
python_executable="python3"
editable=false
no_build_isolation=false
bempp_cl_repo=""

while (($#)); do
    case "$1" in
        --python)
            python_executable="$2"
            shift 2
            ;;
        --workspace)
            workspace_dir="$2"
            shift 2
            ;;
        --editable)
            editable=true
            shift
            ;;
        --bempp-cl)
            bempp_cl_repo="$2"
            shift 2
            ;;
        --no-build-isolation)
            no_build_isolation=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

bempp_repo="$workspace_dir/bempp-rs"
kifmm_repo="$workspace_dir/kifmm-for-rsrs/kifmm"
operators_repo="$bempp_repo/python-operators"
facade_repo="$workspace_dir/so-kernels"
if [[ -z "$bempp_cl_repo" ]]; then
    bempp_cl_repo="$workspace_dir/rsrs-exps/external/bempp-cl"
fi

for path in "$bempp_repo" "$operators_repo" "$kifmm_repo" "$bempp_cl_repo" "$facade_repo"; do
    if [[ ! -d "$path" ]]; then
        echo "Required repository directory not found: $path" >&2
        exit 1
    fi
done

"$python_executable" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"The current KiFMM wrapper requires Python 3.10; got {sys.version.split()[0]}"
    )
print(f"Installing into {sys.executable} ({sys.version.split()[0]})")
PY

pip_args=(--upgrade)
if $editable; then
    pip_args+=(--editable)
fi
native_args=()
if $no_build_isolation; then
    native_args+=(--no-build-isolation)
fi

"$python_executable" -m pip install "${pip_args[@]}" "${native_args[@]}" "$bempp_repo"
"$python_executable" -m pip install "${pip_args[@]}" "${native_args[@]}" "$operators_repo"
"$python_executable" -m pip install "${pip_args[@]}" "${native_args[@]}" "$kifmm_repo"
"$python_executable" -m pip install "${pip_args[@]}" "$bempp_cl_repo"
"$python_executable" -m pip install "${pip_args[@]}" "$facade_repo[dev]"

"$python_executable" - <<'PY'
import bempp
import bempp_rs_operators
import bempp_cl
import kifmm_py
import so_kernels

print("Backend imports succeeded:")
print(f"  bempp:        {bempp.__file__}")
print(f"  operators:    {bempp_rs_operators.__file__}")
print(f"  bempp_cl:     {bempp_cl.__file__}")
print(f"  kifmm_py:     {kifmm_py.__file__}")
print(f"  so_kernels:{so_kernels.__file__}")
print(f"  status:       {so_kernels.backend_status()}")
PY
