#!/usr/bin/env bash
# Creates the Python environment for ndp-cat1 and installs requirements.txt.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
ENV_PREFIX=${PGML_ENV_PREFIX:-$(dirname -- "$REPO_ROOT")/envs/pgml}
BASE_PYTHON=${PGML_BASE_PYTHON:-python3}

# zarr 3 needs Python 3.11 or newer, and numpy 2.0.2 has no wheels for 3.13.
supported_python() {
  "$1" -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] <= (3, 12))' 2>/dev/null
}

fail() {
  echo "error: $*" >&2
  exit 1
}

create_env() {
  supported_python "$BASE_PYTHON" ||
    fail "$BASE_PYTHON is not Python 3.11 or 3.12; name another with PGML_BASE_PYTHON"
  echo "Creating virtual environment at $ENV_PREFIX"
  "$BASE_PYTHON" -m venv "$ENV_PREFIX"
}

# An earlier run that failed or was interrupted can leave a broken environment.
check_env() {
  supported_python "$ENV_PREFIX/bin/python" && "$ENV_PREFIX/bin/python" -m pip --version >/dev/null 2>&1 ||
    fail "$ENV_PREFIX is not a working Python 3.11 or 3.12 environment; delete it and run this script again"
}

install_requirements() {
  echo "Installing $SCRIPT_DIR/requirements.txt"
  "$ENV_PREFIX/bin/python" -m pip install --disable-pip-version-check \
    --requirement "$SCRIPT_DIR/requirements.txt"
}

# Runs a small computation on the GPU: a PyTorch build can see a GPU that it
# has no kernels for, which only fails once a tensor is used.
report_device() {
  "$ENV_PREFIX/bin/python" - <<'PYTHON'
import torch

print(f"torch {torch.__version__}")
if not torch.cuda.is_available():
    print("No GPU visible: training will run on the CPU.")
else:
    name = torch.cuda.get_device_name(0)
    try:
        torch.ones(8, device="cuda").sum().item()
        print(f"GPU ready: {name}")
    except RuntimeError as error:
        print(f"GPU {name} is visible but unusable with this PyTorch build: {error}")
        raise SystemExit(1)
PYTHON
}

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  create_env
fi
check_env
install_requirements
report_device
echo "Environment ready: $ENV_PREFIX/bin/python"
