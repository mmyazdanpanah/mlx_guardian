#!/bin/bash
# MLX Guardian launcher for local checkouts.
# Finds the project venv automatically and forwards CLI options to mlx_guardian.py.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${MLX_GUARDIAN_PYTHON:-}"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "ERROR: Python 3 not found." >&2
    exit 127
  fi
fi

exec "$PYTHON_BIN" "$ROOT_DIR/mlx_guardian.py" "$@"
