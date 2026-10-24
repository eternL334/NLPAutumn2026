#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "$TASK_ROOT/.." && pwd)"
VENV_DIR="${MONOLM_VENV:-$REPO_ROOT/venv}"
PYTHON="$VENV_DIR/bin/python"
ARTIFACTS_DIR="${MONOLM_ARTIFACTS:-$TASK_ROOT/artifacts}"
export PYTHONPATH="$TASK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$TASK_ROOT"

run() {
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

require_environment() {
    if [[ "${DRY_RUN:-0}" != 1 && ! -x "$PYTHON" ]]; then
        echo "Environment missing. Run: bash scripts/setup.sh l40" >&2
        exit 1
    fi
}
