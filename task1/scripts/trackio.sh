#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_environment
export TRACKIO_DIR="$ARTIFACTS_DIR/training/trackio"
run "$VENV_DIR/bin/trackio" show --project monolm --host 127.0.0.1 "$@"
