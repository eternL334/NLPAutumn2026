#!/usr/bin/env bash
# Compatibility entry point; all sources now use pinned revisions.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_environment
run "$PYTHON" -m src download --include-raw "$@"
