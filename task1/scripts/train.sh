#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
require_environment

config="$ARTIFACTS_DIR/training/recommended_config.json"
run "$PYTHON" -m accelerate.commands.launch --num_processes 1 --num_machines 1 \
    --mixed_precision bf16 --num_cpu_threads_per_process 8 --module src \
    train --config "$config" "$@"
