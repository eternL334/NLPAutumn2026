#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

stage="${1:-help}"
if [[ "$stage" != help && "$stage" != -h && "$stage" != --help ]]; then
    require_environment
fi
profile="${2:-l40}"
case "$profile" in l40|5080) ;; *) echo 'Profile must be l40 or 5080' >&2; exit 2 ;; esac
config="$TASK_ROOT/configs/train_$profile.json"

case "$stage" in
    prepare)
        run "$PYTHON" -m src download
        run "$PYTHON" -m src corpus build
        run "$PYTHON" -m src tokenizer selected
        run "$PYTHON" -m src tokenize
        ;;
    calibrate)
        run "$PYTHON" -m src doctor --profile "$profile" --cuda
        run "$PYTHON" -m src check --config "$config"
        run "$PYTHON" -m src benchmark --config "$config"
        ;;
    compare-tokenizers)
        run "$PYTHON" -m src tokenizer run
        ;;
    corpus-analysis)
        run "$PYTHON" -m src download --include-raw
        run "$PYTHON" -m src corpus build
        run "$PYTHON" -m src stats
        run "$PYTHON" -m src audit prepare
        ;;
    help|-h|--help)
        echo 'Usage: bash scripts/pipeline.sh {prepare|calibrate|compare-tokenizers|corpus-analysis} [l40|5080]'
        echo 'Training is a separate explicit command: bash scripts/train.sh'
        ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
