#!/usr/bin/env bash
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

profile="${1:-l40}"
case "$profile" in
    l40) torch_version=2.8.0; cuda=cu128; requirements=l40.txt ;;
    5080) torch_version=2.14.1; cuda=cu130; requirements=rtx5080.txt ;;
    *) echo 'Usage: bash scripts/setup.sh [l40|5080]' >&2; exit 2 ;;
esac
base_python="${PYTHON_BIN:-python3.12}"
run "$base_python" -c 'import sys, platform; assert sys.version_info[:2] == (3, 12); assert platform.system() == "Linux" and platform.machine() == "x86_64"'
if [[ ! -x "$PYTHON" ]]; then
    run "$base_python" -m venv "$VENV_DIR"
fi
run "$PYTHON" -c 'import importlib.metadata as m, sys; wrong = "flash-attn-4" if sys.argv[1] == "l40" else "flash-attn"; installed = {d.metadata["Name"].lower().replace("_", "-") for d in m.distributions()}; assert wrong not in installed, "Different Flash Attention profile already installed. Use a fresh MONOLM_VENV directory."' "$profile"
run "$PYTHON" -m pip install 'pip>=25,<27' 'setuptools==84.2.0' 'wheel==0.48.0'
run "$PYTHON" -m pip install "torch==$torch_version" --index-url "https://download.pytorch.org/whl/$cuda"
run "$PYTHON" -m pip install -r "requirements/$requirements"
run "$PYTHON" -m pip install --no-deps --no-build-isolation -e .
run "$PYTHON" -m ipykernel install --sys-prefix --name monolm --display-name MonoLM
run "$PYTHON" -m pip check
run "$PYTHON" -m src doctor --profile "$profile" --cuda
