#!/usr/bin/env bash
# Colab dependency setup for the instructGPT harness (idempotent).
#
# Run from the instructGPT repo root:
#     bash colab/setup.sh
#
# Installs the exact-ish dependency floor, checks the GPU, and validates the
# frozen benchmark before any generation starts. Voice-out loud at every step
# so a Colab cell shows progress.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "== setup: pip =="
python -m pip install -q --upgrade "transformers>=4.45" accelerate bitsandbytes

echo "== setup: gpu sanity =="
python - <<'PY'
import sys
import torch

cuda = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if cuda else "no GPU"
print(f"torch {torch.__version__} | cuda available: {cuda} | {name}")
if not cuda:
    sys.exit("GPU runtime required (Runtime > Change runtime type > T4).")
import transformers  # noqa: E402

print(f"transformers {transformers.__version__}")

try:
    import bitsandbytes  # noqa: E402

    print(f"bitsandbytes {bitsandbytes.__version__}")
except ImportError:
    print("bitsandbytes missing; --quantize will fail. pip install bitsandbytes")
PY

echo "== setup: frozen benchmark check =="
python validate_benchmark.py

echo "setup ok"