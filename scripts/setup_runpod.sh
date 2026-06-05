#!/usr/bin/env bash
#
# Full setup for runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#
# Why this script exists instead of a single "pip install":
#   - Steps 1-2: Standard PyPI packages; installed from locked requirements files.
#   - Step 3:    datatrove requires a specific git branch not on PyPI.
#   - Step 4:    flash-attn and grouped_gemm are CUDA extensions that compile against
#               PyTorch headers already in the container. --no-build-isolation bypasses
#               pip's isolated build env so they can find those headers. They cannot
#               go in a standard lockfile for the same reason.
#   - Step 5:    NeMo dataset helpers are a C++ pybind11 extension compiled via make;
#               outside the pip system entirely.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "=== Step 1: Install locked PyPI dependencies ==="
uv pip install -r requirements/dev.txt

echo "=== Step 2: Install nanotron (editable, deps already handled by step 1) ==="
uv pip install -e . --no-deps

echo "=== Step 3: Install optional PyPI extras (nanosets + s3) ==="
# wandb, numba, s5cmd, boto3, s3fs — pure Python, just not in core lockfile
uv pip install psutil wandb numba s5cmd boto3 s3fs
uv pip install transformers==4.51.3

echo "=== Step 4: Install datatrove from the required git branch ==="
# The main branch breaks in this RunPod environment; nouamane/avoid-s3 is required
uv pip install "datatrove[io,processing] @ git+https://github.com/huggingface/datatrove.git@nouamane/avoid-s3"

echo "=== Step 5: Build CUDA extension packages ==="
# ninja accelerates the flash-attn compile (~15 min without it, ~5 min with)
uv pip install ninja
uv pip install "flash-attn>=2.5.0,<2.7.0" --no-build-isolation
pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main

echo "=== Step 6: Build NeMo dataset C++ helpers ==="
# Compiles a pybind11 extension used by the NeMo-format indexed dataset loader
cd src/nanotron/data/nemo_dataset
make clean 2>/dev/null || true
make
cd "$REPO_ROOT"

echo ""
echo "=== Setup complete ==="
echo "Verify with: python -c \"import nanotron; import flash_attn; import grouped_gemm; import datatrove; print('OK')\""
