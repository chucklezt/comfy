#!/usr/bin/env bash
# CUDA / NVIDIA environment for ComfyUI on RTX 3090
# Previous ROCm version preserved at env.sh.rocm-backup

# Activate venv
source ~/comfy/venv/bin/activate

# CUDA allocator: expandable segments help with large model loads
# (Wan 14B, especially with GGUF where memory access patterns vary)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HuggingFace
export HF_HUB_DISABLE_TELEMETRY=1
# export HF_HOME=~/.cache/huggingface     # uncomment if you want to relocate

# Optional: tell torch to be quieter about deprecations during dev
# export PYTHONWARNINGS="ignore::DeprecationWarning"

# Sanity: print what we're about to run with
echo "ComfyUI env: $(python --version 2>&1) | torch $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
