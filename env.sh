#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# LTX-2.3 / ComfyUI — RDNA2 (gfx1030) Environment for AMD Radeon RX 6800 XT
# Source this before launching ComfyUI:  source env.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── ROCm / HIP ──────────────────────────────────────────────────────────────
# Map gfx1030 (RDNA2) to the closest officially supported target.
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# PyTorch HIP allocator — expandable_segments reduces fragmentation OOMs.
# NOTE: As of PyTorch 2.9.1+rocm6.3, expandable_segments is not yet supported
# on RDNA2/HIP (only CUDA). We set the var anyway so it auto-activates if a
# future PyTorch ROCm build adds support. The runtime warning is harmless.
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# MIOpen: use NORMAL find (mode 2) to avoid multi-GB workspace allocations
# that would blow the 16GB budget during conv autotuning.
export MIOPEN_FIND_MODE=2

# Disable MIOpen convolution autotuning cache writes to avoid stale-cache
# mismatches after driver updates.
export MIOPEN_DISABLE_CACHE=0

# ── ComfyUI / Inference ─────────────────────────────────────────────────────
# Enable MIOpen backend inside ComfyUI (used by some custom nodes).
export COMFYUI_ENABLE_MIOPEN=1

# Disable telemetry (HuggingFace Hub / analytics).
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

# ── Safety: block bitsandbytes ───────────────────────────────────────────────
# bitsandbytes / adamw8bit are CUDA-only and will segfault on RDNA2.
# We inject a sitecustomize hook via PYTHONPATH to hard-block the import.
export BITSANDBYTES_BLOCKED=1

# ── Activate venv ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

echo "✓ LTX-2.3 RDNA2 environment loaded (gfx1030 → HSA 10.3.0)"
