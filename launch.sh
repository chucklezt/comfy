#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch ComfyUI — LTX-2.3 on RDNA2 (RX 6800 XT, 16GB)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load RDNA2 environment variables.
source "$SCRIPT_DIR/env.sh"

# ── Pre-flight checks ───────────────────────────────────────────────────────
python "$SCRIPT_DIR/diagnostics.py" --quick || {
    echo "ERROR: Diagnostics failed. Fix the issues above before launching."
    exit 1
}

# ── Launch ───────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR/ComfyUI"

exec python main.py \
    --listen \
    --use-pytorch-cross-attention \
    --reserve-vram 2.5 \
    --lowvram \
    "$@"
