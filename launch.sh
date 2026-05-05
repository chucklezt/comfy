#!/usr/bin/env bash
# Launch ComfyUI on RTX 3090
set -e

source ~/comfy/env.sh

cd ~/comfy/ComfyUI

# Flag rationale:
#   --listen                     bind to 0.0.0.0 for LAN access
#   --use-pytorch-cross-attention safe default; switch to
#                                 --use-sage-attention later if installed
#   (no --lowvram)               24 GB is plenty; lowvram hurts perf
#   --reserve-vram 1.0           leave 1 GB for system/display
python main.py \
    --listen \
    --use-pytorch-cross-attention \
    --reserve-vram 1.0 \
    "$@"
