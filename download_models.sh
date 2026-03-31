#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Download LTX-2.3 models (16GB-VRAM optimized GGUF quantizations)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ComfyUI/models"

echo "── LTX-2.3 Model Downloader (16GB VRAM Budget) ──"
echo ""
echo "This script prints the download commands. Review and run them."
echo "You need 'huggingface-cli' or 'wget'."
echo ""

cat <<'INSTRUCTIONS'
═══════════════════════════════════════════════════════════════════════════════
1. TRANSFORMER — LTX-2.3-22B GGUF (Q3_K_M, ~10.1 GB)
   Target: models/diffusion_models/ltx-2.3/

   huggingface-cli download \
     Lightricks/LTX-Video-2.3-22B-GGUF \
     ltxv-22b-0.9.7-dev-Q3_K_M.gguf \
     --local-dir models/diffusion_models/ltx-2.3/

═══════════════════════════════════════════════════════════════════════════════
2. TEXT ENCODER — Gemma-3-12B-IT GGUF (Q4_K_M, ~7.4 GB)
   Target: models/text_encoders/ltx-2.3/

   ⚠  Will be offloaded to CPU RAM after encoding — does NOT compete for VRAM.

   huggingface-cli download \
     bartowski/google_gemma-3-12b-it-GGUF \
     google_gemma-3-12b-it-Q4_K_M.gguf \
     --local-dir models/text_encoders/ltx-2.3/

═══════════════════════════════════════════════════════════════════════════════
3. VAE — LTX-Video VAE (fp16, ~200 MB)
   Target: models/vae/ltx-2.3/

   huggingface-cli download \
     Lightricks/LTX-Video \
     ltx-video-2b-v0.9.1.safetensors \
     --local-dir models/vae/ltx-2.3/

   Or use the VAE bundled with the diffusion model checkpoint if available.

═══════════════════════════════════════════════════════════════════════════════
4. LoRA (OPTIONAL) — ID-LoRA for identity-preserving generation
   Target: models/loras/ltx-2.3/

   Download from the ID-LoRA-LTX2.3 release page and place .safetensors
   files into the loras/ltx-2.3/ directory.

═══════════════════════════════════════════════════════════════════════════════

VRAM BUDGET AT INFERENCE:
  Transformer (Q3_K_M):  ~10.1 GB
  LoRA patch overhead:    ~0.5 GB
  VAE decode (tiled):     ~2.5 GB
  Reserved (OS/driver):    2.5 GB
  ─────────────────────────────────
  Text Encoder:           → CPU offload (0 GB on GPU during decode)
  Total peak GPU:         ~15.6 GB / 16 GB  ✓

INSTRUCTIONS
