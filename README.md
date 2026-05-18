# ComfyUI Video & Image Generation Workspace

Local AI image and video generation on **chuckai** — a personal Ubuntu 22.04 server running
ComfyUI with GGUF-quantized models on an NVIDIA RTX 3090 (24 GB VRAM).

---

## Active Workflows

Three production pipelines are currently running:

| # | Workflow | Model | Input | Output | Status |
|---|---|---|---|---|---|
| 1 | **LTX-2.3 + Character LoRA** | LTX-2.3-22B GGUF Q3_K_M + ID-LoRA | Text prompt + character reference | Video | ✅ Active |
| 2 | **Photo Restoration** | FLUX.1 Kontext fp8 (Pass 1) + CodeFormer (Pass 2) | Degraded photo | Enhanced photo | ✅ Active |
| 3 | **Photo Animation** | Wan 2.2 I2V A14B GGUF | Portrait photo | Animated video | ✅ Validated 2026-05-05 |

---

## Hardware & Stack

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 — 24 GiB GDDR6X |
| CUDA | 12.4 |
| cuDNN | 9.1.0 |
| PyTorch | 2.6.0+cu124 |
| triton | 3.2.0 |
| Python | 3.10.12 |
| ComfyUI | v0.20.1 (commit `2806163f`, 2026-05-04) |
| OS | Ubuntu 22.04.5 LTS |

---

## Table of Contents

- [Workflow 1 — LTX-2.3 + Character LoRA](#workflow-1--ltx-23--character-lora-text-to-video)
- [Workflow 2 — Photo Restoration](#workflow-2--photo-restoration)
- [Workflow 3 — Photo Animation (Wan 2.2 I2V)](#workflow-3--photo-animation-wan-22-i2v)
- [Quick Start](#quick-start)
- [Custom Nodes](#custom-nodes)
- [VRAM Summary](#vram-summary)
- [Useful Commands](#useful-commands)
- [Project History](#project-history)

---

## Workflow 1 — LTX-2.3 + Character LoRA (Text-to-Video)

Generates videos from text prompts with identity-consistent character rendering using
the LTX-2.3 22B video model and ID-LoRA identity-preserving nodes.

### Key Architecture Points

- LTX-2.3 is an audio-video model. The standard `CLIPLoaderGGUF` + `KSampler` path does
  NOT work — it produces a tensor dimension mismatch. Use LTX-specific nodes exclusively.
- The text projection file (`ltx-2.3_text_projection_bf16.safetensors`) is required alongside
  the Gemma text encoder. Without it the embedding shape is wrong.

### Node Pipeline

| Stage | Node |
|---|---|
| Transformer | `UnetLoaderGGUF` |
| Text encoder + connector | `DualCLIPLoaderGGUF` (Gemma GGUF + text projection, type: `ltxv`) |
| Conditioning | `LTXVConditioning` |
| Scheduler | `LTXVScheduler` |
| Sampler | `SamplerCustomAdvanced` |
| VAE decode | `VAEDecodeTiled` (tile_size=256, temporal_size=32) |
| Character LoRA | `IDLoraTwoStageModelLoader` / `IDLoraTwoStageSampler` |

### Model Files

| File | Size | Source |
|---|---|---|
| `ltx-2.3-22b-dev-Q3_K_M.gguf` | 11 GB | `unsloth/LTX-2.3-GGUF` |
| `google_gemma-3-12b-it-Q4_K_M.gguf` | 6.8 GB | `Kijai/LTX2.3_comfy` |
| `ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | `Kijai/LTX2.3_comfy` |
| `LTX23_video_vae_bf16.safetensors` | 1.4 GB | `Kijai/LTX2.3_comfy` |
| ID-LoRA weights | ~1.1 GB | `Lightricks/LTX-Video-2.3` |

Do NOT download `Lightricks/LTX-Video-2.3-22B-GGUF` — that repo does not exist.
Do NOT download the full fp16 checkpoint (`ltx-2.3-22b-dev.safetensors`) — it is 43 GB.

### Baseline Parameters

| Parameter | Value |
|---|---|
| Resolution | 512×512 (must be multiples of 32; safe presets: 512×512, 768×512, 1024×576) |
| Frames | 17 minimum (formula: 1 + 8N; valid: 17, 25, 33, 49, 65…) |
| Steps | 20 |
| CFG | 3.5 |
| Runtime | ~2 min on the RTX 3090 |

### VRAM Budget

| Component | GPU Memory |
|---|---|
| Transformer (GGUF Q3_K_M) | ~11.0 GB |
| Text Projection / Embeddings Connector | ~2.2 GB |
| VAE decode (tiled 256×32) | ~1.9 GB |
| **Peak** | **~13.2 GB** |
| Text Encoder (Gemma-3 Q4_K_M) | CPU offload |

---

## Workflow 2 — Photo Restoration

Two-pass pipeline for restoring degraded, aged, or low-quality photographs. Operates
entirely on still images — no video generation involved.

**Pass 1 — FLUX.1 Kontext (fp8 scaled):** Structural restoration and overall image
enhancement via the FLUX.1 Kontext editing model.

**Pass 2 — CodeFormer:** Face-specific reconstruction to sharpen facial features after
the FLUX pass.

### Node Pipeline

| Stage | Node | Model |
|---|---|---|
| Load | `LoadImage` | Input photo |
| Pass 1 — FLUX restore | FLUX UNet loader + sampler | `flux1-dev-kontext_fp8_scaled.safetensors` |
| Pass 2 — Face restore | CodeFormer node | `codeformer.pth` |
| Save | `SaveImage` | Output |

FLUX.1 Kontext also requires T5-XXL + CLIP-L text encoders and the FLUX VAE.

### Model Files

| File | Size | Directory |
|---|---|---|
| `flux1-dev-kontext_fp8_scaled.safetensors` | ~12 GB | `models/diffusion_models/` or `models/unet/` |
| `codeformer.pth` | ~340 MB | `models/facerestore_models/` |

### VRAM Budget

FLUX.1 Kontext fp8 scaled peaks at ~10–16 GB on the RTX 3090 depending on resolution.
CodeFormer adds ~340 MB. Well within the 24 GB envelope.

---

## Workflow 3 — Photo Animation (Wan 2.2 I2V)

Animates still portrait photographs using Wan 2.2 image-to-video inference. Selected over
LTX-2.3 for this use case because I2V models conditioned on an input frame preserve face
identity significantly better than text-to-video approaches.

### Validated Result — 2026-05-05

- 832×480, 81 frames, 20 steps, CFG 3.5, euler/simple
- Wall time: **17 min 11 sec** (~50 s/step × 10 steps per expert × 2 experts)
- Peak VRAM: **15.1 GB** (61% of 24 GB) — 9 GB headroom, no OOM
- MoE switch confirmed: high-noise expert (steps 0–10) → offload → low-noise (steps 10–20)

### Model Files — Actual Layout

The downloader used underscore-style filenames. Symlinks with the hyphen-style names
expected by the workflow JSON were created in-place:

| Actual file on disk | Size | Symlink (used by workflow) |
|---|---|---|
| `wan2.2_i2v_high_noise_14B_Q5_K_M.gguf` | 11 GB | `wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf` |
| `wan2.2_i2v_low_noise_14B_Q5_K_M.gguf` | 11 GB | `wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.3 GB | (no rename needed) |
| `wan_2.1_vae.safetensors` | 243 MB | (no rename needed) |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | 1.2 GB | `Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors` |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | 1.2 GB | `Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors` |

### Workflow JSON

`wan22_i2v_portrait_animation.json` — based on `Comfy-Org/workflow_templates video_wan2_2_14B_i2v.json`
with these modifications:

| Change | Why |
|---|---|
| `UNETLoader` → `UnetLoaderGGUF` on both transformer nodes | Our transformers are GGUF format |
| Filenames pre-set to actual downloaded files | No dropdown confusion on first load |
| Default resolution 640×640 → 832×480 | Better aspect ratio for portraits |
| Frames: 81 (5 seconds @ 16 fps) | Baseline target duration |
| `Enable 4steps LoRA?` default: OFF | Validate 20-step baseline before Lightning |
| LoadImage filename cleared | Original had a demo filename that would error immediately |
| Negative prompt rewritten in English | Original used Chinese; new version suppresses talking, lip sync, identity drift, face morphing |

**Loader ecosystem:** ComfyUI has two parallel loader paths for Wan. Use native ComfyUI
loaders (not WanVideoWrapper loaders) — the WanVideoWrapper rejects the fp8-scaled text
encoder with `ValueError("fp8 scaled is not supported by this node")`.

### Sampler Configuration

| Mode | Steps | CFG | Split at | Time (3090, measured) | Use for |
|---|---|---|---|---|---|
| Original (Lightning OFF) | 20 | 3.5 | step 10 | **17 min 11 sec** | Finals |
| Turbo (Lightning ON) | 4 | 1.0 | step 2 | ~2-4 min (untested) | Scouting |

### VRAM Budget (measured)

| Component | GPU Memory |
|---|---|
| Transformer (GGUF Q5_K_M, one expert at a time) | 10.4 GB |
| UMT5-XXL text encoder (fp8 scaled, CPU-offloaded after encoding) | 6.4 GB |
| VAE + latents + KV cache (81 frames @ 832×480) | ~4.7 GB |
| **Actual peak** | **15.1 GB (61% of 24 GB)** |

No `--lowvram` needed. `--reserve-vram 1.0` is set in `launch.sh`.

### Common Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `Unknown node type: UnetLoaderGGUF` | ComfyUI-GGUF didn't import | Check `custom_nodes/ComfyUI-GGUF/` exists; restart |
| Red filename in loader dropdown | Symlink missing or broken | Check symlinks in `diffusion_models/` and `loras/` |
| OOM at sampling | Frames or resolution too large | Drop to 49 frames or 480×480 |
| `Tensor size mismatch` | LoRA version mismatch | Confirm both LoRAs are I2V not T2V |
| Pure noise output | ModelSamplingSD3 shift wrong | Both `ModelSamplingSD3` nodes should be 5.0 |
| Identity drifts at ~frame 40 | Temporal coherence limit | Drop to 49 frames; or lower CFG to 3.0 |

---

## Quick Start

```bash
# Source environment (always first)
source ~/comfy/env.sh

# Launch ComfyUI — UI at http://192.168.1.59:8188
~/comfy/launch.sh

# Workflow 3 — Photo Animation (Wan 2.2)
# Load: wan22_i2v_portrait_animation.json
# Upload portrait → Queue
# Lightning OFF = 17 min (finals) | Lightning ON = ~2-4 min (scouting)

# Workflow 2 — Photo Restoration
# Load the restoration workflow JSON, upload degraded photo, queue

# Workflow 1 — LTX-2.3 + Character LoRA
# Load the LTX workflow JSON, set text prompt + character LoRA reference, queue
```

---

## Custom Nodes

| Node | Commit | Purpose |
|---|---|---|
| ComfyUI-GGUF (city96) | `6ea2651` (2026-01-12) | GGUF loading for LTX-2.3 and Wan 2.2 |
| ComfyUI-Manager (ltdrdata) | `8d5c1203` (2026-05-01) | Node management |
| ComfyUI-WanVideoWrapper (kijai) | `df8f3e4` (2026-02-22) | Wan 2.2 I2V support |
| ComfyUI-KJNodes (kijai) | `cd5ad80` (2026-05-03) | Utility nodes for Wan pipeline |
| ComfyUI-VideoHelperSuite (Kosinkadink) | `2984ec4` (2026-04-06) | MP4 output |
| ID-LoRA-LTX2.3-ComfyUI | `9943746` (2026-03-25) | Identity-preserving LoRA for LTX-2.3 |

**Cosmetic startup warning** — `ComfyUI-GGUF: Partial torch compile only, consider updating pytorch` is informational only. Do not upgrade PyTorch.

---

## VRAM Summary

| Workflow | Peak VRAM | Headroom (24 GB) |
|---|---|---|
| LTX-2.3 + Character LoRA | ~13.2 GB | ~10.8 GB |
| Photo Restoration (FLUX Kontext + CodeFormer) | ~10-16 GB | ~8-14 GB |
| Photo Animation (Wan 2.2 I2V A14B) | **15.1 GB** (measured) | ~9 GB |

All three workflows fit comfortably in 24 GB. No `--lowvram` flag required.

---

## Useful Commands

```bash
# Source environment (always do this first)
source ~/comfy/env.sh

# Launch ComfyUI
~/comfy/launch.sh

# GPU status
nvidia-smi
watch -n 1 nvidia-smi   # live during generation

# Confirm PyTorch sees the 3090
source ~/comfy/env.sh && python -c \
  "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"

# Verify Wan 2.2 symlinks + models
ls -lah ~/comfy/ComfyUI/models/diffusion_models/wan2.2* \
        ~/comfy/ComfyUI/models/text_encoders/umt5* \
        ~/comfy/ComfyUI/models/vae/wan_2.1* \
        ~/comfy/ComfyUI/models/loras/wan2.2* \
        ~/comfy/ComfyUI/models/loras/Wan2.2* 2>/dev/null

# Verify LTX-2.3 models
find ~/comfy/ComfyUI/models/diffusion_models/ltx-2.3 \
     ~/comfy/ComfyUI/models/text_encoders/ltx-2.3 \
     ~/comfy/ComfyUI/models/vae/ltx-2.3 \
     -type f -ls 2>/dev/null

# Verify photo restoration models
ls -lah ~/comfy/ComfyUI/models/diffusion_models/flux1-dev-kontext* 2>/dev/null
ls -lah ~/comfy/ComfyUI/models/unet/flux1-dev-kontext* 2>/dev/null
ls -lah ~/comfy/ComfyUI/models/facerestore_models/codeformer.pth 2>/dev/null

# Kill hanging ComfyUI process
pkill -f "python.*main.py"

# Check what's on port 8188
ss -tlnp 2>/dev/null | grep 8188

# Confirm launch.sh uses --reserve-vram (not --lowvram)
grep -E "reserve-vram|lowvram" ~/comfy/launch.sh

# Check custom node import errors
grep -E "(ERROR|IMPORT|Traceback)" ~/comfy/comfyui.log 2>/dev/null | tail -20
```

---

## Project History

| Phase | GPU | Stack | Pipelines | Status |
|---|---|---|---|---|
| 1 | AMD RX 6800 XT (16 GB) | ROCm 6.3 / `torch 2.9.1+rocm6.3` | LTX-2.3 22B text-to-video | ✅ Validated end-to-end |
| 2 | NVIDIA RTX 3090 (24 GB) | CUDA 12.4 / `torch 2.6.0+cu124` | LTX-2.3 carried over; CUDA rebuild | ✅ Stack rebuilt and verified |
| 3 | NVIDIA RTX 3090 (24 GB) | CUDA 12.4 / `torch 2.6.0+cu124` | Wan 2.2 I2V validated; Photo Restoration added; LTX + Character LoRA active | ✅ Three workflows operational |

The RX 6800 XT was physically present alongside the RTX 3090. The ROCm venv
(`torch 2.9.1+rocm6.3`) had no awareness of the NVIDIA card — `torch.cuda.is_available()`
returned False on the 3090. The rebuild in May 2026 corrected this with a fresh CUDA venv.

### ROCm-Era Reference — RX 6800 XT (16 GB)

Preserved for reference. The CUDA build supersedes everything here.

**Phase 1 VRAM budget (16 GB hard limit):**

| Component | GPU Memory |
|---|---|
| LTX-2.3-22B Transformer (GGUF Q3_K_M) | ~11.0 GB |
| Text Projection / Embeddings Connector | ~2.2 GB |
| VAE decode (tiled 256×32) | ~1.9 GB |
| **Peak** | **~13.2 GB (82% of 16 GB)** |
| Text Encoder (Gemma-3 Q4_K_M) | CPU offload |
| Transformer during VAE decode | CPU offload via `--lowvram` |

`--lowvram` was the critical flag on 16 GB — removed in Phase 2 (24 GB headroom makes it
unnecessary).

Full ROCm environment preserved in `~/comfy/env.sh.rocm-backup` and
`~/comfy/launch.sh.rocm-backup` for reference.
