# ComfyUI Video & Image Generation Workspace

Local AI image and video generation on **chuckai** — a personal Ubuntu 22.04 server running
ComfyUI with quantized models on an NVIDIA RTX 3090 (24 GB VRAM).

---

## Active Workflows

| # | Workflow | Model | Input | Output | Status |
|---|---|---|---|---|---|
| 1 | **LTX-2.3 Distilled fp8 + custom LoRA** | LTX-2.3-22B Distilled fp8 + Gemma 3 12B | Text prompt | Video | ✅ Active |
| 2 | **LTX-2.3 GGUF + ID-LoRA** | LTX-2.3-22B GGUF Q3 + ID-LoRA | Text prompt + character ref | Video | ✅ Active |
| 3 | **Photo Restoration** | FLUX.1 Kontext fp8 + CodeFormer/GFPGAN + ESRGAN | Degraded photo | Enhanced photo | ✅ Active |
| 4 | **Photo Animation** | Wan 2.2 I2V A14B GGUF (MoE) | Portrait photo | Animated video | ✅ Validated 2026-05-05 |

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

- [Quick Start](#quick-start)
- [Workflow 1 — LTX-2.3 Distilled fp8](#workflow-1--ltx-23-distilled-fp8-text-to-video)
- [Workflow 2 — LTX-2.3 GGUF + ID-LoRA](#workflow-2--ltx-23-gguf--id-lora-text-to-video-with-identity)
- [Workflow 3 — Photo Restoration](#workflow-3--photo-restoration)
- [Workflow 4 — Photo Animation (Wan 2.2 I2V)](#workflow-4--photo-animation-wan-22-i2v)
- [Model Inventory](#model-inventory)
- [Custom Nodes](#custom-nodes)
- [VRAM Summary](#vram-summary)
- [Saved Workflows](#saved-workflows)
- [Useful Commands](#useful-commands)
- [Project History](#project-history)

---

## Quick Start

```bash
# Source environment (always first)
source ~/comfy/env.sh

# Launch ComfyUI — UI at http://localhost:8188
~/comfy/launch.sh
```

Load a workflow JSON from the UI, then queue.

---

## Workflow 1 — LTX-2.3 Distilled fp8 (Text-to-Video)

Generates video from text prompts using the full fp8 distilled checkpoint. Faster than the
GGUF path — 9 steps with the `res_multistep` sampler. A custom trained character LoRA is
applied at strength 1.1.

### Node Pipeline

| Stage | Node |
|---|---|
| Model load | `CheckpointLoaderSimple` — `ltx-2.3-22b-distilled-fp8.safetensors` |
| Text encoder | `LTXAVTextEncoderLoader` — `gemma_3_12B_it.safetensors` |
| LoRA | `LoraLoaderModelOnly` — `my_first_lora_v1_copy.safetensors` (strength 1.1) |
| Conditioning | `LTXVConditioning` |
| Sampler | `KSampler` — `res_multistep`, 9 steps |
| VAE decode | `VAEDecodeTiled` (tile_size=256, temporal_size=32) |
| Output | `SaveVideo` |

### Baseline Parameters

| Parameter | Value |
|---|---|
| Resolution | 736×480 |
| Frames | 97 |
| Steps | 9 (res_multistep) |

### Saved Workflows

- `ltx2.3_video-test.json` — base configuration
- `ltx2.3_video-testparamus.json` — 736×480, 97 frames, 9 steps

---

## Workflow 2 — LTX-2.3 GGUF + ID-LoRA (Text-to-Video with Identity)

Generates identity-consistent character video from text prompts. Uses GGUF-quantized models
(smaller memory footprint than the fp8 checkpoint path). Requires the ID-LoRA custom nodes.

**Critical:** LTX-2.3 requires its own node set — `CLIPLoaderGGUF` + `KSampler` will produce
a tensor dimension mismatch. Use `DualCLIPLoaderGGUF` + `SamplerCustomAdvanced`.

### Node Pipeline

| Stage | Node |
|---|---|
| Transformer | `UnetLoaderGGUF` — `ltx-2.3/ltx-2.3-22b-dev-Q3_K_M.gguf` |
| Text encoder | `DualCLIPLoaderGGUF` — Gemma GGUF + text projection, type: `ltxv` |
| Conditioning | `LTXVConditioning` |
| Scheduler | `LTXVScheduler` |
| Sampler | `SamplerCustomAdvanced` |
| VAE decode | `VAEDecodeTiled` (tile_size=256, temporal_size=32) |
| Character LoRA | `IDLoraTwoStageModelLoader` / `IDLoraTwoStageSampler` |

### Baseline Parameters

| Parameter | Value |
|---|---|
| Resolution | 512×512 (multiples of 32; safe: 512×512, 768×512, 1024×576) |
| Frames | 17 minimum (1 + 8N formula; valid: 17, 25, 33, 49, 65…) |
| Steps | 20 |
| CFG | 3.5 |
| Runtime | ~2 min on RTX 3090 |

### Hard Constraints

- Video dimensions must be multiples of 32 — non-aligned values cause memory faults
- VAE decode must use tiled mode: tile_size=256, temporal_size=32
- Gemma text encoder CPU-offloads automatically after encoding

### VRAM Budget

| Component | GPU Memory |
|---|---|
| Transformer (GGUF Q3_K_M) | ~11.0 GB |
| Text Projection / Connector | ~2.2 GB |
| VAE decode (tiled 256×32) | ~1.9 GB |
| **Peak** | **~13.2 GB** |
| Text Encoder (Gemma-3 Q4_K_M) | CPU offload |

---

## Workflow 3 — Photo Restoration

Multi-stage pipeline for restoring degraded, aged, or low-quality photographs. Multiple
workflow variants exist for different subjects and restoration levels.

**Pass 1 — FLUX.1 Kontext (fp8 scaled):** Structural restoration and overall image enhancement.

**Pass 2 — Face restore (optional):** CodeFormer or GFPGAN for face-specific sharpening.

**Pass 3 — Upscale (optional):** ESRGAN / RealESRGAN for final resolution increase.

**Alternative:** SUPIR (`SUPIR-v0F` / `SUPIR-v0Q`) for heavy restoration + upscaling in one pass.

### FLUX.1 Kontext Required Models

| Model | File |
|---|---|
| FLUX UNet | `flux1-dev-kontext_fp8_scaled.safetensors` (12 GB) |
| Text encoder (T5) | `t5xxl_fp16.safetensors` (9.2 GB) |
| Text encoder (CLIP) | `clip_l.safetensors` (235 MB) |
| VAE | `ae.safetensors` (320 MB) |
| Face restore | `codeformer.pth` or `GFPGANv1.4.pth` |
| Upscaler | `4x-foolhardy-Remacri.pth` or `RealESRGAN_x4plus.pth` |

### Saved Workflows

- `flux-kontext-restoration.json` — FLUX Kontext pass only (no CodeFormer/upscale)
- `flux-kontext-restoration-fast.json` — full pipeline (FLUX + CodeFormer + upscale)
- `flux-kontext-restoration-fast-face.json`, `-facev1.json`, `-grandma.json`, `-papado.json`, `-papado2.json`, `-nocolor.json` — subject-specific variants

---

## Workflow 4 — Photo Animation (Wan 2.2 I2V)

Animates still portrait photographs. Selected for portrait work because I2V models
conditioned on an input frame preserve face identity significantly better than T2V approaches.

### Validated Result — 2026-05-05

- 832×480, 81 frames, 20 steps, CFG 3.5, euler/simple
- Wall time: **17 min 11 sec** (~50 s/step × 10 steps per expert × 2 experts)
- Peak VRAM: **15.1 GB** (61% of 24 GB) — 9 GB headroom, no OOM

### Architecture

Wan 2.2 14B is a **Mixture-of-Experts (MoE)** model — separate high-noise and low-noise
expert transformers run sequentially. ComfyUI offloads the inactive expert during the switch.
Both transformer files are required.

### Sampler Configuration

| Mode | Steps | CFG | Split at | Time (measured) | Use for |
|---|---|---|---|---|---|
| Original (Lightning OFF) | 20 | 3.5 | step 10 | **17 min 11 sec** | Finals |
| Turbo (Lightning ON) | 4 | 1.0 | step 2 | ~2–4 min | Scouting |

Toggle "Enable 4steps LoRA?" in the subgraph to switch modes.

### VRAM Budget (measured)

| Component | GPU Memory |
|---|---|
| Transformer (GGUF Q5_K_M, one expert) | 10.4 GB |
| UMT5-XXL text encoder (fp8, CPU-offloaded) | 6.4 GB |
| VAE + latents + KV cache (81 frames @ 832×480) | ~4.7 GB |
| **Peak** | **15.1 GB (61% of 24 GB)** |

### Saved Workflows

- `wan22_i2v_portrait_animation.json` — main workflow
- `wan22_i2v_portrait_animation-walk towards.json` — motion variant

---

## Model Inventory

### Checkpoints

| File | Size | Used by |
|---|---|---|
| `ltx-2.3-22b-distilled-fp8.safetensors` | 28 GB | LTX Distilled fp8 workflow (active) |
| `ltx-2.3-22b-dev-fp8.safetensors` | 28 GB | LTX Dev fp8 (reference/alternate) |
| `juggernautXL_v9Rdphoto2Lightning.safetensors` | 6.7 GB | JuggernautXL SDXL — photo realism |
| `SUPIR-v0F.ckpt` | 5.0 GB | SUPIR face-focused restoration |
| `SUPIR-v0Q.ckpt` | 5.0 GB | SUPIR quality-focused restoration |

### Diffusion Models / Transformers

| File | Size | Used by |
|---|---|---|
| `ltx-2.3/ltx-2.3-22b-dev-Q3_K_M.gguf` | 11 GB | LTX GGUF workflow |
| `diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors` | 23 GB | LTX Distilled transformer only (double-nested path) |
| `flux1-dev-kontext_fp8_scaled.safetensors` | 12 GB | Photo Restoration Pass 1 |
| `wan2.2_i2v_high_noise_14B_Q5_K_M.gguf` | 11 GB | Wan 2.2 MoE high-noise expert |
| `wan2.2_i2v_low_noise_14B_Q5_K_M.gguf` | 11 GB | Wan 2.2 MoE low-noise expert |
| `wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf` | — | → symlink |
| `wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf` | — | → symlink |

### Text Encoders

| File | Size | Used by |
|---|---|---|
| `gemma_3_12B_it.safetensors` | 8.8 GB | LTX Distilled fp8 workflow |
| `ltx-2.3/google_gemma-3-12b-it-Q4_K_M.gguf` | 6.8 GB | LTX GGUF workflow |
| `ltx-2.3/ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | LTX GGUF workflow (embeddings connector) |
| `t5xxl_fp16.safetensors` | 9.2 GB | FLUX photo restoration |
| `clip_l.safetensors` | 235 MB | FLUX photo restoration |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.3 GB | Wan 2.2 |

### VAE

| File | Size | Used by |
|---|---|---|
| `ae.safetensors` | 320 MB | FLUX photo restoration |
| `ltx-2.3/LTX23_video_vae_bf16.safetensors` | 1.4 GB | LTX GGUF workflow |
| `wan_2.1_vae.safetensors` | 243 MB | Wan 2.2 |

### LoRAs

| File | Size | Notes |
|---|---|---|
| `my_first_lora_v1_copy.safetensors` | 644 MB | Custom trained — active in LTX Distilled workflow |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.1 GB | Official LTX Distilled LoRA v1.1 |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | 1.2 GB | Wan 2.2 Lightning LoRA |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | 1.2 GB | Wan 2.2 Lightning LoRA |
| `Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors` | — | → symlink |
| `Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors` | — | → symlink |

### Upscale / Face Restoration

| File | Size | Type |
|---|---|---|
| `latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950 MB | LTX-2.3 latent spatial 2× upscaler |
| `upscale_models/4x-foolhardy-Remacri.pth` | 64 MB | ESRGAN 4× — photorealism tuned |
| `upscale_models/RealESRGAN_x4plus.pth` | 64 MB | RealESRGAN 4× — general |
| `facerestore_models/codeformer.pth` | 360 MB | CodeFormer face restoration |
| `facerestore_models/GFPGANv1.4.pth` | 333 MB | GFPGAN v1.4 face restoration |
| `facedetection/detection_Resnet50_Final.pth` | 105 MB | Face detection (used by restore nodes) |
| `facedetection/parsing_parsenet.pth` | 82 MB | Face parsing (used by restore nodes) |

---

## Custom Nodes

| Node | Commit | Purpose |
|---|---|---|
| ComfyUI-GGUF (city96) | `6ea2651` (2026-01-12) | GGUF model loading — LTX and Wan |
| ComfyUI-Manager (ltdrdata) | `8d5c1203` (2026-05-01) | Node management |
| ComfyUI-WanVideoWrapper (kijai) | `df8f3e4` (2026-02-22) | Wan 2.2 I2V support |
| ComfyUI-KJNodes (kijai) | `cd5ad80` (2026-05-03) | Utility nodes for Wan pipeline |
| ComfyUI-VideoHelperSuite (Kosinkadink) | `2984ec4` (2026-04-06) | MP4 output |
| ID-LoRA-LTX2.3-ComfyUI | `9943746` (2026-03-25) | Identity-preserving LoRA for LTX-2.3 |
| facerestore_cf | `ff4d7a5` | CodeFormer face restoration nodes |
| comfyui_gfpgan | `77577e4` | GFPGAN face restoration nodes |
| ComfyUI-SUPIR | `99d49e9` | SUPIR heavy upscaling/restoration |
| ComfyUI_UltimateSDUpscale | `bebd569` | Ultimate SD Upscale tiled upscaling |

---

## VRAM Summary

| Workflow | Peak VRAM | Headroom (24 GB) |
|---|---|---|
| LTX-2.3 GGUF + ID-LoRA | ~13.2 GB | ~10.8 GB |
| Photo Restoration (FLUX Kontext fp8) | ~10–16 GB | ~8–14 GB |
| Photo Animation (Wan 2.2 I2V A14B) | **15.1 GB** (measured) | ~9 GB |

No `--lowvram` flag required for any workflow on the RTX 3090.

---

## Saved Workflows

All in `~/comfy/ComfyUI/user/default/workflows/`:

| File | Pipeline |
|---|---|
| `ltx2.3_video-test.json` | LTX-2.3 Distilled fp8 + custom LoRA |
| `ltx2.3_video-testparamus.json` | LTX-2.3 Distilled fp8 — 736×480, 97 frames, 9 steps |
| `video_ltx2_3_t2v-1.json` | LTX-2.3 subgraph wrapper |
| `video_ltx2_3_t2v-2.json` | LTX-2.3 subgraph wrapper |
| `flux-kontext-restoration.json` | Photo Restoration — FLUX only |
| `flux-kontext-restoration-fast.json` | Photo Restoration — full pipeline |
| `flux-kontext-restoration-fast-face.json` | Photo Restoration — subject variant |
| `flux-kontext-restoration-fast-facev1.json` | Photo Restoration — subject variant |
| `flux-kontext-restoration-fast-grandma.json` | Photo Restoration — subject variant |
| `flux-kontext-restoration-fast-nocolor.json` | Photo Restoration — no-color variant |
| `flux-kontext-restoration-fast-papado.json` | Photo Restoration — subject variant |
| `flux-kontext-restoration-fast-papado2.json` | Photo Restoration — subject variant |
| `wan22_i2v_portrait_animation.json` | Photo Animation — main |
| `wan22_i2v_portrait_animation-walk towards.json` | Photo Animation — motion variant |

---

## Useful Commands

```bash
# Source environment (always first)
source ~/comfy/env.sh

# Launch ComfyUI
~/comfy/launch.sh

# GPU status / live VRAM during generation
nvidia-smi
watch -n 1 nvidia-smi

# Confirm PyTorch sees the 3090
source ~/comfy/env.sh && python -c \
  "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"

# Kill hanging ComfyUI
pkill -f "python.*main.py"

# Check port 8188
ss -tlnp 2>/dev/null | grep 8188

# Check custom node import errors
grep -E "(ERROR|IMPORT|Traceback)" ~/comfy/comfyui.log 2>/dev/null | tail -20

# Confirm launch.sh flags
grep -E "reserve-vram|lowvram|attention" ~/comfy/launch.sh
```

---

## Project History

| Phase | GPU | Stack | Result |
|---|---|---|---|
| 1 | AMD RX 6800 XT (16 GB) | ROCm 6.3 / torch 2.9.1+rocm6.3 | LTX-2.3 22B GGUF text-to-video validated |
| 2 | NVIDIA RTX 3090 (24 GB) | CUDA 12.4 / torch 2.6.0+cu124 | Full CUDA rebuild; Wan 2.2 I2V + Photo Restoration added |
| 3 | NVIDIA RTX 3090 (24 GB) | CUDA 12.4 / torch 2.6.0+cu124 | LTX Distilled fp8 + custom LoRA active; 4 workflows operational |

ROCm environment preserved in `env.sh.rocm-backup` and `launch.sh.rocm-backup` for reference.
