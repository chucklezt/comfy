# CLAUDE.md — ComfyUI Workspace (chuckai)

## Context

You are working on **chuckai** — a local Ubuntu 22.04 server with an **NVIDIA GeForce RTX 3090**
(24 GB VRAM) running CUDA 12.4. The workspace is at `~/comfy/`.

The GPU was migrated from an AMD RX 6800 XT / ROCm 6.3 setup to a full CUDA rebuild
(completed 2026-05-04). ROCm is no longer in use for this workspace.

Four production workflows are currently active:

1. **LTX-2.3 Distilled fp8 + custom LoRA** (text-to-video, audio-video capable) — uses the
   full fp8 distilled checkpoint loaded via `CheckpointLoaderSimple` + `LTXAVTextEncoderLoader`.
   Faster than the GGUF path (9 steps with res_multistep). Has a custom trained LoRA applied.

2. **LTX-2.3 GGUF Q3 + ID-LoRA** (text-to-video with identity) — uses the GGUF-quantized
   transformer and GGUF text encoder. Requires ID-LoRA custom nodes for identity-consistent
   character rendering.

3. **Photo Restoration** — multi-stage pipeline: FLUX.1 Kontext fp8 (structural restore) +
   CodeFormer or GFPGAN (face reconstruction). SUPIR is also available for heavy upscaling.
   Operates on still images. Multiple saved subject-specific workflow variants exist.

4. **Photo Animation — Wan 2.2 I2V 14B GGUF** (image-to-video) — animates still portrait
   photos using a MoE two-expert architecture. First inference validated 2026-05-05.

**Current focus: Lightning mode test + sageattention install.**

---

## Hardware & Stack Reference

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| VRAM | 24 GiB (23.6 GiB usable) |
| CUDA | 12.4 |
| cuDNN | 9.1.0 |
| PyTorch | 2.6.0+cu124 |
| triton | 3.2.0 |
| Python venv | `~/comfy/venv/` (Python 3.10.12) |
| ComfyUI | v0.20.1 (commit `2806163f`, 2026-05-04, at HEAD) |
| ComfyUI root | `~/comfy/ComfyUI/` |
| Custom nodes | `~/comfy/ComfyUI/custom_nodes/` |
| Model staging | `~/comfy/ComfyUI/models/` |
| env script | `~/comfy/env.sh` |
| Launch script | `~/comfy/launch.sh` |

**Always source the environment before running any Python or ComfyUI commands:**
```bash
source ~/comfy/env.sh
```

---

## Custom Nodes

| Node | Commit | Status / Purpose |
|---|---|---|
| ComfyUI-GGUF (city96) | `6ea2651` (2026-01-12) | OK — GGUF loading for LTX-2.3 and Wan 2.2 |
| ComfyUI-Manager (ltdrdata) | `8d5c1203` (2026-05-01) | OK — node management |
| ComfyUI-WanVideoWrapper (kijai) | `df8f3e4` (2026-02-22) | OK — Wan 2.2 I2V support |
| ComfyUI-KJNodes (kijai) | `cd5ad80` (2026-05-03) | OK — utility nodes for Wan pipeline |
| ComfyUI-VideoHelperSuite (Kosinkadink) | `2984ec4` (2026-04-06) | OK — MP4 output |
| ID-LoRA-LTX2.3-ComfyUI | `9943746` (2026-03-25) | Active — identity LoRA for LTX-2.3 GGUF workflow |
| facerestore_cf | `ff4d7a5` | OK — CodeFormer face restoration nodes |
| comfyui_gfpgan | `77577e4` | OK — GFPGAN face restoration nodes |
| ComfyUI-SUPIR | `99d49e9` | OK — SUPIR heavy upscaling/restoration |
| ComfyUI_UltimateSDUpscale | `bebd569` | OK — Ultimate SD Upscale tiled upscaling |

**Known cosmetic notice at startup:**
`ComfyUI-GGUF: Partial torch compile only, consider updating pytorch` — informational only.
GGUF loading and inference work fine on torch 2.6. Do not bump torch to 2.7+ without
validating wheel compatibility (sageattention, flash-attn).

**If ID-LoRA-LTX2.3-ComfyUI shows import failure** (`ltx_core` not installed):
```bash
source ~/comfy/env.sh
pip install ltx-core ltx-pipelines
pip install -e ~/ID-LoRA/ID-LoRA-2.3/packages/ltx-trainer
pip uninstall bitsandbytes -y   # ltx-trainer drags this in — remove immediately
```

---

## Model Inventory

### Checkpoints (`models/checkpoints/`)

| File | Size | Purpose |
|---|---|---|
| `ltx-2.3-22b-distilled-fp8.safetensors` | 28 GB | LTX-2.3 Distilled — full fp8, active production model |
| `ltx-2.3-22b-dev-fp8.safetensors` | 28 GB | LTX-2.3 Dev — full fp8, reference/alternate |
| `juggernautXL_v9Rdphoto2Lightning.safetensors` | 6.7 GB | JuggernautXL SDXL checkpoint — photo-realism |
| `SUPIR-v0F.ckpt` | 5.0 GB | SUPIR face-focused restoration model |
| `SUPIR-v0Q.ckpt` | 5.0 GB | SUPIR quality-focused restoration model |

### Diffusion Models / Transformers (`models/diffusion_models/`)

| File | Size | Purpose |
|---|---|---|
| `ltx-2.3/ltx-2.3-22b-dev-Q3_K_M.gguf` | 11 GB | LTX-2.3 Dev transformer — GGUF Q3_K_M (GGUF workflow) |
| `diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors` | 23 GB | LTX-2.3 Distilled transformer only — mxfp8 block32 (**note: double-nested path**) |
| `flux1-dev-kontext_fp8_scaled.safetensors` | 12 GB | FLUX.1 Kontext — photo restoration (Pass 1) |
| `wan2.2_i2v_high_noise_14B_Q5_K_M.gguf` | 11 GB | Wan 2.2 high-noise expert (actual file) |
| `wan2.2_i2v_low_noise_14B_Q5_K_M.gguf` | 11 GB | Wan 2.2 low-noise expert (actual file) |
| `wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf` | — | → symlink to above |
| `wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf` | — | → symlink to above |

**Note on the double-nested path:** The LTX distilled transformer-only file was downloaded into
`diffusion_models/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors`
— an extra `diffusion_models/` subdirectory inside the `diffusion_models/` model directory.
The active production workflows use `ltx-2.3-22b-distilled-fp8.safetensors` (the full checkpoint
in `checkpoints/`) which does not have this path issue.

### Text Encoders (`models/text_encoders/`)

| File | Size | Purpose |
|---|---|---|
| `gemma_3_12B_it.safetensors` | 8.8 GB | Gemma 3 12B — full precision, used by LTX Distilled fp8 workflow |
| `ltx-2.3/google_gemma-3-12b-it-Q4_K_M.gguf` | 6.8 GB | Gemma 3 12B — GGUF Q4, used by LTX GGUF workflow |
| `ltx-2.3/ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | LTX-2.3 embeddings connector — required for GGUF workflow |
| `t5xxl_fp16.safetensors` | 9.2 GB | T5-XXL fp16 — FLUX text encoder |
| `clip_l.safetensors` | 235 MB | CLIP-L — FLUX text encoder (paired with T5-XXL) |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.3 GB | UMT5-XXL fp8 — Wan 2.2 text encoder |

### VAE (`models/vae/`)

| File | Size | Purpose |
|---|---|---|
| `ae.safetensors` | 320 MB | FLUX VAE — required for photo restoration workflow |
| `ltx-2.3/LTX23_video_vae_bf16.safetensors` | 1.4 GB | LTX-2.3 VAE — used by GGUF workflow |
| `wan_2.1_vae.safetensors` | 243 MB | Wan 2.2 VAE |

**Note:** The LTX Distilled fp8 workflow uses the VAE embedded in the full checkpoint, not
`LTX23_video_vae_bf16.safetensors` directly.

### LoRAs (`models/loras/`)

| File | Size | Purpose |
|---|---|---|
| `my_first_lora_v1_copy.safetensors` | 644 MB | Custom trained LoRA — applied in LTX Distilled workflow |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.1 GB | LTX-2.3 Distilled official LoRA v1.1 |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | 1.2 GB | Wan 2.2 Lightning high-noise LoRA (actual file) |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | 1.2 GB | Wan 2.2 Lightning low-noise LoRA (actual file) |
| `Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors` | — | → symlink to above |
| `Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors` | — | → symlink to above |

### Latent Upscale Models (`models/latent_upscale_models/`)

| File | Size | Purpose |
|---|---|---|
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 950 MB | LTX-2.3 spatial 2× latent upscaler |

### Upscale Models (`models/upscale_models/`)

| File | Size | Purpose |
|---|---|---|
| `4x-foolhardy-Remacri.pth` | 64 MB | ESRGAN 4× — optimized for photorealism |
| `RealESRGAN_x4plus.pth` | 64 MB | RealESRGAN 4× — general purpose |

### Face Restoration / Detection (`models/facerestore_models/`, `models/facedetection/`)

| File | Size | Purpose |
|---|---|---|
| `facerestore_models/codeformer.pth` | 360 MB | CodeFormer — face restoration |
| `facerestore_models/GFPGANv1.4.pth` | 333 MB | GFPGAN v1.4 — face restoration |
| `facedetection/detection_Resnet50_Final.pth` | 105 MB | Face detection (required by restoration nodes) |
| `facedetection/parsing_parsenet.pth` | 82 MB | Face parsing (required by restoration nodes) |

---

## Saved Workflows

Located in `~/comfy/ComfyUI/user/default/workflows/`:

| File | Pipeline | Notes |
|---|---|---|
| `ltx2.3_video-test.json` | LTX-2.3 Distilled fp8 | `ltx-2.3-22b-distilled-fp8.safetensors` + `gemma_3_12B_it.safetensors` + `my_first_lora_v1_copy.safetensors` |
| `ltx2.3_video-testparamus.json` | LTX-2.3 Distilled fp8 | Same as above, 9 steps res_multistep, 736×480, 97 frames |
| `video_ltx2_3_t2v-1.json` | LTX-2.3 (subgraph) | Group-node based wrapper workflow |
| `video_ltx2_3_t2v-2.json` | LTX-2.3 (subgraph) | Group-node based wrapper workflow |
| `flux-kontext-restoration.json` | Photo Restoration (minimal) | FLUX Kontext only, no CodeFormer/upscale pass |
| `flux-kontext-restoration-fast.json` | Photo Restoration (full) | FLUX Kontext + CodeFormer + upscale |
| `flux-kontext-restoration-fast-face.json` | Photo Restoration | Subject-specific variant |
| `flux-kontext-restoration-fast-facev1.json` | Photo Restoration | Subject-specific variant |
| `flux-kontext-restoration-fast-grandma.json` | Photo Restoration | Subject-specific variant (no CodeFormer) |
| `flux-kontext-restoration-fast-nocolor.json` | Photo Restoration | No-color variant (no CodeFormer) |
| `flux-kontext-restoration-fast-papado.json` | Photo Restoration | Subject-specific variant |
| `flux-kontext-restoration-fast-papado2.json` | Photo Restoration | Subject-specific variant |
| `wan22_i2v_portrait_animation.json` | Wan 2.2 I2V | Main portrait animation workflow |
| `wan22_i2v_portrait_animation-walk towards.json` | Wan 2.2 I2V | Motion variant |

A copy of the main Wan 2.2 workflow also lives at `~/comfy/wan22_i2v_portrait_animation.json`.

---

## Workflow 1 — LTX-2.3 Distilled fp8 (Text-to-Video)

### Overview

Uses the full fp8 distilled checkpoint loaded as a standard ComfyUI checkpoint. Faster than
the GGUF workflow — 9 steps with the `res_multistep` sampler. Has a custom trained character
LoRA (`my_first_lora_v1_copy.safetensors`) applied at strength 1.1.

### Pipeline

| Stage | Node | File / Setting |
|---|---|---|
| Model load | `CheckpointLoaderSimple` | `ltx-2.3-22b-distilled-fp8.safetensors` |
| Text encoder | `LTXAVTextEncoderLoader` | `gemma_3_12B_it.safetensors`, ref model: `ltx-2.3-22b-distilled-fp8.safetensors` |
| LoRA | `LoraLoaderModelOnly` | `my_first_lora_v1_copy.safetensors`, strength 1.1 |
| Conditioning | `LTXVConditioning` | LTX-specific |
| Latent | `EmptyLTXVLatentVideo` | |
| Sampler | `KSampler` | `res_multistep` scheduler, 9 steps |
| VAE decode | `VAEDecodeTiled` | tile_size=**256**, temporal_size=32 |
| Output | `SaveVideo` | |

### Baseline Parameters

| Parameter | Value |
|---|---|
| Resolution | 736×480 |
| Frames | 97 |
| Steps | 9 (res_multistep — distilled model) |
| Runtime | Faster than 20-step GGUF path |

### Hard Constraints (same as GGUF path)

- **Video dimensions must be multiples of 32.**
- **VAE decode must use tiled mode.** tile_size=**256**, temporal_size=32.
- **Do not upgrade PyTorch.** `2.6.0+cu124` works for all workflows.

---

## Workflow 2 — LTX-2.3 GGUF Q3 + ID-LoRA (Text-to-Video with Identity)

### Pipeline

LTX-2.3 is an audio-video model. The standard `CLIPLoaderGGUF` + `KSampler` path does NOT
work — it produces a tensor dimension mismatch. Use only LTX-specific nodes.

| Stage | Node | File / Setting |
|---|---|---|
| Transformer | `UnetLoaderGGUF` | `ltx-2.3/ltx-2.3-22b-dev-Q3_K_M.gguf` |
| Text encoder + connector | `DualCLIPLoaderGGUF` | clip_name1: Gemma GGUF, clip_name2: text projection, type: `ltxv` |
| Conditioning | `LTXVConditioning` | LTX-specific — not `CLIPTextEncode` |
| Scheduler | `LTXVScheduler` | LTX-specific — not `BasicScheduler` |
| Sampler | `SamplerCustomAdvanced` | not `KSampler` |
| VAE decode | `VAEDecodeTiled` | tile_size=**256**, temporal_size=32 |
| Character LoRA | ID-LoRA nodes | `IDLoraTwoStageModelLoader` / `IDLoraTwoStageSampler` |

### Baseline Parameters

| Parameter | Value |
|---|---|
| Resolution | 512×512 (must be multiples of 32; safe presets: 512×512, 768×512, 1024×576) |
| Frames | 17 (formula: 1 + 8N, minimum N=2; valid: 17, 25, 33, 49, 65…) |
| Steps | 20 |
| CFG | 3.5 |
| Runtime | ~2 min on the 3090 |

### VRAM Budget

| Component | GPU Memory |
|---|---|
| LTX-2.3-22B Transformer (GGUF Q3_K_M) | ~11.0 GB |
| Text Projection / Embeddings Connector | ~2.2 GB |
| VAE decode (tiled 256×32) | ~1.9 GB |
| **Peak during sampling** | **~13.2 GB (55% of 24 GB)** |
| Text Encoder (Gemma-3 Q4_K_M) | CPU offload |

10+ GB headroom on the 3090. `--lowvram` is not required.

### LTX-2.3 Hard Constraints

- **Video dimensions must be multiples of 32.** Non-aligned values cause memory access faults.
- **Text encoder must be CPU-offloaded.** Gemma-3 GGUF offloads to RAM after encoding.
- **VAE decode must use tiled mode.** tile_size=**256**, temporal_size=32. Do not increase.
- **Do not upgrade PyTorch.** `2.6.0+cu124` works for all three workflows.
- **transformers at 5.4.0.** ID-LoRA was authored against `< 5.0`. If `AttributeError` surfaces during LoRA loading:
  ```bash
  source ~/comfy/env.sh
  pip install 'transformers>=4.52,<5'
  ```
  Do not apply preemptively.

---

## Workflow 3 — Photo Restoration

### Overview

Multi-stage pipeline for restoring degraded, aged, or low-quality photographs. Operates
entirely on still images — no video generation involved.

Multiple workflow variants exist for different subjects. The "fast" variants include a
CodeFormer face-reconstruction pass and upscaling. The "minimal" (`flux-kontext-restoration.json`)
runs only the FLUX Kontext pass.

- **Pass 1 — FLUX.1 Kontext (fp8 scaled):** Structural restoration and overall image
  enhancement using the FLUX.1 Kontext inpainting/editing model.
- **Pass 2 — Face restore (optional):** CodeFormer or GFPGAN for face-specific sharpening.
- **Pass 3 — Upscale (optional):** ESRGAN / RealESRGAN for final resolution increase.
- **Alternative: SUPIR** — heavy restoration and upscaling via SUPIR-v0F / SUPIR-v0Q.

### Pipeline (full variant)

| Stage | Node | Model |
|---|---|---|
| Load image | `LoadImage` | Input photo |
| Pass 1 — FLUX Kontext | `UNETLoader` + sampler | `flux1-dev-kontext_fp8_scaled.safetensors` |
| Text encoders | `DualCLIPLoader` | `t5xxl_fp16.safetensors` + `clip_l.safetensors` |
| FLUX VAE | `VAELoader` | `ae.safetensors` |
| Pass 2 — Face restore | `FaceRestoreCFWithModel` | `codeformer.pth` (or `GFPGANv1.4.pth`) |
| Pass 3 — Upscale | `ImageUpscaleWithModel` | `4x-foolhardy-Remacri.pth` or `RealESRGAN_x4plus.pth` |
| Save | `SaveImage` | Output |

### VRAM Budget

FLUX.1 Kontext fp8 peaks at ~10–16 GB depending on resolution. CodeFormer adds ~360 MB.
Well within the 24 GB envelope.

---

## Workflow 4 — Photo Animation (Wan 2.2 I2V)

### Validated Result — 2026-05-05

- 20-step baseline: 832×480, 81 frames, CFG 3.5, euler/simple
- Wall time: **17 min 11 sec** (~50 s/step × 10 steps per expert × 2 experts)
- Peak VRAM: **15.1 GB** (61% of 24 GB) — 9 GB headroom, no OOM
- MoE switch confirmed: high-noise expert (steps 0–10) → offload → low-noise (steps 10–20)
- Output: `~/comfy/ComfyUI/output/video/Wan2.2_i2v_00001_.mp4`

### Architecture

Wan 2.2 14B uses a **Mixture-of-Experts (MoE) architecture** with separate high-noise and
low-noise expert transformers. They run sequentially — ComfyUI offloads the inactive expert
during the switch. Both transformer files are required; neither alone is sufficient.

### Loader Ecosystem Note

ComfyUI has two parallel loader paths for Wan — do not mix them:

1. **Native ComfyUI loaders** — accept `umt5_xxl_fp8_e4m3fn_scaled.safetensors`. **This is our path.**
2. **WanVideoWrapper loaders** — reject fp8 scaled, require bf16. Raises
   `ValueError("fp8 scaled is not supported by this node")` if given the Comfy-Org file.

### Sampler Configuration

| Mode | Steps | CFG | Split at | Time (3090, measured) | Use for |
|---|---|---|---|---|---|
| Original (Lightning OFF) | 20 | 3.5 | step 10 | **17 min 11 sec** | Finals |
| Turbo (Lightning ON) | 4 | 1.0 | step 2 | ~2–4 min (estimated) | Scouting |

Baseline: 832×480, 81 frames (5 sec @ 16 fps), euler/simple sampler.

### VRAM Budget (measured 2026-05-05)

| Component | GPU Memory |
|---|---|
| Wan 2.2 14B transformer (GGUF Q5_K_M, one expert at a time) | 10.4 GB |
| UMT5-XXL text encoder (fp8 scaled, CPU-offloaded after encoding) | 6.4 GB loaded |
| Wan VAE + latents + KV cache (81 frames @ 832×480) | ~4.7 GB |
| **Actual peak during sampling** | **15.1 GB (61% of 24 GB)** |

No `--lowvram` needed. `--reserve-vram 1.0` is set in `launch.sh`.

### Failure Mode Reference

| Symptom | Cause | Fix |
|---|---|---|
| `Unknown node type: UnetLoaderGGUF` | ComfyUI-GGUF didn't import | Check `custom_nodes/ComfyUI-GGUF/` exists; restart |
| Red filename in loader dropdown | Symlink missing or broken | Check symlinks in `diffusion_models/` and `loras/` |
| OOM at sampling | Resolution or frame count too large | Drop to 49 frames or 480×480 |
| `Tensor size mismatch` | LoRA version mismatch | Confirm both LoRAs are I2V not T2V |
| Pure noise output | ModelSamplingSD3 shift wrong | Both `ModelSamplingSD3` nodes should be 5.0 |
| Identity drifts at ~frame 40 | Temporal coherence limit | Drop to 49 frames; or lower CFG to 3.0 |

---

## Next Steps

### 1. Test Lightning mode (4-step turbo)

Open the Wan 2.2 subgraph on the canvas, toggle "Enable 4steps LoRA?" to `true`. Queue
with same portrait image. Expected: ~2–4 minutes. Use Lightning for scouting, 20-step
for finals.

### 2. Install sageattention (~25-30% speedup)

```bash
source ~/comfy/env.sh
pip install sageattention
```

Then edit `~/comfy/launch.sh`: replace `--use-pytorch-cross-attention` with
`--use-sage-attention`. Triton 3.2.0 is already in the venv. Brings 17 min toward ~12 min.

### 3. Re-verify LTX-2.3 GGUF workflow on RTX 3090

LTX-2.3 GGUF + ID-LoRA text-to-video was validated on the RX 6800 XT but has not been run
on CUDA. The Distilled fp8 workflow (`ltx2.3_video-test.json`) is the active path.

---

## Constraints & Rules

- **Do not reinstall or downgrade PyTorch.** `2.6.0+cu124` is the correct version for the
  current CUDA stack.

- **No bitsandbytes.** Transitive dependency of `ltx-trainer`. Uninstall immediately if
  any `pip install` pulls it in: `pip uninstall bitsandbytes -y`.

- **No ROCm packages.** The ROCm venv is gone. The backup files (`env.sh.rocm-backup`,
  `launch.sh.rocm-backup`) are reference-only.

- **Always source env.sh first.** Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  and activates the venv.

- **Show diffs before editing files.** Before modifying `launch.sh`, `env.sh`, or any
  custom node file, show the current content of the relevant section and explain the change.

- **Autonomous execution.** Do not ask for confirmation before running commands unless the
  action is destructive (deleting files, uninstalling packages, modifying core ComfyUI files).

---

## Useful One-Liners

```bash
# Source environment (always do this first)
source ~/comfy/env.sh

# Launch ComfyUI — UI at http://localhost:8188
~/comfy/launch.sh

# GPU status
nvidia-smi
watch -n 1 nvidia-smi

# Confirm GPU is visible to PyTorch
source ~/comfy/env.sh && python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"

# --- Model checks ---

# LTX-2.3 Distilled fp8 workflow models
ls -lah ~/comfy/ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors
ls -lah ~/comfy/ComfyUI/models/text_encoders/gemma_3_12B_it.safetensors
ls -lah ~/comfy/ComfyUI/models/loras/my_first_lora_v1_copy.safetensors

# LTX-2.3 GGUF workflow models
find ~/comfy/ComfyUI/models/diffusion_models/ltx-2.3 \
     ~/comfy/ComfyUI/models/text_encoders/ltx-2.3 \
     ~/comfy/ComfyUI/models/vae/ltx-2.3 \
     -type f -ls 2>/dev/null

# FLUX / Photo Restoration models
ls -lah ~/comfy/ComfyUI/models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors
ls -lah ~/comfy/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors
ls -lah ~/comfy/ComfyUI/models/text_encoders/clip_l.safetensors
ls -lah ~/comfy/ComfyUI/models/vae/ae.safetensors
ls -lah ~/comfy/ComfyUI/models/facerestore_models/

# Wan 2.2 models and symlinks
ls -lah ~/comfy/ComfyUI/models/diffusion_models/wan2.2* \
        ~/comfy/ComfyUI/models/text_encoders/umt5* \
        ~/comfy/ComfyUI/models/vae/wan_2.1* \
        ~/comfy/ComfyUI/models/loras/wan2.2* \
        ~/comfy/ComfyUI/models/loras/Wan2.2* 2>/dev/null

# All checkpoints
ls -lah ~/comfy/ComfyUI/models/checkpoints/*.safetensors \
        ~/comfy/ComfyUI/models/checkpoints/*.ckpt 2>/dev/null

# --- ComfyUI ops ---

# Check what's listening on port 8188
ss -tlnp 2>/dev/null | grep 8188

# Kill any hanging ComfyUI process
pkill -f "python.*main.py"

# Check custom node import status at startup
grep -E "(ERROR|IMPORT|Traceback)" ~/comfy/comfyui.log 2>/dev/null | tail -20

# Confirm --reserve-vram is in launch.sh (not --lowvram)
grep -E "reserve-vram|lowvram" ~/comfy/launch.sh
```

---

## Deferred Items

| Item | Notes |
|---|---|
| Wan 2.2 Lightning mode test | 4-step turbo — toggle "Enable 4steps LoRA?" in subgraph. Expected ~2-4 min. **Do next.** |
| sageattention install | ~25-30% speedup on Ampere. Triton 3.2.0 already present. Switch `--use-pytorch-cross-attention` → `--use-sage-attention` in `launch.sh`. |
| LTX-2.3 GGUF re-validation on CUDA | Validated on RX 6800 XT; Distilled fp8 path is now primary. GGUF path still works for ID-LoRA workflow. |
| `ltx-2.3-22b-distilled-1.1_transformer_only_mxfp8_block32.safetensors` path issue | Saved in `diffusion_models/diffusion_models/` (double-nested). May need to move if a workflow references it directly. |
| SUPIR workflow | SUPIR-v0F, SUPIR-v0Q, and ComfyUI-SUPIR are all present. No saved workflow yet. |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.1 GB official distilled LoRA present. Not yet used in saved workflows. |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | LTX spatial upscaler present. Not yet wired into a workflow. |
