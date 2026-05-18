# CLAUDE.md — ComfyUI Workspace (chuckai)

## Context

You are working on **chuckai** — a local Ubuntu 22.04 server with an **NVIDIA GeForce RTX 3090**
(24 GB VRAM) running CUDA 12.4. The workspace is at `~/comfy/`.

The GPU was migrated from an AMD RX 6800 XT / ROCm 6.3 setup to a full CUDA rebuild
(completed 2026-05-04). ROCm is no longer in use for this workspace.

Three production workflows are currently active:

1. **LTX-2.3 22B GGUF + Character LoRA** (text-to-video with identity) — generates videos
   from text prompts with identity-consistent character rendering via ID-LoRA nodes.

2. **Photo Restoration** — two-pass pipeline: FLUX.1 Kontext fp8 (structural restore) +
   CodeFormer (face reconstruction). Operates on still images.

3. **Photo Animation — Wan 2.2 I2V 14B GGUF** (image-to-video) — animates still portrait
   photos with strong face identity preservation using a MoE two-expert architecture.
   First inference validated end-to-end 2026-05-05. Output: `video/Wan2.2_i2v_00001_.mp4`.

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

| Node | Commit | Status |
|---|---|---|
| ComfyUI-GGUF (city96) | `6ea2651` (2026-01-12) | OK — required for LTX-2.3 and Wan 2.2 GGUF loading |
| ComfyUI-Manager (ltdrdata) | `8d5c1203` (2026-05-01) | OK |
| ComfyUI-WanVideoWrapper (kijai) | `df8f3e4` (2026-02-22) | OK — required for Wan 2.2 I2V |
| ComfyUI-KJNodes (kijai) | `cd5ad80` (2026-05-03) | OK — utility nodes for Wan pipeline |
| ComfyUI-VideoHelperSuite (Kosinkadink) | `2984ec4` (2026-04-06) | OK — MP4 output |
| ID-LoRA-LTX2.3-ComfyUI | `9943746` (2026-03-25) | Active — required for LTX-2.3 Character LoRA workflow |

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

### Wan 2.2 I2V 14B GGUF (photo animation — validated 2026-05-05)

The downloader used underscore-style filenames. Symlinks with the hyphen-style names
expected by the workflow JSON were created in-place:

```
~/comfy/ComfyUI/models/
├── diffusion_models/
│   ├── wan2.2_i2v_high_noise_14B_Q5_K_M.gguf          (11 GB — actual file)
│   ├── wan2.2_i2v_low_noise_14B_Q5_K_M.gguf           (11 GB — actual file)
│   ├── wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf          → symlink
│   └── wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf           → symlink
├── text_encoders/
│   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors         (6.3 GB)
├── vae/
│   └── wan_2.1_vae.safetensors                        (243 MB)
└── loras/
    ├── wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors  (1.2 GB — actual file)
    ├── wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors   (1.2 GB — actual file)
    ├── Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors          → symlink
    └── Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors           → symlink
```

### LTX-2.3 22B GGUF (text-to-video + Character LoRA)

All files present and intact. Do not move, rename, or re-download.

| Model | File | Size | Path |
|---|---|---|---|
| Transformer | `ltx-2.3-22b-dev-Q3_K_M.gguf` | 11 GB | `models/diffusion_models/ltx-2.3/` |
| Text Encoder | `google_gemma-3-12b-it-Q4_K_M.gguf` | 6.8 GB | `models/text_encoders/ltx-2.3/` |
| Embeddings Connector | `ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | `models/text_encoders/ltx-2.3/` |
| VAE | `LTX23_video_vae_bf16.safetensors` | 1.4 GB | `models/vae/ltx-2.3/` |
| ID-LoRA weights | *(see `models/loras/ltx-2.3/`)* | ~1.1 GB | `models/loras/ltx-2.3/` |

Sources (for re-download if needed):
- Transformer: `unsloth/LTX-2.3-GGUF` (NOT `Lightricks/LTX-Video-2.3-22B-GGUF` — that repo does not exist)
- Text encoder, embeddings connector, VAE: `Kijai/LTX2.3_comfy`
- ID-LoRA weights: `Lightricks/LTX-Video-2.3`
- Full fp16 checkpoint (`ltx-2.3-22b-dev.safetensors`) is 43 GB — do not download

### Photo Restoration Models

| File | Purpose | Directory |
|---|---|---|
| `flux1-dev-kontext_fp8_scaled.safetensors` | FLUX.1 Kontext — structural restore (Pass 1) | `models/diffusion_models/` or `models/unet/` |
| `codeformer.pth` | Face reconstruction (Pass 2) | `models/facerestore_models/` |

FLUX.1 Kontext also requires T5-XXL + CLIP-L text encoders and the FLUX VAE — verify
those are present in `models/clip/` and `models/vae/` respectively.

---

## Workflow 1 — LTX-2.3 22B + Character LoRA (Text-to-Video)

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
| Runtime | ~2 min on the 3090 (was ~3:15 on RX 6800 XT) |

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

## Workflow 2 — Photo Restoration

### Overview

Two-pass pipeline for restoring degraded, aged, or low-quality photographs.

- **Pass 1 — FLUX.1 Kontext (fp8 scaled):** Structural restoration and overall image
  enhancement using the FLUX.1 Kontext inpainting/editing model.
- **Pass 2 — CodeFormer:** Face-specific reconstruction to sharpen facial features.

Operates entirely on still images — no video generation involved.

### Pipeline

| Stage | Node | Model |
|---|---|---|
| Load image | `LoadImage` | Input photo |
| Pass 1 — FLUX Kontext | FLUX UNet loader + sampler | `flux1-dev-kontext_fp8_scaled.safetensors` |
| Pass 2 — Face restore | CodeFormer node | `codeformer.pth` |
| Save | `SaveImage` | Output |

### VRAM Budget

FLUX.1 Kontext fp8 peaks at ~10–16 GB depending on resolution. CodeFormer adds ~340 MB.
Well within the 24 GB envelope.

---

## Workflow 3 — Photo Animation (Wan 2.2 I2V)

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

### Loader Node → Filename Mapping

Workflow JSON expects the hyphen-style names. Symlinks on disk point to the actual
underscore-style files downloaded by the downloader.

| Node | Filename (symlink) | Actual file on disk |
|---|---|---|
| Load High-Noise Transformer | `wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf` | `wan2.2_i2v_high_noise_14B_Q5_K_M.gguf` |
| Load Low-Noise Transformer | `wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf` | `wan2.2_i2v_low_noise_14B_Q5_K_M.gguf` |
| CLIPLoader | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | (same) |
| VAELoader | `wan_2.1_vae.safetensors` | (same) |
| High-Noise Lightning LoRA | `Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors` | `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` |
| Low-Noise Lightning LoRA | `Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors` | `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` |

### Loader Ecosystem Note

ComfyUI has two parallel loader paths for Wan — do not mix them:

1. **Native ComfyUI loaders** — accept `umt5_xxl_fp8_e4m3fn_scaled.safetensors`. **This is our path.**
2. **WanVideoWrapper loaders** — reject fp8 scaled, require bf16. Raises
   `ValueError("fp8 scaled is not supported by this node")` if given the Comfy-Org file.

### Sampler Configuration

| Mode | Steps | CFG | Split at | Time (3090, measured) | Use for |
|---|---|---|---|---|---|
| Original (Lightning OFF) | 20 | 3.5 | step 10 | **17 min 11 sec** | Finals |
| Turbo (Lightning ON) | 4 | 1.0 | step 2 | ~2-4 min (estimated) | Scouting |

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

Open the subgraph on the canvas, toggle "Enable 4steps LoRA?" to `true`. Queue with same
portrait image. Expected: ~2–4 minutes. Compare identity preservation vs baseline.
Use Lightning for scouting, 20-step for finals.

### 2. Install sageattention (~25-30% speedup)

```bash
source ~/comfy/env.sh
pip install sageattention
```

Then edit `~/comfy/launch.sh`: replace `--use-pytorch-cross-attention` with
`--use-sage-attention`. Triton 3.2.0 is already in the venv. Highest-leverage
optimization available — brings 17 min down toward ~12 min on the 3090.

### 3. Re-verify LTX-2.3 on RTX 3090

LTX-2.3 text-to-video was validated on the RX 6800 XT but has not been run on CUDA.
Should work fine and be faster. Test `--lowvram` OFF first (24 GB headroom makes it
unnecessary).

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

# Check Wan 2.2 symlinks and models
ls -lah ~/comfy/ComfyUI/models/diffusion_models/wan2.2* \
        ~/comfy/ComfyUI/models/text_encoders/umt5* \
        ~/comfy/ComfyUI/models/vae/wan_2.1* \
        ~/comfy/ComfyUI/models/loras/Wan2.2* \
        ~/comfy/ComfyUI/models/loras/wan2.2* 2>/dev/null

# Check LTX-2.3 models
find ~/comfy/ComfyUI/models/diffusion_models/ltx-2.3 \
     ~/comfy/ComfyUI/models/text_encoders/ltx-2.3 \
     ~/comfy/ComfyUI/models/vae/ltx-2.3 \
     -type f -ls 2>/dev/null

# Check photo restoration models
ls -lah ~/comfy/ComfyUI/models/diffusion_models/flux1-dev-kontext* 2>/dev/null
ls -lah ~/comfy/ComfyUI/models/unet/flux1-dev-kontext* 2>/dev/null
ls -lah ~/comfy/ComfyUI/models/facerestore_models/codeformer.pth 2>/dev/null

# Check what's listening on port 8188
ss -tlnp 2>/dev/null | grep 8188

# Kill any hanging ComfyUI process
pkill -f "python.*main.py"

# Check custom node import status
grep -E "(ERROR|IMPORT|Traceback)" ~/comfy/comfyui.log 2>/dev/null | tail -20

# Confirm --reserve-vram in launch.sh (not --lowvram)
grep -E "reserve-vram|lowvram" ~/comfy/launch.sh
```

---

## Deferred Items

| Item | Notes |
|---|---|
| sageattention install | ~25-30% attention speedup on Ampere. Triton 3.2.0 already present. Switch `--use-pytorch-cross-attention` → `--use-sage-attention` in `launch.sh`. Do after Lightning test. |
| Q4_K_M Wan transformers | ~2 GB VRAM savings per expert vs Q5_K_M. Enables 720p resolution headroom. |
| LTX-2.3 re-validation on CUDA | Pipeline worked on RX 6800 XT; should be faster on 3090. `--lowvram` likely not needed. Do after Lightning test + sageattention. |
| LTX-2.3 ID-LoRA weights download | ~1.1 GB from `Lightricks/LTX-Video-2.3`. Defer until LTX identity work resumes. |
| VideoForge Wan 2.1 VACE 1.3B | Existing `~/videoforge/` .pth files are compatible with WanVideoWrapper. Potential fast-draft option; needs a separate workflow. |
