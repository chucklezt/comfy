# CLAUDE.md — ComfyUI Workspace (chuckai)

## Context

You are working on **chuckai** — a local Ubuntu 22.04 server with an **NVIDIA GeForce RTX 3090**
(24 GB VRAM) running CUDA 12.4. The workspace is at `~/comfy/`.

The GPU was migrated from an AMD RX 6800 XT / ROCm 6.3 setup to a full CUDA rebuild
(completed 2026-05-04). ROCm is no longer in use for this workspace. The RTX 3090 is the
active GPU.

Two pipelines are operational or in final setup:

1. **LTX-2.3 22B GGUF** (text-to-video) — validated working end-to-end on the RX 6800 XT
   before the migration. Model files are present and intact. Pipeline has NOT been re-validated
   on the RTX 3090 yet — first run will confirm it still works (it should; CUDA is more
   capable, not less).

2. **Wan 2.2 I2V 14B GGUF** (image-to-video, portrait animation) — model downloads in
   progress as of session end. Workflow JSON built and ready to load. First inference not
   yet run.

**Current priority: complete Wan 2.2 I2V first inference.** See NEXT STEPS below.

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
| ComfyUI-GGUF (city96) | `6ea2651` (2026-01-12) | OK |
| ComfyUI-Manager (ltdrdata) | `8d5c1203` (2026-05-01) | OK |
| ComfyUI-WanVideoWrapper (kijai) | `df8f3e4` (2026-02-22) | OK |
| ComfyUI-KJNodes (kijai) | `cd5ad80` (2026-05-03) | OK |
| ComfyUI-VideoHelperSuite (Kosinkadink) | `2984ec4` (2026-04-06) | OK |
| ID-LoRA-LTX2.3-ComfyUI | `9943746` (2026-03-25) | IMPORT FAIL — `ltx_core` not installed. Does not affect Wan 2.2. ComfyUI skips gracefully. |

**Known cosmetic notice at startup:**
`ComfyUI-GGUF: Partial torch compile only, consider updating pytorch` — informational only.
GGUF loading and inference work fine on torch 2.6. Do not bump torch to 2.7+ without
validating wheel compatibility (sageattention, flash-attn).

---

## Model Inventory

### Wan 2.2 I2V 14B GGUF (image-to-video — in setup)

Target layout — downloads were in progress at session end (tmux session `wan22-dl`):

```
~/comfy/ComfyUI/models/
├── diffusion_models/
│   ├── wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf   (~10 GB, bullerwins/Wan2.2-I2V-A14B-GGUF)
│   └── wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf    (~10 GB, same)
├── text_encoders/
│   └── umt5_xxl_fp8_e4m3fn_scaled.safetensors  (~6.7 GB, Comfy-Org/Wan_2.1_ComfyUI_repackaged)
├── vae/
│   └── wan_2.1_vae.safetensors                 (~250 MB, Comfy-Org/Wan_2.2_ComfyUI_Repackaged)
└── loras/
    ├── Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors  (~600 MB, lightx2v/Wan2.2-Lightning)
    └── Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors   (~600 MB, same)
```

Verify completion before proceeding:
```bash
ls -lah ~/comfy/ComfyUI/models/diffusion_models/wan2.2-i2v-A14B-*Noise-Q5_K_M.gguf
ls -lah ~/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
ls -lah ~/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors
ls -lah ~/comfy/ComfyUI/models/loras/Wan2.2-Lightning_I2V-A14B-4steps_*.safetensors
```

If `wan22-dl` is still running: `tmux attach -t wan22-dl` (Ctrl-b d to detach).

### LTX-2.3 22B GGUF (text-to-video — validated, awaiting re-verification on CUDA)

All files present and intact from prior work. Do not move, rename, or re-download.

| Model | File | Size | Path |
|---|---|---|---|
| Transformer | `ltx-2.3-22b-dev-Q3_K_M.gguf` | 11 GB | `models/diffusion_models/ltx-2.3/` |
| Text Encoder | `google_gemma-3-12b-it-Q4_K_M.gguf` | 6.8 GB | `models/text_encoders/ltx-2.3/` |
| Embeddings Connector | `ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | `models/text_encoders/ltx-2.3/` |
| VAE | `LTX23_video_vae_bf16.safetensors` | 1.4 GB | `models/vae/ltx-2.3/` |
| ID-LoRA weights | *(not yet downloaded)* | ~1.1 GB | `models/loras/ltx-2.3/` |

**Sources (for re-download if needed):**
- Transformer: `unsloth/LTX-2.3-GGUF` (NOT `Lightricks/LTX-Video-2.3-22B-GGUF` — that repo does not exist)
- Text encoder, embeddings connector, VAE: `Kijai/LTX2.3_comfy`
- Full fp16 checkpoint (`ltx-2.3-22b-dev.safetensors`) is 43 GB — do not download

---

## NEXT STEPS

### 1. Verify Wan 2.2 downloads completed

```bash
ls -lah ~/comfy/ComfyUI/models/diffusion_models/wan2.2-i2v-A14B-*Noise-Q5_K_M.gguf
ls -lah ~/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
ls -lah ~/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors
ls -lah ~/comfy/ComfyUI/models/loras/Wan2.2-Lightning_I2V-A14B-4steps_*.safetensors
```

All six files must exist with non-zero sizes.

### 2. Load the Wan 2.2 workflow and run first inference

The workflow JSON `wan22_i2v_portrait_animation.json` was built and validated in the
prior session. Get it onto chuckai and drag onto the ComfyUI canvas.

**Launch ComfyUI:**
```bash
~/comfy/launch.sh
# UI at http://localhost:8188
```

**Loader node → filename mapping** — verify each dropdown on the canvas matches:

| Node | Expected filename |
|---|---|
| Load High-Noise Transformer (GGUF) | `wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf` |
| Load Low-Noise Transformer (GGUF) | `wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf` |
| CLIPLoader | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| VAELoader | `wan_2.1_vae.safetensors` |
| High-Noise Lightning LoRA | `Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors` |
| Low-Noise Lightning LoRA | `Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors` |

**Baseline first run** (Lightning OFF by default in the workflow):
- Upload a portrait photo to the Load Image node
- 832×480, 81 frames, 20 steps, CFG 3.5, euler/simple
- Expected ~3-5 minutes on the 3090
- Look for: identity preservation, subtle natural motion, no talking artifacts

**Once baseline works — try Lightning:**
- Open the subgraph, toggle "Enable 4steps LoRA?" to `true`
- Same image, same prompt — ~30-60 seconds
- Use Lightning for scouting, 20-step for finals

**Common first-run failure modes:**

| Symptom | Likely cause | Fix |
|---|---|---|
| `Unknown node type: UnetLoaderGGUF` | ComfyUI-GGUF didn't import | Check custom node exists; restart ComfyUI |
| Red filename in loader dropdown | File missing or wrong directory | Re-run the verify step above |
| OOM at sampling | Frames or resolution too large | Drop to 49 frames or 480×480 |
| `Tensor size mismatch` | LoRA version mismatch | Confirm both LoRAs are I2V, not T2V |
| Pure noise output | ModelSamplingSD3 shift wrong | Both ModelSamplingSD3 nodes should be 5.0 |
| Identity drifts at ~frame 40 | Temporal coherence limit | Drop to 49 frames; or lower CFG to 3.0 |

### 3. Optional optimizations (after baseline + Lightning both work)

- **sageattention**: ~25-30% attention speedup on Ampere. Triton 3.2.0 is already in the
  venv. Install, then switch `--use-pytorch-cross-attention` → `--use-sage-attention` in
  `~/comfy/launch.sh`.
- **Q4_K_M transformers**: ~2 GB VRAM savings per expert if pushing toward 720p resolution.

---

## Wan 2.2 I2V Pipeline Reference

### Architecture note — two loader ecosystems

ComfyUI has two parallel loader paths for Wan:

1. **Native ComfyUI loaders** (`Load CLIP`, `Load VAE`, `Unet Loader (GGUF)`) — accept
   `umt5_xxl_fp8_e4m3fn_scaled.safetensors`. This is our path.
2. **WanVideoWrapper loaders** (`Load WanVideo T5 TextEncoder`, `WanVideo VAE Loader`) —
   reject fp8 scaled, require bf16. WanVideoWrapper's `LoadWanVideoT5TextEncoder` raises
   `ValueError("fp8 scaled is not supported by this node")` if given the fp8-scaled file.

GGUF transformers use native loaders. Do not mix the two ecosystems.

### MoE two-stage sampling

Wan 2.2 14B uses a Mixture-of-Experts architecture with separate high-noise and low-noise
expert transformers. They load and run sequentially — ComfyUI offloads the inactive expert
during the switch. Both must be present; neither alone is sufficient.

### VRAM budget (RTX 3090, 24 GB)

| Component | GPU Memory |
|---|---|
| Wan 2.2 14B transformer (GGUF Q5_K_M, one expert at a time) | ~10-12 GB |
| UMT5-XXL text encoder (fp8 scaled, CPU offloadable) | ~6 GB |
| Wan VAE decode | ~2-3 GB |
| Latents + KV cache (81 frames @ 832×480) | ~2-3 GB |
| **Expected peak** | **~16-20 GB — fits on 24 GB with `--reserve-vram 1.0`** |

No `--lowvram` needed for Wan 2.2. `--reserve-vram 1.0` is set in `launch.sh`.

---

## LTX-2.3 Pipeline Reference

### Confirmed working pipeline (text-to-video, validated on RX 6800 XT — re-verify on 3090)

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

### Baseline inference parameters (validated on RX 6800 XT)

| Parameter | Value |
|---|---|
| Resolution | 512 x 512 |
| Frames | 17 (formula: 1 + 8N, minimum N=2) |
| Steps | 20 |
| CFG | 3.5 |
| Runtime | ~3 min 15 sec (RX 6800 XT; 3090 will be faster) |

### VRAM budget (RTX 3090 — previously validated on 16 GB RX 6800 XT, headroom is larger now)

| Component | GPU Memory |
|---|---|
| LTX-2.3-22B Transformer (GGUF Q3_K_M) | ~11.0 GB |
| Text Projection / Embeddings Connector | ~2.2 GB |
| VAE decode (tiled 256x32) | ~1.9 GB |
| **Peak during sampling** | **~13.2 GB** |
| Text Encoder (Gemma-3 Q4_K_M) | CPU offload |

On the 3090 (24 GB), the LTX-2.3 pipeline has 10+ GB headroom. The `--lowvram` flag that
was required on the 16 GB RX 6800 XT may no longer be needed. Test without it first;
add it back if VAE decode OOMs.

### LTX-2.3 constraints (still apply on CUDA)

- **Video dimensions must be multiples of 32.** Non-aligned values cause memory access faults.
  Safe presets: 512×512, 768×512, 1024×576.
- **Text encoder must be CPU-offloaded.** Gemma-3 GGUF offloads to RAM after encoding. Do
  not change this behavior.
- **VAE decode must use tiled mode.** tile_size=**256**, temporal_size=32. Do not increase
  tile_size or switch to full decode without testing OOM behavior first.
- **Do not upgrade PyTorch** for LTX compatibility reasons — the current 2.6.0+cu124 stack
  works for both pipelines.

### Watch item — transformers version

Currently at `5.4.0`. ID-LoRA-LTX2.3-ComfyUI was authored against `transformers < 5.0`.
No conflict observed during text-to-video inference. If an `AttributeError` or `ImportError`
traceable to `transformers` surfaces during LoRA loading:
```bash
source ~/comfy/env.sh
pip install 'transformers>=4.52,<5'
```
Do not apply preemptively.

---

## Constraints & Rules

- **Do not reinstall or downgrade PyTorch.** `2.6.0+cu124` is the correct version for the
  current CUDA stack. Do not run `pip install --upgrade torch` or similar.

- **No bitsandbytes.** It is a CUDA-only package but has caused segfaults in past
  configurations. If any `pip install` pulls it in as a transitive dependency, uninstall
  immediately: `pip uninstall bitsandbytes -y`.

- **No ROCm packages.** The ROCm venv is gone. Do not install anything from
  `download.pytorch.org/whl/rocm*` or reference `HSA_OVERRIDE_GFX_VERSION`. The backup
  files (`env.sh.rocm-backup`, `launch.sh.rocm-backup`) are reference-only.

- **Always source env.sh first.** Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  and activates the venv.

- **Show diffs before editing files.** Before modifying `launch.sh`, `env.sh`, or any
  custom node file, show the current content of the relevant section and explain the change.

- **Autonomous execution.** Do not ask for confirmation before running commands unless the
  action is destructive (deleting files, uninstalling packages, modifying core ComfyUI
  files). For downloads, workflow submissions, and diagnostic commands, proceed without
  asking.

---

## Useful One-Liners

```bash
# Check GPU VRAM usage
nvidia-smi

# Confirm GPU is visible to PyTorch
source ~/comfy/env.sh && python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.cuda)"

# Confirm torch version
source ~/comfy/env.sh && python -c "import torch; print(torch.__version__)"

# Check Wan 2.2 models are present
ls -lah ~/comfy/ComfyUI/models/diffusion_models/wan2.2* \
        ~/comfy/ComfyUI/models/text_encoders/umt5* \
        ~/comfy/ComfyUI/models/vae/wan_2.1* \
        ~/comfy/ComfyUI/models/loras/Wan2.2* 2>/dev/null

# Check LTX-2.3 models are present
find ~/comfy/ComfyUI/models/diffusion_models/ltx-2.3 \
     ~/comfy/ComfyUI/models/text_encoders/ltx-2.3 \
     ~/comfy/ComfyUI/models/vae/ltx-2.3 \
     -type f -ls 2>/dev/null

# Check if wan22-dl download session is still running
tmux ls 2>/dev/null | grep wan22-dl

# Watch VRAM during generation
watch -n 1 nvidia-smi

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
| `ID-LoRA-LTX2.3-ComfyUI` import failure | `ltx_core` module not installed. Does not affect Wan 2.2 or LTX-2.3 text-to-video. Defer until LTX ID-LoRA work begins. |
| LTX-2.3 re-validation on RTX 3090 | Pipeline worked on RX 6800 XT; should work better on 3090. Verify after Wan 2.2 baseline is done. `--lowvram` may no longer be needed. |
| sageattention install | ~25-30% attention speedup on Ampere. Triton 3.2.0 already present. Do after Wan 2.2 baseline + Lightning both work. |
| LTX-2.3 ID-LoRA download | ~1.1 GB from `Lightricks/LTX-Video-2.3`. Defer until LTX identity work resumes. |
| VideoForge Wan 2.1 VACE 1.3B fast iteration | Existing `~/videoforge/` .pth files are compatible with WanVideoWrapper. Potential fast-iteration option later; would need a separate workflow. |
