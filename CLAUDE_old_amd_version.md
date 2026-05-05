# CLAUDE.md — LTX-2.3 Workspace (chuckai)

## Context

You are working on **chuckai** — a local Ubuntu 22.04 server with an AMD Radeon RX 6800 XT
(16 GB VRAM, RDNA2 / gfx1030) running ROCm 6.3. The workspace is at `~/comfy/`.

End-to-end validation is **complete**. The full LTX-2.3 22B GGUF pipeline is confirmed
working on the RX 6800 XT. This file is the ongoing operational reference for this workspace.

---

## Hardware & Stack Reference

| Item | Value |
|---|---|
| GPU | AMD RX 6800 XT — gfx1030 / RDNA2 |
| VRAM | 16 GB |
| ROCm | 6.3 |
| PyTorch | 2.9.1+rocm6.3 |
| Python venv | `~/comfy/venv/` |
| ComfyUI root | `~/comfy/ComfyUI/` |
| Custom nodes | `~/comfy/ComfyUI/custom_nodes/` |
| Model staging | `~/comfy/ComfyUI/models/` |
| env script | `~/comfy/env.sh` (source before every command) |
| Diagnostics | `~/comfy/diagnostics.py` |
| Launch script | `~/comfy/launch.sh` |

**Always source the environment before running any Python or ComfyUI commands:**
```bash
source ~/comfy/env.sh
```

---

## Confirmed Model Inventory

All models downloaded and validated. Do not move, rename, or re-download these files.

| Model | File | Size | Path |
|---|---|---|---|
| Transformer | `ltx-2.3-22b-dev-Q3_K_M.gguf` | 11 GB | `models/diffusion_models/ltx-2.3/` |
| Text Encoder | `google_gemma-3-12b-it-Q4_K_M.gguf` | 6.8 GB | `models/text_encoders/ltx-2.3/` |
| Embeddings Connector | `ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | `models/text_encoders/ltx-2.3/` |
| VAE | `LTX23_video_vae_bf16.safetensors` | 1.4 GB | `models/vae/ltx-2.3/` |
| ID-LoRA weights | *(not yet downloaded)* | ~1.1 GB | `models/loras/ltx-2.3/` |

**Sources:**
- Transformer: `unsloth/LTX-2.3-GGUF` (not `Lightricks/LTX-Video-2.3-22B-GGUF` — that repo does not exist)
- Text encoder + embeddings connector + VAE: `Kijai/LTX2.3_comfy`
- Full fp16 checkpoint (`ltx-2.3-22b-dev.safetensors`) is 43 GB — incompatible with 16 GB VRAM, do not download

---

## Confirmed Working Pipeline (Text-to-Video)

LTX-2.3 is an audio-video model. The standard ComfyUI `CLIPLoaderGGUF` + `KSampler` pipeline
does **not** work — it produces a tensor dimension mismatch. Always use the LTX-specific nodes
built into ComfyUI core.

| Stage | Node | File / Setting |
|---|---|---|
| Transformer | `UnetLoaderGGUF` | `ltx-2.3/ltx-2.3-22b-dev-Q3_K_M.gguf` |
| Text encoder + connector | `DualCLIPLoaderGGUF` | clip_name1: Gemma GGUF, clip_name2: text projection, type: `ltxv` |
| Conditioning | `LTXVConditioning` | LTX-specific — not `CLIPTextEncode` |
| Scheduler | `LTXVScheduler` | LTX-specific — not `BasicScheduler` |
| Sampler | `SamplerCustomAdvanced` | not `KSampler` |
| VAE decode | `VAEDecodeTiled` | tile_size=**256**, temporal_size=32 |

### Baseline inference parameters (validated)

| Parameter | Value |
|---|---|
| Resolution | 512 x 512 |
| Frames | 17 (formula: 1 + 8N, minimum N=2) |
| Steps | 20 |
| CFG | 3.5 |
| Runtime | ~3 min 15 sec |

### Session startup

```bash
source ~/comfy/env.sh
~/comfy/launch.sh
# UI at http://localhost:8188
```

---

## Constraints & Rules

- **Never install bitsandbytes.** It is CUDA-only and will segfault on RDNA2. The import
  hook in `block_bitsandbytes.py` will catch accidental re-introduction, but don't let it
  get that far. If any `pip install` pulls it in as a transitive dependency, uninstall
  immediately: `pip uninstall bitsandbytes -y`.

- **Never use adamw8bit or any bitsandbytes optimizer.** Use `adafactor` for any training
  config work.

- **Always source env.sh first.** `HSA_OVERRIDE_GFX_VERSION=10.3.0` must be set or PyTorch
  will not see the GPU correctly.

- **Video dimensions must be multiples of 32.** Non-aligned values cause HIP memory access
  faults. Safe presets: 512x512, 768x512, 1024x576.

- **Text encoder must be CPU-offloaded.** The Gemma-3 GGUF offloads to system RAM after
  encoding. This is what makes the 16 GB budget work. Do not change this behavior.

- **VAE decode must use tiled mode.** tile_size=**256**, temporal_size=32. tile_size=512
  causes OOM. Do not switch to full VAE decode or increase tile_size above 256.

- **--lowvram flag is required in launch.sh.** This forces the transformer to offload before
  VAE decode, creating the headroom needed. Do not remove this flag.

- **Do not upgrade PyTorch.** `2.9.1+rocm6.3` is the correct version. Do not run
  `pip install --upgrade torch` or similar.

- **Show diffs before editing files.** Before modifying launch.sh, env.sh, or any custom
  node file, show the current content of the relevant section and explain the change.

- **Autonomous execution.** Do not ask for confirmation before running commands unless the
  action is destructive (deleting files, uninstalling packages, modifying core ComfyUI files).
  For downloads, workflow submissions, and diagnostic commands, proceed without asking.

---

## Watch Items (not broken, monitor as we go)

### transformers version
Currently at `5.4.0`. The ID-LoRA-LTX2.3-ComfyUI node was authored against `transformers < 5.0`
and the README flags a potential incompatibility. No conflict has been observed during
text-to-video inference. Monitor for errors during LoRA loading or prompt encoding.
If you see an `AttributeError` or `ImportError` traceable to `transformers`, the fix is:
```bash
source ~/comfy/env.sh
pip install 'transformers>=4.52,<5'
```
Do not apply this preemptively — only if an actual error surfaces.

---

## VRAM Budget Reference

| Component | GPU Memory |
|---|---|
| LTX-2.3-22B Transformer (GGUF Q3_K_M) | ~11.0 GB |
| Text Projection / Embeddings Connector | ~2.2 GB |
| VAE decode (tiled 256x32) | ~1.9 GB |
| **PEAK during sampling** | **~13.2 GB (79%)** |
| Text Encoder (Gemma-3 Q4_K_M) | CPU offload (0 GB GPU) |
| Transformer during VAE decode | CPU offload via --lowvram (0 GB GPU) |

**--lowvram is essential.** Without it, the transformer stays in VRAM during VAE decode and
there is insufficient headroom for the 1.9 GB tiled allocation. With it, the sequence is:
encode → offload transformer to CPU → VAE decode → done.

If you see an OOM: (1) confirm `--lowvram` is in launch.sh, (2) confirm tile_size=256,
(3) check no other processes hold VRAM via `rocm-smi`.

---

## Useful One-Liners

```bash
# Check GPU VRAM usage
rocm-smi

# Confirm GPU is visible to PyTorch
source ~/comfy/env.sh && python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.hip)"

# Check all models are present
find ~/comfy/ComfyUI/models/diffusion_models/ltx-2.3 \
     ~/comfy/ComfyUI/models/text_encoders/ltx-2.3 \
     ~/comfy/ComfyUI/models/vae/ltx-2.3 \
     -type f -ls 2>/dev/null

# Confirm --lowvram is in launch.sh
grep lowvram ~/comfy/launch.sh

# Confirm DualCLIPLoaderGGUF sees both text encoder files
curl -s http://127.0.0.1:8188/object_info | python3 -c "
import json,sys
d=json.load(sys.stdin)
opts = d.get('DualCLIPLoaderGGUF',{}).get('input',{}).get('required',{})
print('clip_name1:', opts.get('clip_name1',[[]])[0])
print('clip_name2:', opts.get('clip_name2',[[]])[0])
"

# Check transformers version
source ~/comfy/env.sh && python -c "import transformers; print(transformers.__version__)"

# Check bitsandbytes is absent
source ~/comfy/env.sh && python -c "import bitsandbytes" 2>&1 | head -3

# Kill any hanging ComfyUI process
pkill -f "python.*main.py"
```
