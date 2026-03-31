# CLAUDE.md — LTX-2.3 Workspace Validation (chuckai)

## Context

You are working on **chuckai** — a local Ubuntu 22.04 server with an AMD Radeon RX 6800 XT
(16 GB VRAM, RDNA2 / gfx1030) running ROCm 6.3. The workspace is at `~/comfy/`.

This session goal is **full end-to-end validation**: confirm the environment is clean, fix
known issues, download models, and produce a first successful inference frame in ComfyUI.

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

## Known Issues to Resolve (in order)

### 1. Models not yet downloaded
All four model directories exist but are empty. Models must be present before ComfyUI can run.

Target paths:
- `~/comfy/ComfyUI/models/diffusion_models/ltx-2.3/` — LTX-2.3-22B transformer (GGUF Q3_K_M, ~10.1 GB)
- `~/comfy/ComfyUI/models/text_encoders/ltx-2.3/` — Gemma-3-12B-IT text encoder (GGUF Q4_K_M, ~7 GB)
- `~/comfy/ComfyUI/models/vae/ltx-2.3/` — LTX VAE checkpoint
- `~/comfy/ComfyUI/models/loras/ltx-2.3/` — ID-LoRA weights (download only if validating LoRA)

**First, review the download script to get the exact huggingface-cli commands:**
```bash
cat ~/comfy/download_models.sh
```
Then execute the commands it prints. Use `huggingface-cli download` with `--local-dir` pointing
to the correct staging folder above. Do not move or rename model files after downloading.

### 2. --reserve-vram flag unconfirmed

`launch.sh` passes `--reserve-vram 2.5` to ComfyUI's `main.py`. This flag may not exist in
the installed ComfyUI version.

**Check:**
```bash
source ~/comfy/env.sh
python ~/comfy/ComfyUI/main.py --help | grep -i reserve
```

- If the flag exists: proceed as-is.
- If not found: edit `launch.sh` and remove `--reserve-vram 2.5`. VRAM headroom will be
  managed by tiled VAE settings (tile_size=512, temporal_size=32) instead.

---

## Validation Sequence

Work through these steps in order. Do not skip ahead. Report status at each step before
proceeding.

### Step 1 — Environment check
```bash
source ~/comfy/env.sh
python ~/comfy/diagnostics.py
```
All items marked ERROR must pass. WARNING items on empty model directories are expected at
this stage. Fix any ERROR before continuing.

### Step 2 — Verify --reserve-vram
Apply the check from Known Issue #2. Update launch.sh if needed. Show me the relevant line
in launch.sh before and after any change.

### Step 3 — Download models
Apply the check from Known Issue #1. Download transformer and text encoder GGUFs
first (VAE second, LoRA last). Confirm each file lands in the correct staging directory and
is non-zero size. Do not proceed to Step 4 until transformer + text encoder + VAE are present.

### Step 4 — Re-run diagnostics (full)
```bash
source ~/comfy/env.sh
python ~/comfy/diagnostics.py
```
At this point model-presence WARNINGs should clear. All ERRORs must still pass.

### Step 5 — Launch ComfyUI
```bash
~/comfy/launch.sh
```
Confirm it starts without errors and the web UI is reachable at `http://localhost:8188`.
Report the last 20 lines of startup output.

### Step 6 — First inference smoke test
In ComfyUI, load the default LTX workflow (or the reference config at
`~/comfy/comfyui_rdna2.yaml`). Use these safe parameters for the first run:

| Parameter | Value |
|---|---|
| Resolution | 512 × 512 |
| Frames | 17 (minimum for LTX temporal model) |
| Steps | 20 |
| CFG | 3.5 |
| Prompt | "a still camera shot of a red ball on a wooden table" |

Dimensions must be multiples of 32. 512×512 is safe. Do not use arbitrary resolutions.

If inference completes and produces a video file: validation is complete.
If it OOMs or errors: capture the full traceback and report it before attempting any fix.

---

## Constraints & Rules

- **Never install bitsandbytes.** It is CUDA-only and will segfault on RDNA2. The import
  hook in `block_bitsandbytes.py` will catch accidental re-introduction, but don't let it
  get that far. If any `pip install` command pulls it in as a transitive dependency, uninstall
  it immediately: `pip uninstall bitsandbytes -y`.

- **Never use adamw8bit or any bitsandbytes optimizer.** Use `adafactor` for any training
  config work.

- **Always source env.sh first.** `HSA_OVERRIDE_GFX_VERSION=10.3.0` must be set or PyTorch
  will not see the GPU correctly.

- **Video dimensions must be multiples of 32.** Non-aligned values cause HIP memory access
  faults. Safe presets: 512×512, 768×512, 1024×576.

- **Text encoder must be CPU-offloaded.** The Gemma-3 GGUF (~4 GB) must move to system RAM
  immediately after encoding. This is what makes the 16 GB budget work. Do not change this
  behavior.

- **VAE decode must use tiled mode.** tile_size=512, temporal_size=32. Do not switch to
  full VAE decode.

- **Do not upgrade PyTorch.** `2.9.1+rocm6.3` is the correct version. Do not run
  `pip install --upgrade torch` or similar.

- **Show diffs before editing files.** Before modifying launch.sh, env.sh, or any custom
  node file, show the current content of the relevant section and explain the change.

---

## Watch Items (not broken, monitor as we go)

### transformers version
Currently at `5.4.0`. The ID-LoRA-LTX2.3-ComfyUI node was authored against `transformers < 5.0`
and the README flags a potential incompatibility. No conflict has been observed yet. Monitor for
errors during LoRA loading or prompt encoding. If you see an `AttributeError` or `ImportError`
traceable to `transformers`, the fix is:
```bash
source ~/comfy/env.sh
pip install 'transformers>=4.52,<5'
```
Do not apply this preemptively — only if an actual error surfaces.

---

## VRAM Budget Reference

```
LTX-2.3-22B Transformer (GGUF Q3_K_M)   ~10.1 GB  GPU
LoRA patch overhead                        ~0.5 GB  GPU
VAE decode (tiled 512×32)                  ~2.5 GB  GPU
OS/driver reserve                           2.5 GB  GPU
────────────────────────────────────────────────────────
TOTAL PEAK                                ~15.6 GB

Text Encoder (Gemma-3 Q4_K_M)           → CPU RAM  (offloaded after encoding)
```

If you see an OOM that doesn't fit this budget, check: (1) text encoder is actually offloaded,
(2) VAE is in tiled mode, (3) no other processes are holding VRAM (`rocm-smi`).

---

## Useful One-Liners

```bash
# Check GPU VRAM usage
rocm-smi

# Confirm GPU is visible to PyTorch
source ~/comfy/env.sh && python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.hip)"

# Check what's in model directories
find ~/comfy/ComfyUI/models/diffusion_models/ltx-2.3 ~/comfy/ComfyUI/models/text_encoders/ltx-2.3 ~/comfy/ComfyUI/models/vae/ltx-2.3 -type f -ls 2>/dev/null

# Check transformers version
source ~/comfy/env.sh && python -c "import transformers; print(transformers.__version__)"

# Check bitsandbytes is absent
source ~/comfy/env.sh && python -c "import bitsandbytes" 2>&1 | head -3

# Kill any hanging ComfyUI process
pkill -f "python.*main.py"
```
