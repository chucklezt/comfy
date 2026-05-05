# Claude-comfy.md — Session Handoff Notes
**Last updated:** 2026-05-04 (end of session 3)
**Purpose:** Resume point for ComfyUI / Wan 2.2 I2V setup on RTX 3090

---

## Project Goal

Animate still photos of relatives using **Wan 2.2 I2V 14B GGUF** in ComfyUI on
the RTX 3090 (24 GB) in chuckai. No audio. Quality > speed for primary runs;
Lightning 4-step LoRAs available for fast iteration.

Original model candidates considered: Wan 2.1 VACE 1.3B, Wan 2.2 I2V 14B,
LTX 2.3. Picked **Wan 2.2 I2V 14B (GGUF Q5_K_M)** for best identity preservation
on faces. Wan 2.1 VACE 1.3B kept as a possible fast-iteration option later.
LTX 2.3 set aside — its strengths (speed, audio coupling) don't match this use case.

---

## Session 1 — Audit + Full ROCm → CUDA Rebuild (DONE)

### 1. Audit (read-only)
Confirmed the machine had ComfyUI installed at `~/comfy/ComfyUI/` but the entire
software stack was built for an **AMD RX 6800 XT (ROCm 6.3)**. The RTX 3090 was
physically present and detected by `nvidia-smi`, but `torch.cuda.is_available()`
returned False because the venv contained `torch 2.9.1+rocm6.3`.

### 2. Full ROCm → CUDA rebuild (Phases 0–9)

**Phase 0 — Backup**
- `~/comfy/env.sh.rocm-backup` — original AMD env script
- `~/comfy/launch.sh.rocm-backup` — original AMD launch script
- `~/comfy/venv-rocm-pipfreeze.txt` — full pip freeze of old ROCm venv

**Phase 1 — Wiped** `~/comfy/venv` (the ROCm venv)

**Phase 2 — Created** fresh `python3.10 -m venv ~/comfy/venv`

**Phase 3 — Installed** CUDA PyTorch:
```
torch==2.6.0+cu124  torchvision==0.21.0+cu124  torchaudio==2.6.0+cu124
--index-url https://download.pytorch.org/whl/cu124
```
Also installs as dependencies: triton 3.2.0, nvidia-cudnn-cu12 9.1.0, etc.

**Phase 4 — Installed** ComfyUI's `requirements.txt` plus `accelerate` and `gguf`
(not in ComfyUI's own requirements; needed by GGUF custom node).

**Phase 5 — Updated** ComfyUI from commit `076639fe` (2026-03-30) to
`2806163f` (2026-05-04) — was 141 commits behind. Notable additions in those
commits: Wan 2.2 blueprint, SAM3 nodes, CogVideo, SUPIR, frame interpolation,
LTX-2.3 I2V/T2V blueprints.

**Phase 6 — Cloned and installed** four custom nodes:

| Node | Commit | Notes |
|---|---|---|
| ComfyUI-Manager (ltdrdata) | `8d5c1203` | For node management |
| ComfyUI-WanVideoWrapper (kijai) | `df8f3e4` | Main Wan support |
| ComfyUI-KJNodes (kijai) | `cd5ad80` | Utility nodes for Wan pipeline |
| ComfyUI-VideoHelperSuite (Kosinkadink) | `2984ec4` | MP4 output |

WanVideoWrapper also pulled in: `diffusers 0.38.0`, `peft`, `opencv-python`,
`safetensors 0.8.0rc0` (upgraded from 0.7.0).

**Phase 7 — Rewrote** `~/comfy/env.sh` for CUDA (was full of ROCm/HIP vars).
New version: activates venv, sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
disables HF telemetry, prints torch version on source.

**Phase 8 — Rewrote** `~/comfy/launch.sh` for CUDA. Key changes from old version:
- Removed `--lowvram` (unnecessary on 24 GB)
- Removed ROCm-specific cross-attention workarounds
- Changed `--reserve-vram 2.5` → `--reserve-vram 1.0`
- Removed ROCm diagnostics pre-flight
- Kept `--use-pytorch-cross-attention` (safe CUDA default until sage-attention is added)

**Phase 9 — Smoke test PASSED.**
- ComfyUI v0.20.1 came up in 8 seconds
- `/system_stats` confirmed: `device_type: cuda`, GPU: `NVIDIA GeForce RTX 3090`,
  `vram_total: 23.6 GiB`, `pytorch_version: 2.6.0+cu124`
- No ROCm references in startup log
- All five required custom nodes loaded clean

---

## Session 2 — (2026-05-04, same day as session 1)

No additional system changes. Session 1 rebuild completed and verified.
Stopped before starting model downloads — clean breakpoint with no in-flight state.

---

## Session 3 — Model Investigation + Downloads + Workflow JSON (DONE)

### Step 1 — VideoForge File Investigation (DONE)

Investigated `~/videoforge/models/wan21-vace-1.3b/` before downloading anything.
Ran read-only inspection (inventory, sha256 partial, Python torch.load analysis).

**Findings:**

| File | Size | Classification | Notes |
|---|---|---|---|
| `Wan2.1_VAE.pth` | 0.47 GiB, float32 | REUSABLE_AS_IS | Keys are encoder.*/decoder.*/conv2.* — WanVideoWrapper auto-prepends `model.` prefix and accepts .pth |
| `models_t5_umt5-xxl-enc-bf16.pth` | 10.58 GiB, bfloat16 | REUSABLE_AS_IS | Keys are token_embedding.weight and blocks.* — correct internal format for WanVideoWrapper's LoadWanVideoT5TextEncoder |
| `diffusion_pytorch_model.safetensors` | 6.66 GiB, float32 | NOT_REUSABLE | Wan 2.1 VACE 1.3B model (2.154B params, blocks.* + vace_blocks.*); different architecture from Wan 2.2 I2V |

**Critical finding — fp8 scaled text encoder would have failed on load:**
WanVideoWrapper's `LoadWanVideoT5TextEncoder` raises `ValueError("fp8 scaled is not supported by this node")` when the `scaled_fp8` key is present. The originally planned download `umt5_xxl_fp8_e4m3fn_scaled.safetensors` from Comfy-Org/Wan_2.1_ComfyUI_repackaged contains this key and would have been rejected. This investigation avoided a 6.7 GB wasted download and a confusing first-run failure.

**Key clarification about loader ecosystems (applies to Step 2 decisions):**

ComfyUI has two parallel loader ecosystems for Wan:

1. **Native ComfyUI loaders** (`Load CLIP`, `Load VAE`, `Load Diffusion Model` / `Unet Loader (GGUF)`) — accept `umt5_xxl_fp8_e4m3fn_scaled.safetensors`. The official Comfy-Org Wan 2.2 templates use this path.
2. **WanVideoWrapper loaders** (`Load WanVideo T5 TextEncoder`, `WanVideo VAE Loader`, `WanVideo Model Loader`) — reject fp8 scaled, require bf16. The VideoForge bf16 .pth files work here.

Because GGUF transformers run through the native pipeline (`Unet Loader (GGUF)` plugs into native `Load CLIP` and `Load VAE`), our path uses native loaders. The fp8-scaled rejection is irrelevant to us.

### Step 2 — Download Decision (DONE)

**Decision: keep VideoForge and ComfyUI cleanly separated. Download everything fresh into `~/comfy/ComfyUI/models/`.** 4 TB NVMe with plenty of space; clean separation is simpler and avoids VideoForge contamination risk (the VACE 1.3B diffusion file sitting in the same VideoForge directory would need selective exposure management).

**No `extra_model_paths.yaml` created.** ComfyUI's own models directory is the source of truth.

**All five model files queued for download in a single `tmux` session (`wan22-dl`):**

```bash
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

Total: ~28 GB. Downloads were in progress at session end (~13 MB/s, ~35-40 min estimated remaining from session start).

**To check download status:**
```bash
tmux attach -t wan22-dl
# Ctrl-b d to detach
# Or just poll:
ls -lah ~/comfy/ComfyUI/models/diffusion_models/ && du -sh ~/comfy/ComfyUI/models/*/
```

### Step 3 — Workflow JSON Built (DONE)

**`wan22_i2v_portrait_animation.json`** was built and validated in-session. It is saved in the chat outputs and can be fetched from there, or regenerated in chat if needed.

**Based on the official Comfy-Org `video_wan2_2_14B_i2v.json` template.** Changes from the canonical template:

| What changed | Why |
|---|---|
| `UNETLoader` → `UnetLoaderGGUF` (city96) on both transformer nodes | Our transformers are GGUF, not safetensors |
| Filenames point at our actual downloaded files | Avoids dropdown surprises on first load |
| Default resolution 640×640 → 832×480 | Handoff plan baseline |
| Frames 81 (unchanged) | 5-second clip at 16fps |
| `Enable 4steps LoRA?` default: `true` → `false` | Validate baseline before using Lightning |
| LoadImage filename cleared | Original had a demo filename that would error immediately |
| Negative prompt rewritten in English, targeting portrait failure modes | Original used Chinese; new version suppresses talking, lip movement, identity drift, face morphing |
| Markdown notes rewritten to document the GGUF path | Documentation accuracy |

**Loader node → filename mapping (for manual verification on canvas):**

| Node title | Expected filename |
|---|---|
| Load High-Noise Transformer (GGUF) | `wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf` |
| Load Low-Noise Transformer (GGUF) | `wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf` |
| CLIPLoader | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| VAELoader | `wan_2.1_vae.safetensors` |
| High-Noise Lightning LoRA | `Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors` |
| Low-Noise Lightning LoRA | `Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors` |

**Sampler configuration:**
- Original mode (Lightning OFF): 20 steps, CFG 3.5, euler/simple, split at step 10
- Turbo mode (Lightning ON): 4 steps, CFG 1.0, split at step 2
- Both modes use the same MoE two-stage structure: high-noise expert runs steps 0→split, low-noise expert runs split→end

---

## Current State (verified end of session 3)

### Software Stack
| Component | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12 (`~/comfy/venv/`) |
| torch | 2.6.0+cu124 |
| CUDA runtime | 12.4 |
| cuDNN | 9.1.0 |
| triton | 3.2.0 |
| ComfyUI | v0.20.1 (`2806163f`, 2026-05-04, at HEAD) |
| Driver | 595.58.03 (supports CUDA 12.x / advertises 13.2) |

### GPU
- NVIDIA GeForce RTX 3090, 24 GiB VRAM (23.6 GiB usable)
- Idle state: ~23.3 GiB free
- Allocator mode: `cudaMallocAsync`

### Key Files
| File | Purpose |
|---|---|
| `~/comfy/env.sh` | CUDA env — `source` before any command |
| `~/comfy/launch.sh` | Launch ComfyUI |
| `~/comfy/env.sh.rocm-backup` | Old AMD env (kept for reference) |
| `~/comfy/launch.sh.rocm-backup` | Old AMD launch (kept for reference) |
| `~/comfy/venv-rocm-pipfreeze.txt` | Old ROCm package list |

### Custom Nodes
| Node | Commit | Status |
|---|---|---|
| ComfyUI-GGUF (city96) | `6ea2651` (2026-01-12) | OK |
| ComfyUI-Manager | `8d5c1203` (2026-05-01) | OK |
| ComfyUI-WanVideoWrapper (kijai) | `df8f3e4` (2026-02-22) | OK |
| ComfyUI-KJNodes (kijai) | `cd5ad80` (2026-05-03) | OK |
| ComfyUI-VideoHelperSuite | `2984ec4` (2026-04-06) | OK |
| ID-LoRA-LTX2.3-ComfyUI | `9943746` (2026-03-25) | IMPORT FAIL — `ltx_core` not installed (irrelevant to Wan) |

### Known Cosmetic Notice
`ComfyUI-GGUF: Partial torch compile only, consider updating pytorch` — informational only, GGUF loading and inference work fine on torch 2.6. Not addressing; bumping torch to 2.7+ would risk wheel-ecosystem incompatibility (sageattention, flash-attn) for no functional benefit here.

### Model Inventory (ComfyUI's models/ dir)
LTX-2.3 assets plus Wan 2.2 downloads in progress (see Step 2 above).

**LTX-2.3 (pre-existing, verified):**
- `models/diffusion_models/ltx-2.3/ltx-2.3-22b-dev-Q3_K_M.gguf` (11 GB)
- `models/text_encoders/ltx-2.3/google_gemma-3-12b-it-Q4_K_M.gguf` (6.8 GB)
- `models/text_encoders/ltx-2.3/ltx-2.3_text_projection_bf16.safetensors` (2.2 GB)
- `models/vae/ltx-2.3/LTX23_video_vae_bf16.safetensors` (1.4 GB)

**Wan 2.2 (downloading, session 3 end state):**
- HighNoise GGUF: in progress at ~13 MB/s when session ended
- LowNoise GGUF, UMT5-XXL, VAE, both LoRAs: queued in tmux session `wan22-dl`

### Disk
- 3.0 TB free on partition holding `~/comfy` (4 TB NVMe total)

### VideoForge Assets (unchanged, not touched)
```
~/videoforge/models/wan21-vace-1.3b/
├── Wan2.1_VAE.pth                      (0.47 GiB, float32) — REUSABLE for WanVideoWrapper
├── models_t5_umt5-xxl-enc-bf16.pth     (10.58 GiB, bfloat16) — REUSABLE for WanVideoWrapper
└── diffusion_pytorch_model.safetensors  (6.66 GiB, float32) — Wan 2.1 VACE 1.3B, NOT for Wan 2.2
```
These are deliberately not symlinked or used — kept cleanly separated from ComfyUI.

---

## NEXT SESSION — Resume Plan

### Where to start

First, **verify downloads completed:**
```bash
ls -lah ~/comfy/ComfyUI/models/diffusion_models/wan2.2-i2v-A14B-*Noise-Q5_K_M.gguf
ls -lah ~/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
ls -lah ~/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors
ls -lah ~/comfy/ComfyUI/models/loras/Wan2.2-Lightning_I2V-A14B-4steps_*.safetensors
```

All six files must be present with non-zero sizes before proceeding.

If downloads are still running, check: `tmux attach -t wan22-dl`

### Step 4 — Load Workflow + First Inference

1. **Copy the workflow JSON onto chuckai:**
   ```bash
   # From your local machine after downloading from chat:
   scp wan22_i2v_portrait_animation.json chuck@192.168.1.59:~/comfy/ComfyUI/user/default/workflows/
   # Or paste it into Claude Code on chuckai directly
   ```

2. **Launch ComfyUI:**
   ```bash
   ~/comfy/launch.sh
   ```
   Browse to `http://192.168.1.59:8188`

3. **Load the workflow:** drag `wan22_i2v_portrait_animation.json` onto the canvas, or Workflow → Open.

4. **Verify loader node filenames** match the table in the Session 3 section above. If any dropdown is red, click Refresh (or restart ComfyUI).

5. **Upload a portrait photo** to the Load Image node.

6. **Queue the baseline run** (Lightning OFF by default):
   - 832×480, 81 frames, 20 steps, CFG 3.5
   - Expected ~3-5 minutes on the 3090
   - Look for: identity preservation, subtle natural motion (breathing, blink, hair), no talking artifacts

7. **If baseline looks good, try Lightning:**
   - Open the subgraph, toggle "Enable 4steps LoRA?" to `true`
   - Same image, same prompt — ~30-60 seconds
   - Use Lightning for scouting, 20-step for finals

### Common first-run failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Unknown node type: UnetLoaderGGUF` | ComfyUI-GGUF didn't import | Check `~/comfy/ComfyUI/custom_nodes/ComfyUI-GGUF/` exists; restart |
| Red filename in loader dropdown | File missing or wrong directory | Re-run the prerequisites check above |
| OOM at sampling | Frames or resolution too large | Drop to 49 frames or 480×480 |
| `Tensor size mismatch` | LoRA version mismatch | Confirm both LoRAs are I2V, not T2V |
| Pure noise output | ModelSamplingSD3 shift wrong | Both ModelSamplingSD3 nodes should be set to 5.0 |
| Identity drifts at ~frame 40 | Temporal coherence limit | Drop to 49 frames; or lower CFG from 3.5 to 3.0 |

### Step 5 — Optional Optimizations (after baseline + Lightning both work)

1. Install `sageattention` for ~25-30% attention speedup on Ampere:
   - Triton 3.2.0 is already in the venv, so the wheel build path works
   - Switch `--use-pytorch-cross-attention` → `--use-sage-attention` in `~/comfy/launch.sh`

2. Try Q4_K_M instead of Q5_K_M if you want to push toward 720p resolution:
   - ~2 GB VRAM savings per expert; quality hit is small at 14B scale

---

## VRAM Budget Reference (RTX 3090, 24 GB)

Wan 2.2 I2V 14B with Q5_K_M GGUF + MoE 2-stage sampling:

| Component | GPU Memory |
|---|---|
| Wan 2.2 14B transformer (GGUF Q5_K_M, one expert at a time) | ~10–12 GB |
| UMT5-XXL text encoder (fp8 scaled, CPU offloadable) | ~6 GB |
| Wan VAE decode | ~2–3 GB |
| Latents + KV cache (81 frames @ 832×480) | ~2–3 GB |
| **Expected peak** | **~16–20 GB — fits on 24 GB with `--reserve-vram 1.0`** |

No `--lowvram` needed. Two experts load sequentially, not simultaneously — ComfyUI offloads the inactive expert during the MoE switch.

---

## Useful Commands

```bash
# Start ComfyUI
~/comfy/launch.sh

# Activate the venv manually (e.g. for downloads)
source ~/comfy/env.sh

# Check download progress
tmux attach -t wan22-dl    # Ctrl-b d to detach
ls -lah ~/comfy/ComfyUI/models/diffusion_models/
du -sh ~/comfy/ComfyUI/models/*/

# Watch VRAM during generation
watch -n 1 nvidia-smi

# Check what's listening on port 8188
ss -tlnp 2>/dev/null | grep 8188

# Tail ComfyUI logs (if launched via nohup)
tail -f /tmp/comfy-smoketest.log
```

---

## Open Questions / Deferred Items

| Item | Status | Notes |
|---|---|---|
| `ID-LoRA-LTX2.3-ComfyUI` import failure (`ltx_core` not installed) | Deferred | Doesn't affect Wan 2.2; ComfyUI skips gracefully; leave in place |
| sageattention install | Deferred to Step 5 | Triton 3.2.0 is available; do after baseline inference works |
| VideoForge Wan 2.1 VACE 1.3B fast iteration | Deferred | Original plan was to explore this after Wan 2.2 baseline. WanVideoWrapper accepts the existing .pth files; would need a separate workflow |
