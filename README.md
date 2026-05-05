# ComfyUI Video Generation Workspace

Local AI video generation on **chuckai** — a personal Ubuntu 22.04 server running
ComfyUI with GGUF-quantized models. This project has gone through two distinct hardware
phases and now supports two active pipelines.

---

## Table of Contents

- [Project History at a Glance](#project-history-at-a-glance)
- [Current State (Phase 2 — RTX 3090 / CUDA)](#current-state-phase-2--rtx-3090--cuda)
- [Phase 1 — RX 6800 XT / ROCm (Historical Reference)](#phase-1--rx-6800-xt--rocm-historical-reference)
  - [Hardware & System](#hardware--system)
  - [Component Versions (ROCm Era)](#component-versions-rocm-era)
  - [Directory Structure](#directory-structure)
  - [What Was Built — Phase 1](#what-was-built--phase-1)
  - [VRAM Budget — RX 6800 XT](#vram-budget--rx-6800-xt)
  - [Known Caveats — ROCm](#known-caveats--rocm)
  - [All Installed Packages — ROCm Venv](#all-installed-packages--rocm-venv)
- [Phase 2 — RTX 3090 / CUDA Rebuild](#phase-2--rtx-3090--cuda-rebuild)
  - [Why the Rebuild](#why-the-rebuild)
  - [What Was Done — Phase 2](#what-was-done--phase-2)
  - [Component Versions (CUDA Era)](#component-versions-cuda-era)
  - [Custom Nodes — Current](#custom-nodes--current)
  - [LTX-2.3 Pipeline (Re-validated on CUDA)](#ltx-23-pipeline-re-validated-on-cuda)
  - [VRAM Budget — RTX 3090](#vram-budget--rtx-3090)
- [Phase 3 — Wan 2.2 I2V Portrait Animation](#phase-3--wan-22-i2v-portrait-animation)
  - [Model Inventory — Wan 2.2](#model-inventory--wan-22)
  - [Workflow JSON](#workflow-json)
  - [Wan 2.2 Pipeline Reference](#wan-22-pipeline-reference)
  - [VRAM Budget — Wan 2.2](#vram-budget--wan-22)
- [Next Steps](#next-steps)
- [Quick Start](#quick-start)
- [Useful Commands](#useful-commands)

---

## Project History at a Glance

| Phase | GPU | Stack | Pipeline | Status |
|---|---|---|---|---|
| 1 | AMD RX 6800 XT (16 GB) | ROCm 6.3 / `torch 2.9.1+rocm6.3` | LTX-2.3 22B text-to-video | ✅ Validated end-to-end |
| 2 | NVIDIA RTX 3090 (24 GB) | CUDA 12.4 / `torch 2.6.0+cu124` | LTX-2.3 (carried over) | ✅ Stack rebuilt and verified |
| 3 | NVIDIA RTX 3090 (24 GB) | CUDA 12.4 / `torch 2.6.0+cu124` | Wan 2.2 I2V 14B portrait animation | ✅ Validated end-to-end (2026-05-05) |

The RX 6800 XT was physically present in chuckai alongside the RTX 3090. The ROCm venv
was the default until a full investigation in May 2026 revealed that `torch.cuda.is_available()`
was returning False on the 3090 — the existing venv had been built for AMD and the NVIDIA
card was invisible to PyTorch. The rebuild corrected this.

---

## Current State (Phase 2 — RTX 3090 / CUDA)

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 |
| VRAM | 24 GiB (23.6 GiB usable) |
| CUDA | 12.4 |
| Driver | 595.58.03 |
| cuDNN | 9.1.0 |
| PyTorch | 2.6.0+cu124 |
| triton | 3.2.0 |
| Python | 3.10.12 |
| ComfyUI | v0.20.1 (`2806163f`, 2026-05-04) |
| env script | `~/comfy/env.sh` — source before every session |
| Launch | `~/comfy/launch.sh` |

---

## Phase 1 — RX 6800 XT / ROCm (Historical Reference)

This section preserves the complete original documentation from the ROCm era.
The RX 6800 XT and its venv are no longer the active configuration, but the
work done here — especially the VRAM budget analysis, bitsandbytes safety block,
and LTX-specific pipeline discovery — directly informed Phase 2 and Phase 3.

### Hardware & System

| Component           | Value                                        |
|---------------------|----------------------------------------------|
| GPU                 | AMD Radeon RX 6800 XT (16 GB VRAM)           |
| GPU Architecture    | RDNA2 / gfx1030                              |
| CPU                 | Intel Core i7-10700K @ 3.80 GHz              |
| OS                  | Linux 5.15.0-173-generic (Ubuntu)             |
| Python              | 3.10.12                                      |
| ROCm                | 6.3.0-39                                     |
| HIP Runtime         | 6.3.42134-a9a80e791                          |
| ROCk Module         | 6.10.5                                       |

### Component Versions (ROCm Era)

#### Core ML Stack

| Package            | Version             | Notes                                       |
|--------------------|---------------------|---------------------------------------------|
| torch              | 2.9.1+rocm6.3      | ROCm HIP build                              |
| torchvision        | 0.24.1+rocm6.3     |                                             |
| torchaudio         | 2.9.1+rocm6.3      |                                             |
| pytorch-triton-rocm| 3.5.1               | Triton compiler for ROCm                    |
| transformers       | 5.4.0               |                                             |
| safetensors        | 0.7.0               |                                             |
| accelerate         | 1.13.0              |                                             |

#### LTX / LoRA Stack

| Package            | Version             | Notes                                       |
|--------------------|---------------------|---------------------------------------------|
| ltx-core           | 1.0.0               | LTX model core (PyPI)                       |
| ltx-pipelines      | 1.0.0               | LTX inference pipelines (PyPI)              |
| ltx-trainer        | 1.0.0               | Quantization support (source install from ID-LoRA repo) |
| peft               | 0.18.1              | HuggingFace LoRA/adapter engine             |
| optimum-quanto     | 0.2.7               | int8 quantization (HIP-safe, replaces bitsandbytes) |

#### GGUF / Quantization

| Package            | Version             | Notes                                       |
|--------------------|---------------------|---------------------------------------------|
| gguf               | 0.18.0              | GGUF model format reader                    |
| sentencepiece      | 0.2.1               | Tokenizer for Gemma text encoder            |

### Directory Structure

```
comfy/
├── env.sh                          # RDNA2 environment variables (source before use)
├── launch.sh                       # One-command ComfyUI launcher with pre-flight checks
├── diagnostics.py                  # Hardware/software validation script
├── block_bitsandbytes.py           # Python import hook to block bitsandbytes on RDNA2
├── download_models.sh              # Prints huggingface-cli commands for model downloads
├── comfyui_rdna2.yaml              # Reference config for workflow node settings
├── env.sh.rocm-backup              # ← preserved ROCm env (reference only)
├── launch.sh.rocm-backup           # ← preserved ROCm launch (reference only)
├── venv-rocm-pipfreeze.txt         # ← preserved full pip freeze of ROCm venv
│
├── venv/                           # Python 3.10 virtual environment (now CUDA — see Phase 2)
│
├── ComfyUI/                        # ComfyUI application
│   ├── main.py
│   ├── custom_nodes/
│   │   ├── ComfyUI-GGUF/          # GGUF model loader
│   │   └── ID-LoRA-LTX2.3-ComfyUI/
│   └── models/
│       ├── diffusion_models/ltx-2.3/
│       ├── text_encoders/ltx-2.3/
│       ├── vae/ltx-2.3/
│       └── loras/ltx-2.3/
│
└── ID-LoRA/                        # Upstream ID-LoRA repo (cloned for ltx-trainer)
    └── ID-LoRA-2.3/packages/
        ├── ltx-core/
        ├── ltx-pipelines/
        └── ltx-trainer/
```

### What Was Built — Phase 1

#### 1. Core Environment & Driver Config

Created `env.sh` — must be sourced before any session. Sets:

| Variable                        | Value                    | Purpose                                                       |
|---------------------------------|--------------------------|---------------------------------------------------------------|
| `HSA_OVERRIDE_GFX_VERSION`      | `10.3.0`                 | Maps gfx1030 (RDNA2) to closest supported HIP target          |
| `PYTORCH_HIP_ALLOC_CONF`       | `expandable_segments:True` | Reduces VRAM fragmentation OOMs (future-proofing; not yet active on RDNA2 HIP) |
| `PYTORCH_ALLOC_CONF`           | `expandable_segments:True` | Same, renamed var for PyTorch >= 2.8                          |
| `MIOPEN_FIND_MODE`             | `2`                      | NORMAL find mode — prevents multi-GB workspace allocations     |
| `MIOPEN_DISABLE_CACHE`         | `0`                      | Keeps MIOpen cache active but avoids stale-cache mismatches    |
| `COMFYUI_ENABLE_MIOPEN`        | `1`                      | Enables MIOpen backend in ComfyUI custom nodes                 |
| `HF_HUB_DISABLE_TELEMETRY`     | `1`                      | Disables HuggingFace telemetry                                 |
| `DO_NOT_TRACK`                  | `1`                      | General analytics opt-out                                      |
| `BITSANDBYTES_BLOCKED`         | `1`                      | Activates the Python import hook to block bitsandbytes         |

```bash
#!/usr/bin/env bash
# LTX-2.3 / ComfyUI — RDNA2 (gfx1030) Environment for AMD Radeon RX 6800 XT
# (HISTORICAL — replaced by CUDA env.sh in Phase 2)

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export MIOPEN_FIND_MODE=2
export MIOPEN_DISABLE_CACHE=0
export COMFYUI_ENABLE_MIOPEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export BITSANDBYTES_BLOCKED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
echo "✓ LTX-2.3 RDNA2 environment loaded (gfx1030 → HSA 10.3.0)"
```

#### 2. Python Venv & PyTorch for ROCm

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Result: `torch 2.9.1+rocm6.3`, `torchvision 0.24.1+rocm6.3`, `torchaudio 2.9.1+rocm6.3`.

#### 3. ComfyUI Installation

ComfyUI cloned to `ComfyUI/`. Model subdirectories created with `ltx-2.3/` staging folders.

#### 4. Custom Nodes (ROCm era)

Two custom nodes installed:

**ComfyUI-GGUF** — `UnetLoaderGGUF` and `CLIPLoaderGGUF` for loading GGUF-quantized models.

**ID-LoRA-LTX2.3-ComfyUI** — 5 nodes for identity-preserving video generation:
- `IDLoraModelLoader` — one-stage pipeline
- `IDLoraTwoStageModelLoader` — two-stage pipeline with spatial upsampler
- `IDLoraPromptEncoder` — text prompt encoding with Gemma-3
- `IDLoraOneStageSampler` — single-resolution generation
- `IDLoraTwoStageSampler` — 2x spatial upsampling generation

#### 5. LoRA Capability (ID-LoRA-LTX2.3)

`ltx-core` and `ltx-pipelines` installed from PyPI. `ltx-trainer` not on PyPI — installed
from source:

```bash
git clone --depth 1 https://github.com/ID-LoRA/ID-LoRA.git
pip install -e ID-LoRA/ID-LoRA-2.3/packages/ltx-trainer
```

This pulled in `peft 0.18.1` and `optimum-quanto 0.2.7`. **`ltx-trainer` also dragged in
`bitsandbytes 0.49.2`** — immediately uninstalled because it is CUDA-only and will segfault
on RDNA2:

```bash
pip uninstall bitsandbytes -y
```

Verified import chain:
```
ltx_trainer.quantization.quantize_model  ✓
ltx_core.loader.LoraPathStrengthAndSDOps ✓
ltx_pipelines.utils.ModelLedger          ✓
peft 0.18.1                              ✓
optimum-quanto 0.2.7                     ✓
bitsandbytes                             ✗  (blocked/absent — safe)
```

#### 6. bitsandbytes Safety Block

Created `block_bitsandbytes.py` — a Python meta-path import hook. Intercepts
`import bitsandbytes` and raises a clear error instead of allowing a silent segfault.
Activated when `BITSANDBYTES_BLOCKED=1` is set.

```python
class _BitsAndBytesBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    BLOCKED = {"bitsandbytes"}

    def find_module(self, fullname, path=None):
        top = fullname.split(".")[0]
        if top in self.BLOCKED:
            return self

    def load_module(self, fullname):
        raise ImportError(
            f"'{fullname}' is blocked on RDNA2 (gfx1030). "
            "bitsandbytes / adamw8bit are CUDA-only and will segfault on AMD HIP. "
            "Use adafactor instead. To unblock: unset BITSANDBYTES_BLOCKED"
        )
```

#### 7. Diagnostic Script

`diagnostics.py` — validates the full stack. Checks:

| Check | Type | Details |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | Error | Must be `10.3.0` |
| Expandable segments | Error | `PYTORCH_ALLOC_CONF` set |
| `MIOPEN_FIND_MODE` | Warn | Should be `2` |
| PyTorch version | Info | Prints version string |
| HIP version | Error | `torch.version.hip` must not be None |
| `torch.cuda.is_available()` | Error | ROCm HIP device must be visible |
| GPU name & VRAM | Info | Prints device name and total memory |
| Device is RX 6800 XT | Warn | Checks "6800" in device name |
| HIP tensor compute | Error | 256×256 matmul smoke test |
| SDPA available | Error | `scaled_dot_product_attention` importable |
| bitsandbytes not installed | Error | Import must fail |
| Transformer GGUF present | Warn | Checks `diffusion_models/ltx-2.3/` |
| Text Encoder GGUF present | Warn | Checks `text_encoders/ltx-2.3/` |
| VAE present | Warn | Checks `vae/ltx-2.3/` |
| Custom node directories | Warn | Both ComfyUI-GGUF and ID-LoRA present |

#### 8. Launch Script (ROCm era)

```bash
exec python main.py \
    --listen \
    --use-pytorch-cross-attention \
    --reserve-vram 2.5 \
    --lowvram \
    "$@"
```

**`--lowvram` was required on the 16 GB RX 6800 XT.** Without it the transformer stays in
VRAM during VAE decode, leaving only ~1 GB free — insufficient for the 1.9 GB tiled
allocation.

#### 9. Model Download Helper

`download_models.sh` — prints `huggingface-cli download` commands for review before execution.

**Validated model inventory for 16 GB VRAM:**

| Model | File | Size | Source |
|---|---|---|---|
| LTX-2.3-22B Transformer | `ltx-2.3-22b-dev-Q3_K_M.gguf` | 11 GB | `unsloth/LTX-2.3-GGUF` (NOT `Lightricks/LTX-Video-2.3-22B-GGUF` — that repo does not exist) |
| Gemma-3-12B-IT Text Encoder | `google_gemma-3-12b-it-Q4_K_M.gguf` | 6.8 GB | `Kijai/LTX2.3_comfy` |
| LTX Text Projection | `ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | `Kijai/LTX2.3_comfy` |
| LTX VAE | `LTX23_video_vae_bf16.safetensors` | 1.4 GB | `Kijai/LTX2.3_comfy` |
| ID-LoRA weights | *(not yet downloaded)* | ~1.1 GB | `Lightricks/LTX-Video-2.3` |

Full fp16 checkpoint (`ltx-2.3-22b-dev.safetensors`) is 43 GB — incompatible with 16 GB VRAM, not downloaded.

#### 10. Workflow Reference Config

`comfyui_rdna2.yaml` — documents recommended ComfyUI node settings for the RDNA2 configuration.

```yaml
# ComfyUI Workflow Defaults — LTX-2.3 on RDNA2 (RX 6800 XT, 16 GB)
# VALIDATED — end-to-end smoke test passed

transformer:
  node: "UnetLoaderGGUF"
  model: "ltx-2.3/ltx-2.3-22b-dev-Q3_K_M.gguf"

text_encoder:
  node: "DualCLIPLoaderGGUF"          # NOT CLIPLoaderGGUF (single)
  clip_name1: "ltx-2.3/google_gemma-3-12b-it-Q4_K_M.gguf"
  clip_name2: "ltx-2.3/ltx-2.3_text_projection_bf16.safetensors"  # REQUIRED
  type: "ltxv"

conditioning:
  node: "LTXVConditioning"             # NOT CLIPTextEncode

scheduler:
  node: "LTXVScheduler"               # NOT BasicScheduler

sampler:
  node: "SamplerCustomAdvanced"        # NOT KSampler

vae:
  node: "VAEDecodeTiled"
  tile_size: 256                       # 512 causes OOM — do not increase
  temporal_size: 32

dimension_presets:
  - { name: "square",       width: 512,  height: 512  }   # validated baseline
  - { name: "landscape_hd", width: 768,  height: 512  }
  - { name: "portrait_hd",  width: 512,  height: 768  }
  - { name: "wide",         width: 1024, height: 576  }

frame_counts:
  # Formula: 1 + 8N (minimum N=2 → 17 frames)
  - 17    # minimum / baseline
  - 25
  - 33
  - 49
```

### VRAM Budget — RX 6800 XT

16 GB total. Aggressive offloading required.

| Component | GPU Memory |
|---|---|
| LTX-2.3-22B Transformer (GGUF Q3_K_M) | ~11.0 GB |
| Text Projection / Embeddings Connector | ~2.2 GB |
| VAE decode (tiled 256×32) | ~1.9 GB |
| **Peak during sampling** | **~13.2 GB (79%)** |
| Text Encoder (Gemma-3 Q4_K_M) | CPU offload (0 GB GPU) |
| Transformer during VAE decode | CPU offload via `--lowvram` (0 GB GPU) |

`--lowvram` was the critical flag. The sequence: encode → offload transformer to CPU → VAE
decode. Without it, OOM.

### Known Caveats — ROCm

1. **`expandable_segments` not yet active on RDNA2/HIP.** Set for forward compatibility;
   PyTorch 2.9.1+rocm6.3 prints a harmless warning. Auto-resolves if ROCm HIP adds support.

2. **`bitsandbytes` is a transitive dependency of `ltx-trainer`.** Uninstall immediately
   after any re-install: `pip uninstall bitsandbytes -y`.

3. **`transformers` 5.4.0.** ID-LoRA README flags potential 5.x incompatibility. No issues
   observed during text-to-video inference. If `AttributeError` appears during LoRA loading:
   `pip install 'transformers>=4.52,<5'`.

4. **Standard ComfyUI `KSampler` pipeline does not work with LTX-2.3.** LTX-2.3 is an
   audio-video model; `CLIPLoaderGGUF` (single) + `KSampler` produces a tensor dimension
   mismatch. Always use `DualCLIPLoaderGGUF` + `LTXVConditioning` + `LTXVScheduler` +
   `SamplerCustomAdvanced`.

5. **Text projection file is required.** `ltx-2.3_text_projection_bf16.safetensors` (2.2 GB)
   must sit alongside the Gemma GGUF in `models/text_encoders/ltx-2.3/`. Without it the
   embedding shape is wrong.

6. **Video dimensions must be multiples of 32.** Non-aligned values cause HIP memory access
   faults. Safe presets: 512×512, 768×512, 1024×576.

7. **Frame count formula: 1 + 8N.** Valid: 17, 25, 33, 49, 65... Minimum is 17 (N=2).

8. **First inference is slow.** MIOpen compiles GPU kernels for gfx1030 on first use.
   Expect 10-15 minutes on the very first generation; subsequent runs start in seconds.

9. **ID-LoRA pipeline requires the full 43 GB safetensors checkpoint.** Not usable on 16 GB
   VRAM as-is. Deferred.

### All Installed Packages — ROCm Venv

Full `pip list` preserved in `~/comfy/venv-rocm-pipfreeze.txt`. Highlights:

```
torch                  2.9.1+rocm6.3   ROCm wheel
torchvision            0.24.1+rocm6.3  ROCm wheel
torchaudio             2.9.1+rocm6.3   ROCm wheel
pytorch-triton-rocm    3.5.1
transformers           5.4.0
safetensors            0.7.0
accelerate             1.13.0
ltx-core               1.0.0
ltx-pipelines          1.0.0
ltx-trainer            1.0.0           Source (ID-LoRA repo)
peft                   0.18.1
optimum-quanto         0.2.7
gguf                   0.18.0
sentencepiece          0.2.1
diffusers              0.38.0
opencv-python          4.13.0.92
numpy                  2.2.6
huggingface_hub        1.8.0
wandb                  0.25.1
insightface            0.7.3
```

---

## Phase 2 — RTX 3090 / CUDA Rebuild

### Why the Rebuild

The RTX 3090 (24 GB) was physically installed in chuckai alongside the RX 6800 XT. An audit
in May 2026 revealed that `torch.cuda.is_available()` returned `False` on the 3090 — the
existing venv (`torch 2.9.1+rocm6.3`) was built exclusively for AMD HIP and had no awareness
of the NVIDIA device. `nvidia-smi` confirmed the 3090 was present and driver-ready; only
PyTorch was blind to it.

The 3090's 24 GB VRAM vs the 6800 XT's 16 GB makes it the better choice for all video
generation work: larger models fit without aggressive offloading, higher resolutions become
viable, and CUDA's ecosystem (sageattention, flash-attn, xformers) is more mature than ROCm's.

### What Was Done — Phase 2

**Phase 0 — Backups taken before any changes:**
```
~/comfy/env.sh.rocm-backup              # original AMD env
~/comfy/launch.sh.rocm-backup           # original AMD launch
~/comfy/venv-rocm-pipfreeze.txt         # full pip freeze of ROCm venv
```

**Phase 1 — Wiped** `~/comfy/venv` (the ROCm venv).

**Phase 2 — Created** fresh Python 3.10 venv:
```bash
python3.10 -m venv ~/comfy/venv
```

**Phase 3 — Installed** CUDA PyTorch:
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124
```
Pulled in as dependencies: `triton 3.2.0`, `nvidia-cudnn-cu12 9.1.0`.

**Phase 4 — Installed** ComfyUI requirements plus `accelerate` and `gguf`:
```bash
pip install -r ~/comfy/ComfyUI/requirements.txt
pip install accelerate gguf
```

**Phase 5 — Updated** ComfyUI itself from commit `076639fe` (2026-03-30) to `2806163f`
(2026-05-04) — was 141 commits behind HEAD. Notable additions in those commits: Wan 2.2
blueprint, SAM3 nodes, CogVideo, SUPIR, frame interpolation, LTX-2.3 I2V/T2V blueprints.

**Phase 6 — Cloned and installed** additional custom nodes needed for Wan 2.2 and
ongoing work:

| Node | Commit | Purpose |
|---|---|---|
| ComfyUI-Manager (ltdrdata) | `8d5c1203` | Node management |
| ComfyUI-WanVideoWrapper (kijai) | `df8f3e4` | Wan 2.2 support |
| ComfyUI-KJNodes (kijai) | `cd5ad80` | Utility nodes for Wan pipeline |
| ComfyUI-VideoHelperSuite (Kosinkadink) | `2984ec4` | MP4 output |

WanVideoWrapper pulled in: `diffusers 0.38.0`, `peft`, `opencv-python`,
`safetensors 0.8.0rc0`.

**Phase 7 — Rewrote** `~/comfy/env.sh` for CUDA:

```bash
#!/usr/bin/env bash
# ComfyUI — CUDA (RTX 3090) Environment
# source this before launching ComfyUI

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

python -c "import torch; print('✓ torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"
```

Key changes from ROCm version: removed `HSA_OVERRIDE_GFX_VERSION`, `MIOPEN_*`, and
`COMFYUI_ENABLE_MIOPEN`; replaced `PYTORCH_HIP_ALLOC_CONF` with
`PYTORCH_CUDA_ALLOC_CONF`.

**Phase 8 — Rewrote** `~/comfy/launch.sh` for CUDA:

```bash
exec python main.py \
    --listen \
    --use-pytorch-cross-attention \
    --reserve-vram 1.0 \
    "$@"
```

Key changes from ROCm version: removed `--lowvram` (unnecessary on 24 GB); changed
`--reserve-vram 2.5` → `1.0` (the 3090 has ample headroom).

**Phase 9 — Smoke test PASSED:**
- ComfyUI v0.20.1 started in 8 seconds
- `/system_stats` confirmed: `device_type: cuda`, GPU `NVIDIA GeForce RTX 3090`,
  `vram_total: 23.6 GiB`, `pytorch_version: 2.6.0+cu124`
- No ROCm references in startup log
- All custom nodes loaded (one exception: `ID-LoRA-LTX2.3-ComfyUI` — see below)

### Component Versions (CUDA Era)

| Package | Version | Notes |
|---|---|---|
| torch | 2.6.0+cu124 | CUDA build |
| torchvision | 0.21.0+cu124 | |
| torchaudio | 2.6.0+cu124 | |
| triton | 3.2.0 | |
| nvidia-cudnn-cu12 | 9.1.0 | |
| transformers | 5.4.0 | Carried over |
| safetensors | 0.8.0rc0 | Upgraded by WanVideoWrapper |
| accelerate | current | |
| diffusers | 0.38.0 | Pulled in by WanVideoWrapper |
| gguf | current | |
| peft | current | |
| opencv-python | current | |

### Custom Nodes — Current

| Node | Commit | Status |
|---|---|---|
| ComfyUI-GGUF (city96) | `6ea2651` (2026-01-12) | ✅ OK |
| ComfyUI-Manager (ltdrdata) | `8d5c1203` (2026-05-01) | ✅ OK |
| ComfyUI-WanVideoWrapper (kijai) | `df8f3e4` (2026-02-22) | ✅ OK |
| ComfyUI-KJNodes (kijai) | `cd5ad80` (2026-05-03) | ✅ OK |
| ComfyUI-VideoHelperSuite (Kosinkadink) | `2984ec4` (2026-04-06) | ✅ OK |
| ID-LoRA-LTX2.3-ComfyUI | `9943746` (2026-03-25) | ⚠️ IMPORT FAIL — `ltx_core` not installed. Does not affect Wan 2.2 or LTX text-to-video. ComfyUI skips gracefully. Fix when LTX ID-LoRA work resumes: `pip install ltx-core ltx-pipelines && pip install -e ~/ID-LoRA/ID-LoRA-2.3/packages/ltx-trainer && pip uninstall bitsandbytes -y` |

**Known cosmetic startup warning:**
`ComfyUI-GGUF: Partial torch compile only, consider updating pytorch` — informational only.
Inference works fine on torch 2.6. Do not bump to 2.7+ without validating sageattention
and flash-attn wheel compatibility.

### LTX-2.3 Pipeline (Re-validated on CUDA)

All four LTX-2.3 model files from Phase 1 are present in `ComfyUI/models/`. The pipeline
has not yet been run on the 3090 — the VRAM headroom has grown from 16 GB to 24 GB, so
constraints loosen:

- `--lowvram` is **no longer required** — test without it first
- `tile_size=256` is still recommended; 512 may now work but is untested
- Runtime will be faster than the ~3 min 15 sec baseline on the RX 6800 XT

The node pipeline itself is unchanged from Phase 1:

| Stage | Node |
|---|---|
| Transformer | `UnetLoaderGGUF` |
| Text encoder | `DualCLIPLoaderGGUF` (both Gemma GGUF + text projection, type: `ltxv`) |
| Conditioning | `LTXVConditioning` |
| Scheduler | `LTXVScheduler` |
| Sampler | `SamplerCustomAdvanced` |
| VAE decode | `VAEDecodeTiled` (tile_size=256, temporal_size=32) |

### VRAM Budget — RTX 3090

| Component | GPU Memory |
|---|---|
| LTX-2.3-22B Transformer (GGUF Q3_K_M) | ~11.0 GB |
| Text Projection / Embeddings Connector | ~2.2 GB |
| VAE decode (tiled 256×32) | ~1.9 GB |
| **Peak during sampling** | **~13.2 GB (55% of 24 GB)** |
| Text Encoder (Gemma-3 Q4_K_M) | CPU offload |

~10 GB of headroom vs the 16 GB RX 6800 XT configuration. Larger resolutions and longer
frame counts are now viable without VRAM risk.

---

## Phase 3 — Wan 2.2 I2V Portrait Animation

**Goal:** animate still photos of relatives. Wan 2.2 I2V 14B was selected over LTX-2.3 for
this use case because identity preservation on faces is better in I2V models conditioned on
an input frame, while LTX-2.3's strength is text-to-video. No audio needed.

**Model selection rationale:**

| Candidate | Reason considered | Decision |
|---|---|---|
| Wan 2.1 VACE 1.3B | Fast, already on disk in `~/videoforge/` | Set aside — VACE is for controlled video editing, not portrait animation |
| Wan 2.2 I2V 14B GGUF | Best identity preservation; 14B scale; two-expert MoE | ✅ Selected |
| LTX 2.3 | Already set up; fast | Set aside — text-to-video model, no input-image conditioning |

### Key Finding — Two Loader Ecosystems

An investigation of pre-existing VideoForge files (`~/videoforge/models/wan21-vace-1.3b/`)
before downloading revealed a critical distinction that shaped all file decisions:

**ComfyUI has two parallel loader paths for Wan models:**

1. **Native ComfyUI loaders** (`Load CLIP`, `Load VAE`, `Unet Loader (GGUF)`) — accept
   `umt5_xxl_fp8_e4m3fn_scaled.safetensors`. Used by all official Comfy-Org Wan templates
   and by the GGUF pipeline.

2. **WanVideoWrapper loaders** (`Load WanVideo T5 TextEncoder`, etc.) — reject the
   fp8-scaled file. `LoadWanVideoT5TextEncoder` raises
   `ValueError("fp8 scaled is not supported by this node")` if given the Comfy-Org file.
   Requires bf16 format.

GGUF transformers run through native loaders. The VideoForge bf16 `.pth` files would work
with WanVideoWrapper but not with our pipeline. The Comfy-Org fp8-scaled safetensors is
the correct choice for GGUF workflows — and it's the smaller file (6.7 GB vs 10.58 GB).

**Decision: download everything fresh into `~/comfy/ComfyUI/models/` and keep VideoForge
cleanly separated.** No symlinks, no `extra_model_paths.yaml`.

### Model Inventory — Wan 2.2

All files present and confirmed (2026-05-05). The downloader used underscore-style names;
symlinks with the workflow's expected hyphen-style names were created in-place.

| Actual file on disk | Size | Symlink (used by workflow) |
|---|---|---|
| `wan2.2_i2v_high_noise_14B_Q5_K_M.gguf` | 11 GB | `wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf` |
| `wan2.2_i2v_low_noise_14B_Q5_K_M.gguf` | 11 GB | `wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf` |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.3 GB | (no rename needed) |
| `wan_2.1_vae.safetensors` | 243 MB | (no rename needed) |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | 1.2 GB | `Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors` |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | 1.2 GB | `Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors` |

### Workflow JSON

`wan22_i2v_portrait_animation.json` was built and validated. Based on the official
`Comfy-Org/workflow_templates video_wan2_2_14B_i2v.json` template with targeted modifications:

| Change | Why |
|---|---|
| `UNETLoader` → `UnetLoaderGGUF` on both transformer nodes | Our transformers are GGUF format |
| Filenames pre-set to our actual downloaded files | No dropdown confusion on first load |
| Default resolution 640×640 → 832×480 | Better aspect ratio for portraits |
| Frames: 81 (5 seconds @ 16 fps) | Baseline target duration |
| `Enable 4steps LoRA?` default: OFF | Validate 20-step baseline before Lightning |
| LoadImage filename cleared | Original had a demo filename that would error immediately |
| Negative prompt rewritten in English targeting portrait failures | Original used Chinese text; new version specifically suppresses talking, lip sync, identity drift, and face morphing |
| Markdown notes updated | Reflects GGUF path and actual file layout |

**Loader node → filename mapping:**

| Node | Filename |
|---|---|
| Load High-Noise Transformer (GGUF) | `wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf` |
| Load Low-Noise Transformer (GGUF) | `wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf` |
| CLIPLoader | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| VAELoader | `wan_2.1_vae.safetensors` |
| High-Noise Lightning LoRA | `Wan2.2-Lightning_I2V-A14B-4steps_HIGH.safetensors` |
| Low-Noise Lightning LoRA | `Wan2.2-Lightning_I2V-A14B-4steps_LOW.safetensors` |

### Wan 2.2 Pipeline Reference

Wan 2.2 14B uses a **Mixture-of-Experts (MoE) architecture** with separate high-noise and
low-noise expert transformers. They load and sample sequentially — ComfyUI offloads the
inactive expert during the switch. Both files are required; neither alone is sufficient.

**Sampler configuration (measured on RTX 3090, 832×480, 81 frames):**

| Mode | Steps | CFG | Split at | Time (3090) |
|---|---|---|---|---|
| Baseline (Lightning OFF) | 20 | 3.5 | step 10 | **17 min 11 sec** (~50 s/step) |
| Turbo (Lightning ON) | 4 | 1.0 | step 2 | ~2-4 min (untested) |

Use Original for finals. Use Lightning to scout whether a prompt direction or input photo
will work before committing to a full render.

**First-run failure modes:**

| Symptom | Cause | Fix |
|---|---|---|
| `Unknown node type: UnetLoaderGGUF` | ComfyUI-GGUF didn't import | Check `custom_nodes/ComfyUI-GGUF/` exists; restart |
| Red filename in loader dropdown | File missing or wrong directory | Re-run verify commands above |
| OOM at sampling | Resolution or frame count too large | Drop to 49 frames or 480×480 |
| `Tensor size mismatch` | LoRA version mismatch | Confirm both LoRAs are I2V not T2V |
| Pure noise output | ModelSamplingSD3 shift wrong | Both `ModelSamplingSD3` nodes should be 5.0 |
| Identity drifts at ~frame 40 | Temporal coherence limit of 14B model | Drop to 49 frames; or lower CFG to 3.0 |

### VRAM Budget — Wan 2.2 (measured 2026-05-05)

| Component | GPU Memory |
|---|---|
| Wan 2.2 14B transformer (GGUF Q5_K_M, one expert at a time) | 10.4 GB |
| UMT5-XXL text encoder (fp8 scaled) | 6.4 GB, CPU-offloaded after encoding |
| VAE + latents + KV cache (81 frames @ 832×480) | ~4.7 GB |
| **Actual peak during sampling** | **15.1 GB (61% of 24 GB)** |

No `--lowvram` needed. 9 GB headroom at peak.

---

## Next Steps

### ✅ Done — Wan 2.2 I2V first inference (2026-05-05)

20-step baseline completed end-to-end on the RTX 3090:
- 832×480, 81 frames, CFG 3.5, euler/simple
- Wall time: **17 min 11 sec** — MoE switch confirmed (high-noise 0→10, low-noise 10→20)
- Peak VRAM: 15.1 GB / 24 GB — no OOM
- Output: `ComfyUI/output/video/Wan2.2_i2v_00001_.mp4`

### Immediate

- **Test Wan 2.2 Lightning mode.** Toggle "Enable 4steps LoRA?" in the subgraph. CFG 1.0,
  same image — expected ~2-4 min. Use for scouting; use 20-step baseline for finals.

- **Install sageattention.** ~25-30% attention speedup on Ampere. Triton 3.2.0 already present:
  `pip install sageattention`, then swap `--use-pytorch-cross-attention` → `--use-sage-attention`
  in `launch.sh`. Will bring Wan 2.2 step time from ~50 s → ~35-38 s.

### Near-term

- **Re-verify LTX-2.3 on the 3090.** The pipeline has never been run on CUDA. Should work
  fine and be faster. Test `--lowvram` OFF first; the 24 GB headroom likely makes it
  unnecessary.

- **Fix `ID-LoRA-LTX2.3-ComfyUI` import.** When LTX identity work resumes:
  ```bash
  source ~/comfy/env.sh
  pip install ltx-core ltx-pipelines
  pip install -e ~/ID-LoRA/ID-LoRA-2.3/packages/ltx-trainer
  pip uninstall bitsandbytes -y
  ```

- **LTX-2.3 ID-LoRA weights.** ~1.1 GB from `Lightricks/LTX-Video-2.3`. Download when
  ID-LoRA work begins.

### Longer-term options

- **Q4_K_M Wan transformers.** ~2 GB VRAM savings per expert vs Q5_K_M. Enables 720p
  resolution headroom with minimal quality cost at 14B scale.
- **Wan 2.1 VACE 1.3B fast iteration.** The existing `~/videoforge/` files are intact and
  the `.pth` format is compatible with WanVideoWrapper. Could be wired up as a cheap
  60-frame draft mode.

---

## Quick Start

### Wan 2.2 I2V (validated ✅)

```bash
# 1. Launch
~/comfy/launch.sh          # UI at http://192.168.1.59:8188

# 2. Load workflow from the workflow menu:
#    wan22_i2v_portrait_animation.json

# 3. Upload a portrait photo → Queue
#    Lightning OFF (default): 20 steps, ~17 min, full quality
#    Lightning ON (subgraph toggle): 4 steps, ~2-4 min, scouting quality

# Output: ~/comfy/ComfyUI/output/video/Wan2.2_i2v_NNNNN_.mp4
```

### LTX-2.3 Text-to-Video (Phase 1 pipeline, awaiting CUDA re-validation)

```bash
# 1. Source environment and launch
~/comfy/launch.sh

# 2. Build workflow using these nodes (in order):
#    UnetLoaderGGUF → DualCLIPLoaderGGUF → LTXVConditioning
#    → LTXVScheduler → SamplerCustomAdvanced → VAEDecodeTiled
#    (see comfyui_rdna2.yaml for exact settings)

# 3. Baseline: 512×512, 17 frames, 20 steps, CFG 3.5
# 4. Run diagnostics anytime
python ~/comfy/diagnostics.py
```

---

## Useful Commands

```bash
# Source environment (always do this first)
source ~/comfy/env.sh

# Launch ComfyUI
~/comfy/launch.sh

# Check GPU
nvidia-smi
watch -n 1 nvidia-smi    # live during generation

# Confirm PyTorch sees the 3090
source ~/comfy/env.sh && python -c \
  "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0))"

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

# Check download session
tmux ls 2>/dev/null | grep wan22-dl
tmux attach -t wan22-dl    # Ctrl-b d to detach

# Check what's listening on 8188
ss -tlnp 2>/dev/null | grep 8188

# Kill hanging ComfyUI process
pkill -f "python.*main.py"

# Confirm --reserve-vram (not --lowvram) in launch.sh
grep -E "reserve-vram|lowvram" ~/comfy/launch.sh

# Run diagnostics
python ~/comfy/diagnostics.py
```
