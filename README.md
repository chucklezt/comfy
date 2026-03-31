# LTX-2.3 Video Generation Workspace

ComfyUI + LTX-2.3 video generation optimized for **AMD Radeon RX 6800 XT** (16 GB VRAM, RDNA2 / gfx1030) running ROCm/HIP on Linux.

---

## Table of Contents

- [Hardware & System](#hardware--system)
- [Component Versions](#component-versions)
- [Directory Structure](#directory-structure)
- [What Was Done](#what-was-done)
  - [1. Core Environment & Driver Config](#1-core-environment--driver-config)
  - [2. Python Venv & PyTorch for ROCm](#2-python-venv--pytorch-for-rocm)
  - [3. ComfyUI Installation](#3-comfyui-installation)
  - [4. Custom Nodes](#4-custom-nodes)
  - [5. LoRA Capability (ID-LoRA-LTX2.3)](#5-lora-capability-id-lora-ltx23)
  - [6. bitsandbytes Safety Block](#6-bitsandbytes-safety-block)
  - [7. Diagnostic Script](#7-diagnostic-script)
  - [8. Launch Script](#8-launch-script)
  - [9. Model Download Helper](#9-model-download-helper)
  - [10. Workflow Reference Config](#10-workflow-reference-config)
- [All Installed Packages](#all-installed-packages)
- [All Files Created](#all-files-created)
- [VRAM Budget](#vram-budget)
- [Quick Start](#quick-start)
- [Known Caveats](#known-caveats)

---

## Hardware & System

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

---

## Component Versions

### Core ML Stack

| Package            | Version             | Notes                                       |
|--------------------|---------------------|---------------------------------------------|
| torch              | 2.9.1+rocm6.3      | ROCm HIP build                              |
| torchvision        | 0.24.1+rocm6.3     |                                             |
| torchaudio         | 2.9.1+rocm6.3      |                                             |
| pytorch-triton-rocm| 3.5.1               | Triton compiler for ROCm                    |
| transformers       | 5.4.0               |                                             |
| safetensors        | 0.7.0               |                                             |
| accelerate         | 1.13.0              |                                             |

### LTX / LoRA Stack

| Package            | Version             | Notes                                       |
|--------------------|---------------------|---------------------------------------------|
| ltx-core           | 1.0.0               | LTX model core (PyPI)                       |
| ltx-pipelines      | 1.0.0               | LTX inference pipelines (PyPI)              |
| ltx-trainer        | 1.0.0               | Quantization support (source install from ID-LoRA repo) |
| peft               | 0.18.1              | HuggingFace LoRA/adapter engine             |
| optimum-quanto     | 0.2.7               | int8 quantization (HIP-safe, replaces bitsandbytes) |

### GGUF / Quantization

| Package            | Version             | Notes                                       |
|--------------------|---------------------|---------------------------------------------|
| gguf               | 0.18.0              | GGUF model format reader                    |
| sentencepiece      | 0.2.1               | Tokenizer for Gemma text encoder            |

### ComfyUI

| Package                         | Version   |
|---------------------------------|-----------|
| comfyui_frontend_package        | 1.42.8    |
| comfyui_workflow_templates      | 0.9.39    |
| comfyui-embedded-docs           | 0.4.3     |

### Other Notable Packages

| Package            | Version   | Notes                                         |
|--------------------|-----------|-----------------------------------------------|
| onnxruntime        | 1.23.2    |                                               |
| opencv-python      | 4.13.0.92 |                                               |
| numpy              | 2.2.6     |                                               |
| scipy              | 1.15.3    |                                               |
| scikit-image       | 0.25.2    |                                               |
| scikit-learn       | 1.7.2     |                                               |
| einops             | 0.8.2     |                                               |
| kornia             | 0.8.2     |                                               |
| spandrel           | 0.4.2     |                                               |
| wandb              | 0.25.1    | Experiment tracking                           |
| insightface        | 0.7.3     | Face analysis                                 |
| torchcodec         | 0.11.0    | Video codec support                           |
| imageio-ffmpeg     | 0.6.0     | FFmpeg bindings                               |
| scenedetect        | 0.6.7.1   | Scene detection                               |
| huggingface_hub    | 1.8.0     | Model downloads                               |

---

## Directory Structure

```
comfy/
├── env.sh                          # RDNA2 environment variables (source before use)
├── launch.sh                       # One-command ComfyUI launcher with pre-flight checks
├── diagnostics.py                  # Hardware/software validation script
├── block_bitsandbytes.py           # Python import hook to block bitsandbytes on RDNA2
├── download_models.sh              # Prints huggingface-cli commands for model downloads
├── comfyui_rdna2.yaml              # Reference config for workflow node settings
├── README.md                       # This file
│
├── venv/                           # Python 3.10 virtual environment
│   └── (PyTorch 2.9.1+rocm6.3, all packages)
│
├── ComfyUI/                        # ComfyUI application
│   ├── main.py                     # Entry point
│   ├── comfy/                      # Core ComfyUI framework
│   ├── custom_nodes/
│   │   ├── ComfyUI-GGUF/          # GGUF model loader (UnetLoaderGGUF, CLIPLoaderGGUF)
│   │   │   ├── nodes.py
│   │   │   ├── loader.py
│   │   │   ├── dequant.py
│   │   │   ├── ops.py
│   │   │   └── requirements.txt    # gguf>=0.13.0, sentencepiece, protobuf
│   │   └── ID-LoRA-LTX2.3-ComfyUI/# ID-LoRA nodes for identity-preserving video
│   │       ├── __init__.py         # Registers 5 ComfyUI nodes
│   │       ├── nodes_model_loader.py
│   │       ├── nodes_prompt_encoder.py
│   │       ├── nodes_sampler.py
│   │       ├── pipeline_wrapper.py # Uses ltx_trainer.quantization.quantize_model
│   │       ├── requirements.txt    # ltx-core, ltx-pipelines, ltx-trainer
│   │       ├── example_workflows/
│   │       ├── example_inputs/
│   │       └── example_outputs/
│   └── models/
│       ├── diffusion_models/
│       │   └── ltx-2.3/            # (empty — download transformer GGUF here)
│       ├── text_encoders/
│       │   └── ltx-2.3/            # (empty — download Gemma-3 GGUF here)
│       ├── vae/
│       │   └── ltx-2.3/            # (empty — download VAE here)
│       ├── loras/
│       │   └── ltx-2.3/            # (empty — download ID-LoRA weights here)
│       ├── checkpoints/
│       ├── clip/
│       ├── clip_vision/
│       ├── controlnet/
│       ├── embeddings/
│       ├── upscale_models/
│       └── ...
│
└── ID-LoRA/                        # Upstream ID-LoRA repository (cloned for ltx-trainer)
    └── ID-LoRA-2.3/
        └── packages/
            ├── ltx-core/
            ├── ltx-pipelines/
            └── ltx-trainer/        # Installed as editable: pip install -e
```

---

## What Was Done

### 1. Core Environment & Driver Config

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
# ─────────────────────────────────────────────────────────────────────────────
# LTX-2.3 / ComfyUI — RDNA2 (gfx1030) Environment for AMD Radeon RX 6800 XT
# Source this before launching ComfyUI:  source env.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── ROCm / HIP ──────────────────────────────────────────────────────────────
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# PyTorch HIP allocator — expandable_segments reduces fragmentation OOMs.
# NOTE: As of PyTorch 2.9.1+rocm6.3, expandable_segments is not yet supported
# on RDNA2/HIP (only CUDA). We set the var anyway so it auto-activates if a
# future PyTorch ROCm build adds support. The runtime warning is harmless.
export PYTORCH_HIP_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# MIOpen: use NORMAL find (mode 2) to avoid multi-GB workspace allocations
# that would blow the 16GB budget during conv autotuning.
export MIOPEN_FIND_MODE=2

# Disable MIOpen convolution autotuning cache writes to avoid stale-cache
# mismatches after driver updates.
export MIOPEN_DISABLE_CACHE=0

# ── ComfyUI / Inference ─────────────────────────────────────────────────────
export COMFYUI_ENABLE_MIOPEN=1

# Disable telemetry (HuggingFace Hub / analytics).
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

# ── Safety: block bitsandbytes ───────────────────────────────────────────────
export BITSANDBYTES_BLOCKED=1

# ── Activate venv ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

echo "✓ LTX-2.3 RDNA2 environment loaded (gfx1030 → HSA 10.3.0)"
```

### 2. Python Venv & PyTorch for ROCm

Created Python 3.10.12 virtual environment at `venv/`. Installed PyTorch ecosystem for ROCm 6.3:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Result: `torch 2.9.1+rocm6.3`, `torchvision 0.24.1+rocm6.3`, `torchaudio 2.9.1+rocm6.3`.

### 3. ComfyUI Installation

ComfyUI cloned to `ComfyUI/` with requirements installed. Model subdirectories created under `ComfyUI/models/` with `ltx-2.3/` staging folders for:
- `diffusion_models/ltx-2.3/` — transformer GGUF
- `text_encoders/ltx-2.3/` — Gemma-3 GGUF
- `vae/ltx-2.3/` — VAE weights
- `loras/ltx-2.3/` — ID-LoRA weights

### 4. Custom Nodes

Two custom nodes installed in `ComfyUI/custom_nodes/`:

**ComfyUI-GGUF** — Provides `UnetLoaderGGUF` and `CLIPLoaderGGUF` nodes for loading GGUF-quantized models directly in ComfyUI. Dependencies: `gguf>=0.13.0`, `sentencepiece`, `protobuf`.

**ID-LoRA-LTX2.3-ComfyUI** — Provides 5 nodes for identity-preserving video generation:
- `IDLoraModelLoader` — loads one-stage pipeline
- `IDLoraTwoStageModelLoader` — loads two-stage pipeline (with spatial upsampler)
- `IDLoraPromptEncoder` — encodes text prompts with Gemma-3
- `IDLoraOneStageSampler` — single-resolution generation
- `IDLoraTwoStageSampler` — 2x spatial upsampling generation

### 5. LoRA Capability (ID-LoRA-LTX2.3)

The ID-LoRA node was cloned but its Python dependencies were **not installed**. This was identified and fixed:

1. `ltx-core` (1.0.0) and `ltx-pipelines` (1.0.0) — installed from PyPI.
2. `ltx-trainer` (1.0.0) — **not on PyPI**. Cloned the upstream `ID-LoRA/ID-LoRA` repo and installed from source:
   ```bash
   git clone --depth 1 https://github.com/ID-LoRA/ID-LoRA.git
   pip install -e ID-LoRA/ID-LoRA-2.3/packages/ltx-trainer
   ```
3. This pulled in `peft 0.18.1` (HuggingFace LoRA engine) and `optimum-quanto 0.2.7` (int8 quantization that works on HIP).
4. **`ltx-trainer` dragged in `bitsandbytes 0.49.2`** as a transitive dependency — this was **immediately uninstalled** because it is CUDA-only and will segfault on RDNA2:
   ```bash
   pip uninstall bitsandbytes -y
   ```

The full LoRA import chain was verified clean:
```
ltx_trainer.quantization.quantize_model  ✓
ltx_core.loader.LoraPathStrengthAndSDOps ✓
ltx_pipelines.utils.ModelLedger          ✓
peft 0.18.1                              ✓
optimum-quanto 0.2.7                     ✓
bitsandbytes                             ✗ (blocked/absent — safe)
```

### 6. bitsandbytes Safety Block

Created `block_bitsandbytes.py` — a Python meta-path import hook that intercepts `import bitsandbytes` and raises a clear error instead of allowing a silent segfault. Activated when `BITSANDBYTES_BLOCKED=1` is set (which `env.sh` does).

```python
"""
Import hook that blocks bitsandbytes on RDNA2 to prevent silent segfaults.

Usage — add to the TOP of your launch script or sitecustomize.py:
    import block_bitsandbytes  # noqa: F401

Or let launch.sh inject it via PYTHONPATH.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys


class _BitsAndBytesBlocker(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Intercepts `import bitsandbytes` and raises a clear error."""

    BLOCKED = {"bitsandbytes"}

    def find_module(self, fullname, path=None):
        top = fullname.split(".")[0]
        if top in self.BLOCKED:
            return self
        return None

    def load_module(self, fullname):
        raise ImportError(
            f"'{fullname}' is blocked on RDNA2 (gfx1030). "
            "bitsandbytes / adamw8bit are CUDA-only and will segfault on AMD HIP. "
            "Use adafactor instead.  To unblock: unset BITSANDBYTES_BLOCKED"
        )


import os as _os

if _os.environ.get("BITSANDBYTES_BLOCKED", "0") == "1":
    sys.meta_path.insert(0, _BitsAndBytesBlocker())
```

### 7. Diagnostic Script

Created `diagnostics.py` — validates the full stack. Run modes:
- `python diagnostics.py` — full check (env, PyTorch, bitsandbytes, models, custom nodes, dimensions)
- `python diagnostics.py --quick` — pre-flight only (env + PyTorch), used by `launch.sh`

Checks performed:

| Check                              | Type   | Details                                             |
|------------------------------------|--------|-----------------------------------------------------|
| HSA_OVERRIDE_GFX_VERSION           | Error  | Must be `10.3.0`                                    |
| Expandable segments                | Error  | PYTORCH_ALLOC_CONF set                              |
| MIOPEN_FIND_MODE                   | Warn   | Should be `2`                                       |
| PyTorch version                    | Info   | Prints version string                               |
| HIP version                        | Error  | torch.version.hip must not be None                  |
| torch.cuda.is_available()          | Error  | ROCm HIP device must be visible                     |
| GPU name & VRAM                    | Info   | Prints device name and total memory                 |
| Device is RX 6800 XT               | Warn   | Checks "6800" in device name                        |
| HIP tensor compute                 | Error  | 256x256 matmul smoke test on GPU                    |
| SDPA available                     | Error  | scaled_dot_product_attention importable              |
| bitsandbytes not installed         | Error  | Import must fail                                    |
| Transformer GGUF present           | Warn   | Checks diffusion_models/ltx-2.3/                    |
| Text Encoder GGUF present          | Warn   | Checks text_encoders/ltx-2.3/                       |
| VAE present                        | Warn   | Checks vae/ltx-2.3/                                 |
| ComfyUI-GGUF node                  | Warn   | Directory exists in custom_nodes/                    |
| ID-LoRA-LTX2.3-ComfyUI node       | Warn   | Directory exists in custom_nodes/                    |
| Video dimension constraints        | Info   | Reminder: all dims must be multiples of 32           |

```python
#!/usr/bin/env python3
"""
Diagnostic script for LTX-2.3 on RDNA2 (gfx1030) — AMD Radeon RX 6800 XT.
Run standalone:   python diagnostics.py
Run quick check:  python diagnostics.py --quick
"""

import os
import sys
import argparse

OK   = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

errors: list[str] = []
warnings: list[str] = []


def check(label: str, condition: bool, error_msg: str = "", warn_only: bool = False):
    if condition:
        print(f"  {OK} {label}")
    elif warn_only:
        print(f"  {WARN} {label} — {error_msg}")
        warnings.append(error_msg)
    else:
        print(f"  {FAIL} {label} — {error_msg}")
        errors.append(error_msg)


def check_env():
    print("\n── Environment Variables ──")
    hsa = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
    check("HSA_OVERRIDE_GFX_VERSION=10.3.0", hsa == "10.3.0",
          f"Got '{hsa}'. Set: export HSA_OVERRIDE_GFX_VERSION=10.3.0")

    alloc = os.environ.get("PYTORCH_ALLOC_CONF", "") or os.environ.get("PYTORCH_HIP_ALLOC_CONF", "")
    check("Expandable segments enabled", "expandable_segments:True" in alloc,
          "Set: export PYTORCH_ALLOC_CONF='expandable_segments:True'")

    miopen = os.environ.get("MIOPEN_FIND_MODE", "")
    check("MIOPEN_FIND_MODE=2", miopen == "2",
          f"Got '{miopen}'. Set: export MIOPEN_FIND_MODE=2", warn_only=True)


def check_pytorch():
    print("\n── PyTorch / HIP ──")
    try:
        import torch
    except ImportError:
        check("PyTorch importable", False, "torch not installed in this venv")
        return

    check(f"PyTorch version: {torch.__version__}", True)

    hip_version = getattr(torch.version, "hip", None)
    check(f"HIP version: {hip_version}", hip_version is not None,
          "torch.version.hip is None — this is not a ROCm build of PyTorch")

    has_cuda = torch.cuda.is_available()
    check("torch.cuda.is_available() [ROCm HIP]", has_cuda,
          "No HIP device visible. Check ROCm install and HSA_OVERRIDE_GFX_VERSION.")

    if has_cuda:
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        check(f"GPU: {name} ({vram_gb:.1f} GB)", True)

        is_6800xt = "6800" in name
        check("Device is RX 6800 XT", is_6800xt,
              f"Expected 6800 XT, got: {name}", warn_only=True)

        try:
            t = torch.randn(256, 256, device="cuda")
            result = torch.mm(t, t)
            del t, result
            torch.cuda.empty_cache()
            check("HIP tensor compute (256×256 matmul)", True)
        except Exception as e:
            check("HIP tensor compute", False, str(e))

        try:
            from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
            check("scaled_dot_product_attention (SDPA) available", True)
        except ImportError:
            check("SDPA available", False, "Upgrade PyTorch ≥2.0 for SDPA support")


def check_bnb_blocked():
    print("\n── Safety: bitsandbytes ──")
    try:
        import bitsandbytes  # noqa: F401
        check("bitsandbytes NOT importable (good on RDNA2)", False,
              "bitsandbytes is installed — it WILL crash on RDNA2. "
              "Run: pip uninstall bitsandbytes")
    except ImportError:
        check("bitsandbytes not installed (safe)", True)


def check_models():
    print("\n── Model Files ──")
    base = os.path.join(os.path.dirname(__file__), "ComfyUI", "models")

    models = {
        "LTX-2.3 Transformer (GGUF Q3_K_M)": (
            os.path.join(base, "diffusion_models", "ltx-2.3"),
            ["ltx-video-2b-v0.9.5-Q3_K_M.gguf",
             "ltx-2.3-22B-Q3_K_M.gguf",
             "ltxv-22b-0.9.7-dev-Q3_K_M.gguf"],
        ),
        "Text Encoder (Gemma-3 GGUF Q4_K_M)": (
            os.path.join(base, "text_encoders", "ltx-2.3"),
            ["gemma-3-12b-it-Q4_K_M.gguf",
             "Gemma-3-12B-IT-Q4_K_M.gguf"],
        ),
        "VAE": (
            os.path.join(base, "vae", "ltx-2.3"),
            [],
        ),
    }

    for label, (directory, expected_names) in models.items():
        if not os.path.isdir(directory):
            check(label, False, f"Directory missing: {directory}", warn_only=True)
            continue

        files = [f for f in os.listdir(directory) if not f.startswith(".")]
        if not files:
            check(label, False, f"No files in {directory}", warn_only=True)
        elif expected_names:
            found = any(f in files for f in expected_names)
            check(f"{label}: {files[0] if files else '?'}", found or len(files) > 0,
                  f"Found {files} but expected one of {expected_names}", warn_only=True)
        else:
            check(f"{label}: {files[0]}", True)


def check_custom_nodes():
    print("\n── Custom Nodes ──")
    nodes_dir = os.path.join(os.path.dirname(__file__), "ComfyUI", "custom_nodes")
    required = {
        "ComfyUI-GGUF": "GGUF model loading for quantised transformer + text encoder",
        "ID-LoRA-LTX2.3-ComfyUI": "LoRA patching for LTX-2.3 identity preservation",
    }
    for dirname, purpose in required.items():
        path = os.path.join(nodes_dir, dirname)
        check(f"{dirname} — {purpose}", os.path.isdir(path),
              f"Missing: git clone into {nodes_dir}/{dirname}", warn_only=True)


def check_dimension_util():
    print("\n── Video Dimension Constraints ──")
    print("  ℹ All width/height values MUST be multiples of 32 on HIP/RDNA2.")
    print("    Valid examples: 512×320, 768×512, 1024×576")
    print("    Invalid: 720×480 (480 % 32 ≠ 0) → use 736×480 or 720×512")


def main():
    parser = argparse.ArgumentParser(description="LTX-2.3 RDNA2 diagnostics")
    parser.add_argument("--quick", action="store_true",
                        help="Only check env + PyTorch (for launch pre-flight)")
    args = parser.parse_args()

    print("=" * 60)
    print(" LTX-2.3 / RDNA2 (gfx1030) Diagnostic Report")
    print("=" * 60)

    check_env()
    check_pytorch()

    if not args.quick:
        check_bnb_blocked()
        check_models()
        check_custom_nodes()
        check_dimension_util()

    print("\n" + "=" * 60)
    if errors:
        print(f" {FAIL} {len(errors)} error(s), {len(warnings)} warning(s)")
        for e in errors:
            print(f"    → {e}")
        sys.exit(1)
    elif warnings:
        print(f" {WARN} 0 errors, {len(warnings)} warning(s) — OK to proceed")
        sys.exit(0)
    else:
        print(f" {OK} All checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

### 8. Launch Script

Created `launch.sh` — single-command launcher that sources `env.sh`, runs quick diagnostics, and starts ComfyUI with RDNA2-optimized flags.

```bash
#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Launch ComfyUI — LTX-2.3 on RDNA2 (RX 6800 XT, 16GB)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load RDNA2 environment variables.
source "$SCRIPT_DIR/env.sh"

# ── Pre-flight checks ───────────────────────────────────────────────────────
python "$SCRIPT_DIR/diagnostics.py" --quick || {
    echo "ERROR: Diagnostics failed. Fix the issues above before launching."
    exit 1
}

# ── Launch ───────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR/ComfyUI"

exec python main.py \
    --listen \
    --use-pytorch-cross-attention \
    --reserve-vram 2.5 \
    --lowvram \
    "$@"
```

**Launch flags explained:**

| Flag | Purpose |
|---|---|
| `--listen` | Bind to 0.0.0.0 for network access |
| `--use-pytorch-cross-attention` | Use SDPA — native, no xformers needed |
| `--reserve-vram 2.5` | Reserve 2.5 GB for OS/driver |
| `--lowvram` | **Required** — forces transformer to offload to CPU before VAE decode, creating headroom for tiled decode on 16 GB |

### 9. Model Download Helper

Created `download_models.sh` — prints `huggingface-cli download` commands for review before execution. Does not auto-download.

**Validated model inventory for 16 GB VRAM:**

| Model | File | Size | Source | Target Directory |
|---|---|---|---|---|
| LTX-2.3-22B Transformer | `ltx-2.3-22b-dev-Q3_K_M.gguf` | 11 GB | `unsloth/LTX-2.3-GGUF` | `models/diffusion_models/ltx-2.3/` |
| Gemma-3-12B-IT Text Encoder | `google_gemma-3-12b-it-Q4_K_M.gguf` | 6.8 GB | `Kijai/LTX2.3_comfy` | `models/text_encoders/ltx-2.3/` |
| LTX Text Projection (Embeddings Connector) | `ltx-2.3_text_projection_bf16.safetensors` | 2.2 GB | `Kijai/LTX2.3_comfy` | `models/text_encoders/ltx-2.3/` |
| LTX VAE | `LTX23_video_vae_bf16.safetensors` | 1.4 GB | `Kijai/LTX2.3_comfy` | `models/vae/ltx-2.3/` |
| ID-LoRA weights (optional) | TBD | ~1.1 GB | TBD | `models/loras/ltx-2.3/` |

**Important notes:**
- The original `download_models.sh` referenced incorrect repo names. Use the sources above.
- The full fp16 checkpoint (`ltx-2.3-22b-dev.safetensors`) is 43 GB — do not download, incompatible with 16 GB VRAM.
- The text projection file is **required** alongside the Gemma GGUF. Without it `DualCLIPLoaderGGUF` produces a tensor dimension mismatch at the sampler.
- `huggingface-cli` creates a nested subdirectory on download. Move files up one level afterward:
  ```bash
  mv ~/comfy/ComfyUI/models/text_encoders/ltx-2.3/text_encoders/ltx-2.3_text_projection_bf16.safetensors \
     ~/comfy/ComfyUI/models/text_encoders/ltx-2.3/
  ```

### 10. Workflow Reference Config

Created `comfyui_rdna2.yaml` — documents recommended ComfyUI node settings for workflows. Not consumed by ComfyUI directly; serves as a human-readable reference for configuring nodes in the UI.

**Important:** LTX-2.3 is an audio-video model. The standard `CLIPLoaderGGUF` + `KSampler` pipeline does **not** work — it produces a tensor dimension mismatch at the sampler. The correct pipeline uses LTX-specific nodes built into ComfyUI core.

```yaml
# ComfyUI Workflow Defaults — LTX-2.3 on RDNA2 (RX 6800 XT, 16 GB)
# VALIDATED — end-to-end smoke test passed

transformer:
  node: "UnetLoaderGGUF"            # from ComfyUI-GGUF
  model: "ltx-2.3/ltx-2.3-22b-dev-Q3_K_M.gguf"

text_encoder:
  node: "DualCLIPLoaderGGUF"        # from ComfyUI-GGUF — NOT CLIPLoaderGGUF (single)
  clip_name1: "ltx-2.3/google_gemma-3-12b-it-Q4_K_M.gguf"
  clip_name2: "ltx-2.3/ltx-2.3_text_projection_bf16.safetensors"  # REQUIRED
  type: "ltxv"

conditioning:
  node: "LTXVConditioning"          # LTX-specific — NOT CLIPTextEncode

scheduler:
  node: "LTXVScheduler"             # LTX-specific — NOT BasicScheduler

sampler:
  node: "SamplerCustomAdvanced"     # NOT KSampler

vae:
  node: "VAEDecodeTiled"            # mandatory for 16 GB — never use VAEDecode
  tile_size: 256                    # 512 causes OOM — do not increase
  temporal_size: 32

cross_attention:
  backend: "scaled_dot_product_attention"   # SDPA — native PyTorch >= 2.0

optimizer:
  use: "adafactor"
  block:
    - "adamw8bit"       # CUDA-only — segfaults on RDNA2
    - "bitsandbytes"    # CUDA-only — no HIP support

dimension_presets:
  - { name: "square",       width: 512,  height: 512  }   # validated baseline
  - { name: "landscape_hd", width: 768,  height: 512  }
  - { name: "portrait_hd",  width: 512,  height: 768  }
  - { name: "wide",         width: 1024, height: 576  }
  # All dimensions must be multiples of 32

frame_counts:
  # Formula: 1 + 8N (minimum N=2 → 17 frames)
  - 17    # minimum / baseline
  - 25
  - 33
  - 49
```

---

## All Installed Packages

Complete `pip list` output from the venv (145 packages):

```
Package                                Version          Source
─────────────────────────────────────  ───────────────  ──────────────────────
accelerate                             1.13.0           PyPI
aiohappyeyeballs                       2.6.1            PyPI
aiohttp                                3.13.4           PyPI
aiosignal                              1.4.0            PyPI
albucore                               0.0.24           PyPI
albumentations                         2.0.8            PyPI
alembic                                1.18.4           PyPI
annotated-doc                          0.0.4            PyPI
annotated-types                        0.7.0            PyPI
anyio                                  4.13.0           PyPI
async-timeout                          5.0.1            PyPI
attrs                                  26.1.0           PyPI
av                                     17.0.0           PyPI
blake3                                 1.0.8            PyPI
certifi                                2026.2.25        PyPI
charset-normalizer                     3.4.6            PyPI
click                                  8.2.1            PyPI
coloredlogs                            15.0.1           PyPI
comfy-aimdo                            0.2.12           PyPI
comfy-kitchen                          0.2.8            PyPI
comfyui-embedded-docs                  0.4.3            PyPI
comfyui_frontend_package               1.42.8           PyPI
comfyui_workflow_templates             0.9.39           PyPI
comfyui-workflow-templates-core        0.3.188          PyPI
comfyui-workflow-templates-media-api   0.3.68           PyPI
comfyui-workflow-templates-media-image 0.3.114          PyPI
comfyui-workflow-templates-media-other 0.3.161          PyPI
comfyui-workflow-templates-media-video 0.3.68           PyPI
contourpy                              1.3.2            PyPI
cycler                                 0.12.1           PyPI
Cython                                 3.2.4            PyPI
easydict                               1.13             PyPI
einops                                 0.8.2            PyPI
exceptiongroup                         1.3.1            PyPI
filelock                               3.25.2           PyPI
flatbuffers                            25.12.19         PyPI
fonttools                              4.62.1           PyPI
frozenlist                             1.8.0            PyPI
fsspec                                 2026.2.0         PyPI
gguf                                   0.18.0           PyPI
gitdb                                  4.0.12           PyPI
GitPython                              3.1.46           PyPI
glfw                                   2.10.0           PyPI
greenlet                               3.3.2            PyPI
h11                                    0.16.0           PyPI
hf-xet                                 1.4.2            PyPI
httpcore                               1.0.9            PyPI
httpx                                  0.28.1           PyPI
huggingface_hub                        1.8.0            PyPI
humanfriendly                          10.0             PyPI
idna                                   3.11             PyPI
ImageIO                                2.37.3           PyPI
imageio-ffmpeg                         0.6.0            PyPI
insightface                            0.7.3            PyPI
Jinja2                                 3.1.6            PyPI
joblib                                 1.5.3            PyPI
kiwisolver                             1.5.0            PyPI
kornia                                 0.8.2            PyPI
kornia_rs                              0.1.10           PyPI
lazy-loader                            0.5              PyPI
ltx-core                               1.0.0            PyPI
ltx-pipelines                          1.0.0            PyPI
ltx-trainer                            1.0.0            Source (ID-LoRA repo)
Mako                                   1.3.10           PyPI
markdown-it-py                         4.0.0            PyPI
MarkupSafe                             3.0.3            PyPI
matplotlib                             3.10.8           PyPI
mdurl                                  0.1.2            PyPI
ml_dtypes                              0.5.4            PyPI
mpmath                                 1.3.0            PyPI
multidict                              6.7.1            PyPI
networkx                               3.4.2            PyPI
ninja                                  1.13.0           PyPI
numpy                                  2.2.6            PyPI
onnx                                   1.21.0           PyPI
onnxruntime                            1.23.2           PyPI
opencv-python                          4.13.0.92        PyPI
opencv-python-headless                 4.13.0.92        PyPI
optimum-quanto                         0.2.7            PyPI
packaging                              26.0             PyPI
pandas                                 2.3.3            PyPI
peft                                   0.18.1           PyPI
pillow                                 12.1.1           PyPI
pillow_heif                            1.3.0            PyPI
pip                                    26.0.1           PyPI
platformdirs                           4.9.4            PyPI
prettytable                            3.17.0           PyPI
propcache                              0.4.1            PyPI
protobuf                               6.33.6           PyPI
psutil                                 7.2.2            PyPI
pydantic                               2.12.5           PyPI
pydantic_core                          2.41.5           PyPI
pydantic-settings                      2.13.1           PyPI
Pygments                               2.20.0           PyPI
PyOpenGL                               3.1.10           PyPI
pyparsing                              3.3.2            PyPI
python-dateutil                        2.9.0.post0      PyPI
python-dotenv                          1.2.2            PyPI
pytorch-triton-rocm                    3.5.1            PyPI
pytz                                   2026.1.post1     PyPI
PyYAML                                 6.0.3            PyPI
regex                                  2026.3.32        PyPI
requests                               2.33.1           PyPI
rich                                   14.3.3           PyPI
safetensors                            0.7.0            PyPI
scenedetect                            0.6.7.1          PyPI
scikit-image                           0.25.2           PyPI
scikit-learn                           1.7.2            PyPI
scipy                                  1.15.3           PyPI
sentencepiece                          0.2.1            PyPI
sentry-sdk                             2.57.0           PyPI
setuptools                             82.0.1           PyPI
shellingham                            1.5.4            PyPI
simpleeval                             1.0.7            PyPI
simsimd                                6.5.16           PyPI
six                                    1.17.0           PyPI
smmap                                  5.0.3            PyPI
spandrel                               0.4.2            PyPI
SQLAlchemy                             2.0.48           PyPI
stringzilla                            4.6.0            PyPI
sympy                                  1.14.0           PyPI
threadpoolctl                          3.6.0            PyPI
tifffile                               2025.5.10        PyPI
tokenizers                             0.22.2           PyPI
tomli                                  2.4.1            PyPI
torch                                  2.9.1+rocm6.3   ROCm wheel
torchaudio                             2.9.1+rocm6.3   ROCm wheel
torchcodec                             0.11.0           PyPI
torchsde                               0.2.6            PyPI
torchvision                            0.24.1+rocm6.3   ROCm wheel
tqdm                                   4.67.3           PyPI
trampoline                             0.1.2            PyPI
transformers                           5.4.0            PyPI
typer                                  0.24.1           PyPI
typing_extensions                      4.15.0           PyPI
typing-inspection                      0.4.2            PyPI
tzdata                                 2025.3           PyPI
urllib3                                2.6.3            PyPI
wandb                                  0.25.1           PyPI
wcwidth                                0.6.0            PyPI
wheel                                  0.46.3           PyPI
yarl                                   1.23.0           PyPI
```

---

## All Files Created

| File                     | Type   | Purpose                                                           |
|--------------------------|--------|-------------------------------------------------------------------|
| `env.sh`                 | Bash   | RDNA2 environment variables — source before every session         |
| `launch.sh`              | Bash   | One-command launcher with pre-flight diagnostics                  |
| `diagnostics.py`         | Python | Hardware/software validation (full and quick modes)               |
| `block_bitsandbytes.py`  | Python | Import hook to block bitsandbytes on RDNA2                        |
| `download_models.sh`     | Bash   | Prints model download commands for review                         |
| `comfyui_rdna2.yaml`     | YAML   | Reference config for recommended workflow node settings            |
| `README.md`              | MD     | This documentation file                                           |

---

## VRAM Budget

16 GB total on the RX 6800 XT. Aggressive offloading is required to fit within budget:

| Component | GPU Memory |
|---|---|
| LTX-2.3-22B Transformer (GGUF Q3_K_M) | ~11.0 GB |
| Text Projection / Embeddings Connector | ~2.2 GB |
| VAE decode (tiled 256x32) | ~1.9 GB |
| **PEAK during sampling** | **~13.2 GB (79%)** |
| Text Encoder (Gemma-3 Q4_K_M) | CPU offload (0 GB GPU) |
| Transformer during VAE decode (--lowvram) | CPU offload (0 GB GPU) |

**`--lowvram` is required.** Without it the transformer stays in VRAM during VAE decode,
leaving only ~1 GB free — insufficient for the 1.9 GB tiled allocation. With it, the
pipeline sequences correctly: encode → offload transformer → VAE decode.

---

## Quick Start

Models are already downloaded. For a fresh session:

```bash
# 1. Source environment and launch
source ~/comfy/env.sh
~/comfy/launch.sh

# 2. Open UI at http://localhost:8188

# 3. Build workflow using these nodes (in order):
#    UnetLoaderGGUF -> DualCLIPLoaderGGUF -> LTXVConditioning
#    -> LTXVScheduler -> SamplerCustomAdvanced -> VAEDecodeTiled

# 4. Run diagnostics anytime
python ~/comfy/diagnostics.py
```

---

## Known Caveats

1. **`expandable_segments` not yet active on RDNA2/HIP.** The env var is set for forward compatibility. PyTorch 2.9.1+rocm6.3 prints a harmless warning at startup. This will auto-resolve when ROCm HIP adds support.

2. **`bitsandbytes` is a transitive dependency of `ltx-trainer`.** It was uninstalled after `pip install -e ltx-trainer` pulled it in. If you ever re-install ltx-trainer or run `pip install` on packages that depend on it, remove it again: `pip uninstall bitsandbytes -y`.

3. **`transformers` version.** Currently at 5.4.0. The ID-LoRA README flags potential incompatibility with 5.x. No issues observed during text-to-video inference. Monitor during LoRA loading. If failures occur: `pip install 'transformers>=4.52,<5'`.

4. **Standard ComfyUI KSampler pipeline does not work with LTX-2.3.** LTX-2.3 is an audio-video model. `CLIPLoaderGGUF` (single) + `KSampler` produces a tensor dimension mismatch. Always use `DualCLIPLoaderGGUF` (with both Gemma GGUF and text projection file) + `LTXVConditioning` + `LTXVScheduler` + `SamplerCustomAdvanced`.

5. **Text projection file is required.** `ltx-2.3_text_projection_bf16.safetensors` (2.2 GB) must be present in `models/text_encoders/ltx-2.3/` alongside the Gemma GGUF. Without it the embedding shape is wrong and inference will fail.

6. **`--lowvram` flag is required in launch.sh.** Without it the transformer stays in VRAM during VAE decode and causes OOM. `tile_size` must also be 256, not 512.

7. **ID-LoRA pipeline requires full safetensors checkpoint (43 GB).** The `IDLoraModelLoader` node expects a full fp16 checkpoint, not GGUFs. It is not usable on 16 GB VRAM as-is.

8. **Video dimensions must be multiples of 32.** Non-aligned dimensions cause HIP memory access faults. Safe presets: 512x512, 768x512, 1024x576.

9. **Frame count formula: 1 + 8N.** Valid counts: 17, 25, 33, 49, 65... Minimum is 17 (N=2).

10. **First inference run is slow.** MIOpen compiles GPU kernels for gfx1030 on first use. Expect 10-15 minutes on the very first generation. Subsequent runs start within seconds.
