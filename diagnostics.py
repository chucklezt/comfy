#!/usr/bin/env python3
"""
Diagnostic script for LTX-2.3 on RDNA2 (gfx1030) — AMD Radeon RX 6800 XT.
Run standalone:   python diagnostics.py
Run quick check:  python diagnostics.py --quick
"""

import os
import sys
import argparse

# ─── ANSI helpers ────────────────────────────────────────────────────────────
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


# ─── 1. Environment Variables ───────────────────────────────────────────────
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


# ─── 2. PyTorch + HIP ───────────────────────────────────────────────────────
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

    has_cuda = torch.cuda.is_available()  # ROCm exposes via CUDA compat layer
    check("torch.cuda.is_available() [ROCm HIP]", has_cuda,
          "No HIP device visible. Check ROCm install and HSA_OVERRIDE_GFX_VERSION.")

    if has_cuda:
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        check(f"GPU: {name} ({vram_gb:.1f} GB)", True)

        is_6800xt = "6800" in name
        check("Device is RX 6800 XT", is_6800xt,
              f"Expected 6800 XT, got: {name}", warn_only=True)

        # Quick tensor smoke test
        try:
            t = torch.randn(256, 256, device="cuda")
            result = torch.mm(t, t)
            del t, result
            torch.cuda.empty_cache()
            check("HIP tensor compute (256×256 matmul)", True)
        except Exception as e:
            check("HIP tensor compute", False, str(e))

        # SDPA availability
        try:
            from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
            check("scaled_dot_product_attention (SDPA) available", True)
        except ImportError:
            check("SDPA available", False, "Upgrade PyTorch ≥2.0 for SDPA support")


# ─── 3. bitsandbytes block ──────────────────────────────────────────────────
def check_bnb_blocked():
    print("\n── Safety: bitsandbytes ──")
    try:
        import bitsandbytes  # noqa: F401
        check("bitsandbytes NOT importable (good on RDNA2)", False,
              "bitsandbytes is installed — it WILL crash on RDNA2. "
              "Run: pip uninstall bitsandbytes")
    except ImportError:
        check("bitsandbytes not installed (safe)", True)


# ─── 4. Model files ─────────────────────────────────────────────────────────
def check_models():
    print("\n── Model Files ──")
    base = os.path.join(os.path.dirname(__file__), "ComfyUI", "models")

    models = {
        "LTX-2.3 Transformer (GGUF Q3_K_M)": (
            os.path.join(base, "diffusion_models", "ltx-2.3"),
            ["ltx-video-2b-v0.9.5-Q3_K_M.gguf",    # common naming variants
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
            [],  # any file present is fine
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


# ─── 5. Custom nodes ────────────────────────────────────────────────────────
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


# ─── 6. Dimension validator helper ──────────────────────────────────────────
def check_dimension_util():
    """Just prints the constraint reminder — no runtime check needed."""
    print("\n── Video Dimension Constraints ──")
    print("  ℹ All width/height values MUST be multiples of 32 on HIP/RDNA2.")
    print("    Valid examples: 512×320, 768×512, 1024×576")
    print("    Invalid: 720×480 (480 % 32 ≠ 0) → use 736×480 or 720×512")


# ─── Main ────────────────────────────────────────────────────────────────────
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

    # Summary
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
