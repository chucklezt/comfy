#!/usr/bin/env python3
"""
Submit Wan 2.2 I2V first-inference run to ComfyUI via API.
Lightning=OFF (20-step baseline). Switch nodes inlined with Lightning=False values.
"""
import json, sys, time, requests
from pathlib import Path

BASE = "http://127.0.0.1:8188"
IMAGE_PATH = Path("/home/chuck/comfy/test1.jpg")
CLIENT_ID = "wan22-baseline-run"

# ── 1. Upload image ──────────────────────────────────────────────────────────
print(f"Uploading {IMAGE_PATH.name} ...")
with open(IMAGE_PATH, "rb") as f:
    r = requests.post(
        f"{BASE}/upload/image",
        files={"image": (IMAGE_PATH.name, f, "image/jpeg")},
        data={"overwrite": "true"},
    )
r.raise_for_status()
upload_name = r.json()["name"]
print(f"  → uploaded as: {upload_name}")

# ── 2. Build API-format prompt ───────────────────────────────────────────────
# Lightning=OFF: switch nodes bypassed, values inlined directly.
# Node IDs preserved from workflow so log cross-references work.

POSITIVE = (
    "A gentle, natural breathing motion. Subject's eyes blink slowly. "
    "Soft hair movement from a faint breeze. Subtle warm lighting shift. "
    "Cinematic photographic quality, shallow depth of field, photorealistic."
)
NEGATIVE = (
    "talking, mouth movement, lip sync, speaking, singing, extra limbs, "
    "extra fingers, deformed face, distorted face, warped features, "
    "identity drift, morphing face, melting features, low quality, blurry, "
    "jpeg artifacts, oversaturated, washed out, static, frozen, glitching, flickering"
)

prompt = {
    # ── Loaders ─────────────────────────────────────────────────────────────
    "97": {
        "class_type": "LoadImage",
        "inputs": {"image": upload_name, "upload": "image"},
    },
    "84": {
        "class_type": "CLIPLoader",
        "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            "type": "wan",
            "device": "default",
        },
    },
    "90": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
    },
    "95": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "wan2.2-i2v-A14B-HighNoise-Q5_K_M.gguf"},
    },
    "96": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "wan2.2-i2v-A14B-LowNoise-Q5_K_M.gguf"},
    },
    # ── ModelSamplingSD3 shift=5.0 (Lightning OFF → no LoRA passthrough) ────
    "104": {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": ["95", 0], "shift": 5.0},
    },
    "103": {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": ["96", 0], "shift": 5.0},
    },
    # ── Text conditioning ────────────────────────────────────────────────────
    "93": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["84", 0], "text": POSITIVE},
    },
    "89": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["84", 0], "text": NEGATIVE},
    },
    # ── I2V conditioning + empty latent ─────────────────────────────────────
    "98": {
        "class_type": "WanImageToVideo",
        "inputs": {
            "positive":   ["93", 0],
            "negative":   ["89", 0],
            "vae":        ["90", 0],
            "width":      832,
            "height":     480,
            "length":     81,
            "batch_size": 1,
            "start_image": ["97", 0],
        },
    },
    # ── Sampling: high-noise expert, steps 0→10 ──────────────────────────────
    "86": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model":                   ["104", 0],
            "add_noise":               "enable",
            "noise_seed":              264244520398999,
            "steps":                   20,
            "cfg":                     3.5,
            "sampler_name":            "euler",
            "scheduler":               "simple",
            "positive":                ["98", 0],
            "negative":                ["98", 1],
            "latent_image":            ["98", 2],
            "start_at_step":           0,
            "end_at_step":             10,
            "return_with_leftover_noise": "enable",
        },
    },
    # ── Sampling: low-noise expert, steps 10→20 ──────────────────────────────
    "85": {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model":                   ["103", 0],
            "add_noise":               "disable",
            "noise_seed":              0,
            "steps":                   20,
            "cfg":                     3.5,
            "sampler_name":            "euler",
            "scheduler":               "simple",
            "positive":                ["98", 0],
            "negative":                ["98", 1],
            "latent_image":            ["86", 0],
            "start_at_step":           10,
            "end_at_step":             20,
            "return_with_leftover_noise": "disable",
        },
    },
    # ── Decode + output ──────────────────────────────────────────────────────
    "87": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["85", 0], "vae": ["90", 0]},
    },
    "94": {
        "class_type": "CreateVideo",
        "inputs": {"images": ["87", 0], "fps": 16.0},
    },
    "108": {
        "class_type": "SaveVideo",
        "inputs": {
            "video":           ["94", 0],
            "filename_prefix": "video/Wan2.2_i2v",
            "format":          "auto",
            "codec":           "auto",
        },
    },
}

# ── 3. Submit prompt ─────────────────────────────────────────────────────────
payload = {"prompt": prompt, "client_id": CLIENT_ID}
print("Queuing prompt ...")
r = requests.post(f"{BASE}/prompt", json=payload)
if r.status_code != 200:
    print(f"ERROR {r.status_code}: {r.text}")
    sys.exit(1)
resp = r.json()
prompt_id = resp.get("prompt_id", "?")
print(f"  → queued, prompt_id: {prompt_id}")
print(f"\nMonitor at http://localhost:8188")
print(f"Tail log:  tail -f ~/comfy/comfyui.log")

# ── 4. Poll until done ───────────────────────────────────────────────────────
print("\nWaiting for completion", end="", flush=True)
start = time.time()
while True:
    time.sleep(5)
    hist = requests.get(f"{BASE}/history/{prompt_id}").json()
    if prompt_id in hist:
        elapsed = time.time() - start
        entry = hist[prompt_id]
        status = entry.get("status", {})
        print(f"\n\nDone in {elapsed:.0f}s  status={status.get('status_str','?')}")
        outputs = entry.get("outputs", {})
        for nid, out in outputs.items():
            print(f"  node {nid}: {out}")
        break
    print(".", end="", flush=True)
