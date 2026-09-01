#!/usr/bin/env python3
"""
End-to-end denoise-stage benchmark presets for SGLang Diffusion.

Measures denoise latency (primary metric ★) and peak GPU memory.
All model configs are kept in exact sync with benchmark-and-profile.md.

Usage:
    # Single model
    cd /path/to/sglang
    python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py --model flux

    # Tag the run for later compare_perf.py usage
    python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py --model flux --label tuned

    # Opt in to a compile control (presets are eager by default)
    python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py --model flux --torch-compile

    # Check Eager/BCG at lossless/high on one GPU set; high+BCG is invalid when
    # request-scoped DiT fusions mount only after lossless graph capture.
    python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py --model sana-video --quality-bcg-matrix --model-cache-root /task/model-caches --cleanup-model-cache

    # Clean an isolated model cache even if the run fails or is interrupted
    python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py --model longcat-image --model-cache-root /task/model-caches --cleanup-model-cache

    # All preset models
    python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py --all

    # Show preset order, model path, and nightly mapping
    python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py --list-models

For gated Hugging Face repos such as FLUX, export HF_TOKEN first:
    export HF_TOKEN=<your_hf_token>

Input images required for image-guided models:
    ASSET_DIR=$(python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py print-assets-dir --mkdir)
    wget -O "${ASSET_DIR}/cat.png" \
      https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
    wget -O "${ASSET_DIR}/mova_single_person.jpg" \
      https://github.com/OpenMOSS/MOVA/raw/main/assets/single_person.jpg
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diffusion_skill_env import (
    ensure_dir,
    get_assets_dir,
    get_output_dir,
    get_repo_root,
    pick_idle_gpus,
)

REPO_ROOT = get_repo_root()
ASSET_DIR = ensure_dir(get_assets_dir(REPO_ROOT))
NIGHTLY_CONFIG_PATH = (
    REPO_ROOT / "scripts" / "ci" / "utils" / "diffusion" / "comparison_configs.json"
)
GATED_MODELS = {
    "flux",
    "flux2",
    "flux2-klein",
    "flux2-klein-base",
    "stable-diffusion-3.5-medium",
}
DIFFUSERS_FALLBACK_SIGNALS = (
    "falling back to diffusers backend",
    "using diffusers backend",
    "loaded diffusers pipeline",
)
BENCHMARK_QUALITY_LEVELS = ("lossless", "high")
BCG_CAPTURE_SIGNAL = "[diffusion bcg] captured"
BCG_INVALID_SIGNALS = (
    "[diffusion bcg] capture failed",
    "[diffusion bcg] disabled",
    "[diffusion bcg] serving signature missed",
    "no graph will be captured",
    "quality='high' cannot be used with breakable cuda graphs",
)
BCG_LATE_QUALITY_FUSION_SIGNAL = "quality fusion mounted after BCG capture"
QUALITY_BCG_ABBA_MATRIX = (
    ("eager-lossless-a", "lossless", False),
    ("bcg-lossless-a", "lossless", True),
    ("bcg-lossless-b", "lossless", True),
    ("eager-lossless-b", "lossless", False),
    ("eager-high-a", "high", False),
    ("bcg-high-a", "high", True),
    ("bcg-high-b", "high", True),
    ("eager-high-b", "high", False),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CATALOG_TABLE_WIDTH = 140
RESULTS_TABLE_WIDTH = 124
MODEL_CACHE_MARKER = ".sglang-diffusion-benchmark-cache"
MODEL_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".pt",
    ".pth",
    ".safetensors",
}
GENERATED_OUTPUT_SUFFIXES = {
    ".glb",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".obj",
    ".png",
    ".wav",
    ".webp",
}
NIGHTLY_PRESET_ORDER = (
    "flux",
    "flux2",
    "qwen",
    "qwen-edit",
    "zimage",
    "wan-t2v",
    "wan-ti2v",
    "ltx23-ti2v-two-stage",
    "ideogram4-fp8",
    "cosmos3-super-t2v",
    "wan-i2v",
    "minimax-h3-t2va",
)

LINGBOT_VIDEO_PROMPT = json.dumps(
    {
        "comprehensive_description": {
            "scene_content_description": (
                "A small silver robot arm on a white table slowly reaches "
                "toward a red cube. The background is a plain, softly lit "
                "laboratory wall."
            ),
            "camera_movement_description": (
                "The camera is static at eye level, medium shot, with the "
                "robot arm centered and in sharp focus."
            ),
        },
        "camera_info": {
            "color": "Neutral",
            "frame_size": "Medium",
            "shot_type_angle": "Eye level",
            "lens_size": "Medium",
            "composition": "Center",
            "lighting": "Soft light",
            "lighting_type": "Artificial light",
        },
        "world_knowledge": [],
        "prominent_elements": [
            {
                "name": "robot arm",
                "description": "A small silver robot arm with a two-finger gripper.",
                "actions": [
                    {
                        "timestamp": "[0.0s - 1.0s]",
                        "action": "reaches toward the red cube",
                    }
                ],
                "location": "center of the frame",
                "relative_size": "dominant",
                "shape_and_color": "articulated silver metal arm",
                "texture": "brushed metal",
                "appearance_details": "two-finger gripper, visible joints",
                "relationship": "reaching toward the red cube on the table",
                "orientation": "upright, base on the table",
                "pose": "reaching",
                "expression": "",
                "clothing": "",
                "gender": "",
                "skin_tone_and_texture": "",
            }
        ],
    },
    separators=(",", ":"),
)
LINGBOT_WORLD_CONFIG_OVERRIDES = {
    "actions": [["w"] for _ in range(9)],
}

# ---------------------------------------------------------------------------
# Model configs — kept in exact sync with benchmark-and-profile.md
# Nightly-aligned presets mirror scripts/ci/utils/diffusion/comparison_configs.json
# first, followed by current-source extras and skill-only stress / coverage presets.
# Each entry produces the same `sglang generate` command as shown in that doc.
# ---------------------------------------------------------------------------
MODELS = {
    # 1. Nightly: flux1_dev_t2i_1024
    "flux": {
        "nightly_case_id": "flux1_dev_t2i_1024",
        "path": "black-forest-labs/FLUX.1-dev",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-gpus=2",
            "--tp-size=2",
            "--component-residency=dit=resident",
        ],
    },
    # 2. Nightly: flux2_dev_t2i_1024
    "flux2": {
        "nightly_case_id": "flux2_dev_t2i_1024",
        "path": "black-forest-labs/FLUX.2-dev",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-gpus=2",
            "--tp-size=2",
            "--component-residency=dit=resident",
        ],
    },
    # 3. Nightly: qwen_image_2512_t2i_1024
    "qwen": {
        "nightly_case_id": "qwen_image_2512_t2i_1024",
        "path": "Qwen/Qwen-Image-2512",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-gpus=2",
            "--tp-size=2",
        ],
    },
    # 4. Nightly: qwen_image_edit_2511
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "qwen-edit": {
        "nightly_case_id": "qwen_image_edit_2511",
        "path": "Qwen/Qwen-Image-Edit-2511",
        "prompt": "Make the cat wear a red hat",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-gpus=2",
            "--tp-size=2",
        ],
    },
    # 5. Nightly: zimage_turbo_t2i_1024
    "zimage": {
        "nightly_case_id": "zimage_turbo_t2i_1024",
        "path": "Tongyi-MAI/Z-Image-Turbo",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-gpus=2",
            "--tp-size=2",
        ],
    },
    # 6. Nightly: wan22_t2v_a14b_720p
    "wan-t2v": {
        "nightly_case_id": "wan22_t2v_a14b_720p",
        "path": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "prompt": "A cat and a dog baking a cake together in a kitchen.",
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
            "--num-gpus=4",
            "--enable-cfg-parallel",
            "--ulysses-degree=2",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
        ],
    },
    # 7. Nightly: wan22_ti2v_5b_720p
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "wan-ti2v": {
        "nightly_case_id": "wan22_ti2v_5b_720p",
        "path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
        ],
    },
    # 8. Nightly: ltx2.3_twostage_ti2v_2gpus
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "ltx23-ti2v-two-stage": {
        "nightly_case_id": "ltx2.3_twostage_ti2v_2gpus",
        "path": "Lightricks/LTX-2.3",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--pipeline-class-name=LTX2TwoStagePipeline",
            "--width=768",
            "--height=512",
            "--num-frames=121",
            "--num-gpus=2",
            "--cfg-parallel-size=2",
        ],
    },
    # 9. Nightly: ideogram4_fp8_t2i_2gpu
    "ideogram4-fp8": {
        "nightly_case_id": "ideogram4_fp8_t2i_2gpu",
        "path": "ideogram-ai/ideogram-4-fp8",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-gpus=2",
            "--tp-size=2",
            "--attention-backend=fa",
        ],
    },
    # 10. Nightly: cosmos3_super_t2v_2gpu
    "cosmos3-super-t2v": {
        "nightly_case_id": "cosmos3_super_t2v_2gpu",
        "path": "nvidia/Cosmos3-Super",
        "prompt": "A cat and a dog baking a cake together in a kitchen.",
        "env": {
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
            "--num-gpus=2",
            "--tp-size=2",
        ],
    },
    # Explicit throughput comparator. CFG parallelism changes sampling numerics,
    # so compare its output against the TP=2 preset before using the speedup.
    "cosmos3-super-t2v-cfg2tp2": {
        "path": "nvidia/Cosmos3-Super",
        "prompt": "A cat and a dog baking a cake together in a kitchen.",
        "env": {
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
            "--num-gpus=4",
            "--tp-size=2",
        ],
    },
    # 11. Nightly: wan22_i2v_a14b_720p
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "wan-i2v": {
        "nightly_case_id": "wan22_i2v_a14b_720p",
        "path": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
            "--num-gpus=4",
            "--enable-cfg-parallel",
            "--ulysses-degree=2",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
        ],
    },
    # 12. Nightly: minimax_h3_t2va_5s
    # MiniMax-H3 owns its temporal canvas through target.duration_seconds, so
    # the model-specific sampling fields are passed through --config instead
    # of generic --width/--height/--num-frames flags.
    "minimax-h3-t2va": {
        "nightly_case_id": "minimax_h3_t2va_5s",
        "path": "MiniMaxAI/MiniMax-H3",
        "prompt": "At night, while their owner sleeps in a bedroom, three cats march in loudly playing tiny brass instruments, then abruptly file out.",
        "seed": 1101,
        "config_overrides": {
            "task": "t2va",
            "conditions": [],
            "target": {
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 5.0,
            },
            "audio_flow_shift": 3.0,
            "flow_shift": 12.0,
            "num_inference_steps": 50,
        },
        "extra_args": [
            "--model-variant=fl2va",
            "--num-gpus=4",
            "--tp-size=2",
            "--ulysses-degree=2",
            "--performance-mode=speed",
            "--enable-torch-compile=false",
        ],
        # H3 eager BF16/FP32 is the consistency ground truth. Current
        # torch.compile changes numerical output, so never add the global
        # helper default --enable-torch-compile flag for this preset.
        "force_eager": True,
        "nightly_cli_ignored": {
            "width",
            "height",
            "num-frames",
            "fps",
            "num-inference-steps",
        },
    },
    # Source-tracked extras from current registry / GPU test coverage.
    "longcat-image": {
        "path": "meituan-longcat/LongCat-Image",
        "prompt": "A red panda reading a book beside a sunlit window.",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=50",
            "--guidance-scale=4.5",
            "--enable-prompt-rewrite=false",
            "--performance-mode=manual",
        ],
    },
    "longcat-image-edit": {
        "path": "meituan-longcat/LongCat-Image-Edit",
        "prompt": "Make the cat wear a red hat.",
        "image_path": "https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
        "bcg_warmup_resolutions": ["1264x848"],
        "extra_args": [
            "--enable-prompt-rewrite=false",
            "--performance-mode=manual",
        ],
    },
    "longcat-image-edit-turbo": {
        "path": "meituan-longcat/LongCat-Image-Edit-Turbo",
        "prompt": "Make the cat wear a red hat.",
        "image_path": "https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
        "bcg_warmup_resolutions": ["1264x848"],
        "extra_args": [
            "--enable-prompt-rewrite=false",
            "--performance-mode=manual",
        ],
    },
    # The original Qwen edit checkpoint has a separate pipeline config from
    # the 2509/2511 multi-image checkpoints, so keep an explicit preset.
    "qwen-edit-base": {
        "path": "Qwen/Qwen-Image-Edit",
        "prompt": "Make the cat wear a red hat.",
        "image_path": "https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    "qwen-image-layered": {
        "path": "Qwen/Qwen-Image-Layered",
        "prompt": "a high quality, cute halloween themed illustration, consistent style and lighting",
        "image_path": "https://raw.githubusercontent.com/QwenLM/Qwen-Image-Layered/main/assets/test_images/4.png",
        "extra_args": [
            "--num-frames=4",
            "--width=640",
            "--height=640",
        ],
    },
    "stable-diffusion-3.5-medium": {
        "path": "stabilityai/stable-diffusion-3.5-medium-diffusers",
        "prompt": "A red panda reading a book beside a sunlit window.",
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    "sana-video": {
        "path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        "prompt": "A curious raccoon walks through a sunlit forest. motion score: 30.",
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=17",
            "--fps=16",
            "--num-inference-steps=8",
            "--guidance-scale=6.0",
            "--performance-mode=manual",
        ],
    },
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "sana-wm-bidirectional": {
        "path": "Efficient-Large-Model/SANA-WM_bidirectional",
        "prompt": "a camera moving forward and turning left",
        "image_path": str(ASSET_DIR / "cat.png"),
        "seed": 42,
        "extra_args": [
            "--pipeline-class-name=SanaWMTwoStagePipeline",
            "--width=1280",
            "--height=704",
            "--num-frames=49",
            "--fps=16",
            "--num-inference-steps=20",
            "--guidance-scale=4.5",
            "--action=w-16,wl-16,l-16",
            "--performance-mode=manual",
        ],
    },
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "sana-wm-streaming": {
        "path": "Efficient-Large-Model/SANA-WM_streaming",
        "prompt": "a camera moving forward and turning left",
        "image_path": str(ASSET_DIR / "cat.png"),
        "seed": 42,
        "extra_args": [
            "--pipeline-class-name=SanaWMTwoStagePipeline",
            "--streaming",
            "--refiner-chunked",
            "--width=1280",
            "--height=704",
            "--num-frames=49",
            "--fps=16",
            "--action=w-16,wl-16,l-16",
            "--performance-mode=manual",
        ],
    },
    "lingbot-video-moe": {
        "path": "robbyant/lingbot-video-moe-30b-a3b",
        "prompt": LINGBOT_VIDEO_PROMPT,
        "seed": 0,
        "extra_args": [
            "--width=384",
            "--height=640",
            "--num-frames=17",
            "--fps=16",
            "--num-inference-steps=12",
            "--text-encoder-cpu-offload",
            "--performance-mode=manual",
        ],
    },
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "lingbot-world": {
        "path": "robbyant/lingbot-world-fast-diffusers",
        "prompt": "A slow aerial orbit around a pastel island hotel in the ocean.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "seed": 42,
        "config_overrides": LINGBOT_WORLD_CONFIG_OVERRIDES,
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=9",
            "--fps=16",
            "--num-inference-steps=4",
            "--guidance-scale=1.0",
            "--text-encoder-cpu-offload",
            "--warmup-mode=off",
        ],
    },
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "lingbot-world-v2": {
        "path": "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
        "prompt": "A slow aerial orbit around a pastel island hotel in the ocean.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "seed": 42,
        "config_overrides": LINGBOT_WORLD_CONFIG_OVERRIDES,
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=9",
            "--fps=16",
            "--num-inference-steps=4",
            "--guidance-scale=1.0",
            "--text-encoder-cpu-offload",
            "--warmup-mode=off",
        ],
    },
    "fastwan21-t2v-1.3b": {
        "path": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        "prompt": "A curious raccoon walks through a sunlit forest.",
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=61",
            "--fps=16",
            "--num-inference-steps=3",
            "--performance-mode=manual",
            "--dit-layerwise-offload=false",
            "--dit-cpu-offload=false",
        ],
    },
    "wan21-t2v-1.3b": {
        "path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "prompt": "A curious raccoon walks through a sunlit forest.",
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=81",
            "--fps=16",
            "--num-inference-steps=50",
            "--guidance-scale=3.0",
        ],
    },
    "wan21-t2v-14b": {
        "path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "prompt": "A curious raccoon",
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=81",
            "--fps=16",
            "--num-inference-steps=50",
            "--guidance-scale=5.0",
            "--num-gpus=4",
            "--enable-cfg-parallel",
            "--ulysses-degree=2",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
        ],
    },
    "wan21-i2v-14b-480p": {
        "path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=81",
            "--fps=16",
            "--num-inference-steps=50",
            "--guidance-scale=5.0",
            "--num-gpus=4",
            "--enable-cfg-parallel",
            "--ulysses-degree=2",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
        ],
    },
    "wan21-i2v-14b-720p": {
        "path": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
            "--fps=16",
            "--num-inference-steps=50",
            "--guidance-scale=5.0",
            "--num-gpus=4",
            "--enable-cfg-parallel",
            "--ulysses-degree=2",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
        ],
    },
    "wan21-fun-inp-1.3b": {
        "path": "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=81",
            "--fps=16",
            "--num-inference-steps=50",
            "--guidance-scale=6.0",
        ],
    },
    "krea2-turbo": {
        "path": "krea/Krea-2-Turbo",
        "prompt": "A red fox sitting in fresh snow, golden hour, photorealistic.",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=8",
            "--guidance-scale=1.0",
        ],
    },
    "krea2-raw": {
        "path": "krea/Krea-2-Raw",
        "prompt": "A red fox sitting in fresh snow, golden hour, photorealistic.",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=50",
            "--guidance-scale=4.5",
        ],
    },
    "ideogram4-fast": {
        "path": "fal/ideogram-v4-fast",
        "prompt": "A vintage travel poster for Kyoto with crisp readable lettering.",
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    "ideogram4-instant": {
        "path": "fal/ideogram-v4-instant",
        "prompt": "A vintage travel poster for Kyoto with crisp readable lettering.",
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    "longlive2-t2v": {
        "path": "Rabinovich/LongLive-2.0-5B-Diffusers",
        "prompt": "A curious raccoon",
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=61",
            "--num-inference-steps=4",
            "--guidance-scale=1.0",
        ],
    },
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "longlive2-i2v": {
        "path": "Rabinovich/LongLive-2.0-5B-Diffusers",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=960",
            "--height=928",
            "--num-frames=61",
            "--num-inference-steps=4",
            "--guidance-scale=1.0",
        ],
    },
    "fast-hunyuan": {
        "path": "FastVideo/FastHunyuan-diffusers",
        "prompt": "A curious raccoon",
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=61",
            "--num-inference-steps=6",
        ],
    },
    "turbowan21-t2v-1.3b": {
        "path": "IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers",
        "prompt": "A curious raccoon",
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=81",
            "--num-inference-steps=4",
        ],
    },
    "turbowan21-t2v-14b-480p": {
        "path": "IPostYellow/TurboWan2.1-T2V-14B-Diffusers",
        "prompt": "A curious raccoon",
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=81",
            "--num-inference-steps=4",
        ],
    },
    "turbowan21-t2v-14b-720p": {
        "path": "IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers",
        "prompt": "A curious raccoon",
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
            "--num-inference-steps=4",
        ],
    },
    "turbowan22-i2v-a14b": {
        "path": "IPostYellow/TurboWan2.2-I2V-A14B-Diffusers",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
            "--fps=16",
            "--num-inference-steps=4",
            "--guidance-scale=3.5",
            "--guidance-scale-2=3.5",
            "--num-gpus=4",
            "--enable-cfg-parallel",
            "--ulysses-degree=2",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
        ],
    },
    "helios-mid": {
        "path": "BestWishYsh/Helios-Mid",
        "prompt": "A curious raccoon",
        "extra_args": [
            "--width=640",
            "--height=384",
            "--num-frames=33",
            "--num-inference-steps=20",
        ],
    },
    "helios-distilled": {
        "path": "BestWishYsh/Helios-Distilled",
        "prompt": "A curious raccoon",
        "extra_args": [
            "--width=640",
            "--height=384",
            "--num-frames=33",
            "--num-inference-steps=10",
            "--guidance-scale=1.0",
        ],
    },
    "joy-echo": {
        "path": "jdopensource/JoyAI-Echo",
        "prompt": "A curious raccoon",
        "seed": 42,
        "config_overrides": {
            "enable_memory_bank": False,
        },
        "extra_args": [
            "--width=640",
            "--height=384",
            "--num-frames=33",
            "--num-inference-steps=8",
            "--num-gpus=2",
            "--ulysses-degree=2",
        ],
    },
    "cosmos3-edge-t2i": {
        "path": "nvidia/Cosmos3-Edge",
        "prompt": "A warehouse robot folds a blue cloth on a clean workbench.",
        "seed": 0,
        "env": {
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
        "extra_args": [
            "--width=640",
            "--height=640",
            "--num-frames=1",
            "--num-inference-steps=35",
            "--guidance-scale=7.0",
            "--performance-mode=manual",
        ],
    },
    "cosmos3-edge-t2v": {
        "path": "nvidia/Cosmos3-Edge",
        "prompt": "A warehouse robot carefully places a blue box on a shelf.",
        "seed": 42,
        "env": {
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=81",
            "--fps=24",
            "--num-inference-steps=35",
            "--guidance-scale=5.0",
            "--performance-mode=manual",
        ],
    },
    "cosmos3-edge-i2v": {
        "path": "nvidia/Cosmos3-Edge",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "seed": 42,
        "env": {
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=81",
            "--fps=24",
            "--num-inference-steps=35",
            "--guidance-scale=5.0",
            "--performance-mode=manual",
        ],
    },
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "cosmos3-super-i2v": {
        "path": "nvidia/Cosmos3-Super-Image2Video",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "seed": 42,
        "env": {
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
            "--fps=24",
            "--num-inference-steps=35",
            "--guidance-scale=6.0",
            "--flow-shift=10.0",
            "--num-gpus=2",
            "--tp-size=2",
        ],
    },
    "cosmos3-super-t2i-distilled": {
        "path": "nvidia/Cosmos3-Super-Text2Image-4Step",
        "prompt": "A warehouse robot folds a blue cloth on a clean workbench.",
        "seed": 0,
        "env": {
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
        "extra_args": [
            "--width=640",
            "--height=640",
            "--num-frames=1",
            "--guidance-scale=1.0",
            "--num-gpus=4",
            "--tp-size=4",
            "--performance-mode=manual",
        ],
    },
    "ltx25": {
        "path": "Lightricks/LTX-2.5-Diffusers",
        "prompt": "A cat and a dog baking a cake together in a kitchen.",
        "extra_args": [
            "--pipeline-class-name=LTX2Pipeline",
            "--width=960",
            "--height=544",
            "--num-frames=121",
            "--fps=24",
            "--num-inference-steps=8",
            "--guidance-scale=1.0",
            "--performance-mode=manual",
        ],
    },
    "ltx25-diffusion-decoder": {
        "path": "Lightricks/LTX-2.5-Diffusers",
        "prompt": "A cat and a dog baking a cake together in a kitchen.",
        "extra_args": [
            "--pipeline-class-name=LTX2Pipeline",
            "--width=960",
            "--height=544",
            "--num-frames=121",
            "--fps=24",
            "--num-inference-steps=8",
            "--guidance-scale=1.0",
            "--use-diffusion-decoder",
            "--performance-mode=manual",
        ],
    },
    "ltx2": {
        "path": "Lightricks/LTX-2",
        "prompt": "A cat and a dog baking a cake together in a kitchen.",
        "extra_args": [
            "--pipeline-class-name=LTX2TwoStagePipeline",
            "--width=768",
            "--height=512",
            "--num-frames=121",
            "--num-gpus=2",
            "--enable-cfg-parallel",
        ],
    },
    "qwen-image": {
        "path": "Qwen/Qwen-Image",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "qwen-edit-2509": {
        "path": "Qwen/Qwen-Image-Edit-2509",
        "prompt": "Make the cat wear a red hat",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    "zimage-base": {
        "path": "Tongyi-MAI/Z-Image",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    "flux2-klein": {
        "path": "black-forest-labs/FLUX.2-klein-4B",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--dit-layerwise-offload",
            "false",
        ],
    },
    "flux2-klein-base": {
        "path": "black-forest-labs/FLUX.2-klein-base-4B",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--dit-layerwise-offload",
            "false",
        ],
    },
    "cosmos3-nano-t2i": {
        "path": "nvidia/Cosmos3-Nano",
        "prompt": "A red cube on a white table, product photo.",
        "env": {
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-frames=1",
            "--num-inference-steps=35",
        ],
    },
    "cosmos3-nano-t2v": {
        "path": "nvidia/Cosmos3-Nano",
        "prompt": "A blue box slides across a clean warehouse floor.",
        "env": {
            "SGLANG_DISABLE_COSMOS3_GUARDRAILS": "1",
        },
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=9",
            "--num-inference-steps=4",
        ],
    },
    "ernie-image-turbo": {
        "path": "baidu/ERNIE-Image-Turbo",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    "glm-image": {
        "path": "zai-org/GLM-Image",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    "sana-1.5-1.6b": {
        "path": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
        ],
    },
    "fastwan22-ti2v-5b": {
        "path": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        "prompt": "The cat starts walking slowly towards the camera.",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1280",
            "--height=720",
            "--num-frames=81",
        ],
    },
    # Blackwell-only ModelOpt NVFP4 comparator.
    "wan22-t2v-nvfp4": {
        "path": "nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4",
        "prompt": "A cat and a dog baking a cake together in a kitchen.",
        "extra_args": [
            "--width=832",
            "--height=480",
            "--num-frames=81",
            "--performance-mode=manual",
            "--dit-layerwise-offload=false",
            "--dit-cpu-offload=false",
        ],
    },
    "ltx23-hq-two-stage": {
        "path": "Lightricks/LTX-2.3",
        "prompt": "A beautiful sunset over the ocean",
        "env": {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
        "extra_args": [
            "--pipeline-class-name=LTX2TwoStageHQPipeline",
            "--ltx2-two-stage-device-mode=original",
            "--width=1920",
            "--height=1088",
            "--num-frames=121",
        ],
    },
    # Skill-only extra preset
    "ltx23-one-stage": {
        "path": "Lightricks/LTX-2.3",
        "prompt": "A beautiful sunset over the ocean",
        "negative_prompt": "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static.",
        "seed": 1234,
        "extra_args": [
            "--width=768",
            "--height=512",
            "--num-frames=121",
            "--fps=24",
            "--num-inference-steps=30",
            "--guidance-scale=3.0",
            "--num-gpus=2",
        ],
    },
    # Skill-only extra preset
    "ltx23-two-stage": {
        "path": "Lightricks/LTX-2.3",
        "prompt": "A beautiful sunset over the ocean",
        "negative_prompt": "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static.",
        "seed": 1234,
        "extra_args": [
            "--pipeline-class-name=LTX2TwoStagePipeline",
            "--width=1536",
            "--height=1024",
            "--num-frames=121",
            "--fps=24",
            "--num-inference-steps=30",
            "--guidance-scale=3.0",
            "--num-gpus=2",
        ],
    },
    # Skill-only extra preset
    "ltx23-two-stage-cfg-parallel": {
        "path": "Lightricks/LTX-2.3",
        "prompt": "A beautiful sunset over the ocean",
        "negative_prompt": "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static.",
        "seed": 1234,
        "extra_args": [
            "--pipeline-class-name=LTX2TwoStagePipeline",
            "--width=1536",
            "--height=1024",
            "--num-frames=121",
            "--fps=24",
            "--num-inference-steps=30",
            "--guidance-scale=3.0",
            "--num-gpus=2",
            "--cfg-parallel-size=2",
        ],
    },
    # Skill-only extra preset
    "hunyuanvideo": {
        "path": "hunyuanvideo-community/HunyuanVideo",
        "prompt": "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.",
        "extra_args": [
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
            "--num-frames=65",
            "--width=960",
            "--height=544",
            "--num-inference-steps=30",
        ],
    },
    # Skill-only extra presets
    # Require: <repo>/inputs/diffusion_benchmark/figs/mova_single_person.jpg
    "mova-360p": {
        "path": "OpenMOSS-Team/MOVA-360p",
        "prompt": 'A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, "I would also believe that this advance in AI recently was not unexpected."',
        "image_path": str(ASSET_DIR / "mova_single_person.jpg"),
        "extra_args": [
            "--adjust-frames=false",
            "--num-gpus=2",
            "--ulysses-degree=2",
            "--num-frames=193",
            "--fps=24",
            "--num-inference-steps=2",
        ],
    },
    "mova-720p": {
        "path": "OpenMOSS-Team/MOVA-720p",
        "prompt": 'A man in a blue blazer and glasses speaks in a formal indoor setting, framed by wooden furniture and a filled bookshelf. Quiet room acoustics underscore his measured tone as he delivers his remarks. At one point, he says, "I would also believe that this advance in AI recently was not unexpected."',
        "image_path": str(ASSET_DIR / "mova_single_person.jpg"),
        "extra_args": [
            "--adjust-frames=false",
            "--num-gpus=4",
            "--ring-degree=1",
            "--ulysses-degree=4",
            "--num-frames=193",
            "--fps=24",
            "--num-inference-steps=2",
        ],
    },
    # Skill-only extra preset
    "helios": {
        "path": "BestWishYsh/Helios-Base",
        "prompt": "A curious raccoon",
        "extra_args": [
            "--width=640",
            "--height=384",
            "--num-frames=33",
            "--dit-layerwise-offload",
            "false",
            "--dit-cpu-offload",
            "false",
            "--text-encoder-cpu-offload",
            "false",
            "--vae-cpu-offload",
            "false",
        ],
    },
    # Skill-only extra preset
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "joyai-edit": {
        "path": "jdopensource/JoyAI-Image-Edit-Diffusers",
        "prompt": "Make the cat wear a red hat",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=40",
            "--guidance-scale=4.0",
            "--dit-layerwise-offload",
            "false",
            "--dit-cpu-offload",
            "false",
            "--num-gpus=2",
            "--enable-cfg-parallel",
            "--ulysses-degree=1",
        ],
    },
    # Skill-only extra preset
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "firered-edit-1.0": {
        "path": "FireRedTeam/FireRed-Image-Edit-1.0",
        "prompt": "Make the cat wear a red hat",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=40",
            "--guidance-scale=4.0",
            "--dit-layerwise-offload",
            "false",
            "--dit-cpu-offload",
            "false",
            "--num-gpus=2",
            "--enable-cfg-parallel",
            "--ulysses-degree=1",
        ],
    },
    # Skill-only extra preset
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "firered-edit-1.1": {
        "path": "FireRedTeam/FireRed-Image-Edit-1.1",
        "prompt": "Make the cat wear a red hat",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=40",
            "--guidance-scale=4.0",
            "--dit-layerwise-offload",
            "false",
            "--dit-cpu-offload",
            "false",
            "--num-gpus=2",
            "--enable-cfg-parallel",
            "--ulysses-degree=1",
        ],
    },
    # Skill-only extra preset
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "hunyuan3d-shape": {
        "path": "tencent/Hunyuan3D-2",
        "prompt": "generate 3d mesh",
        "image_path": str(ASSET_DIR / "cat.png"),
        "config_overrides": {
            "paint_enable": False,
        },
        "extra_args": [
            "--num-inference-steps=50",
            "--guidance-scale=5.0",
            "--dit-layerwise-offload",
            "false",
            "--dit-cpu-offload",
            "false",
        ],
    },
}


def required_gpus_for_model(model_key: str) -> int:
    parsed_args = _parse_cli_args(MODELS[model_key].get("extra_args", []))
    if "num-gpus" in parsed_args:
        return int(parsed_args["num-gpus"])
    if model_key in {"wan-t2v", "wan-i2v"}:
        return 4
    if model_key == "mova-720p":
        return 4
    if model_key in {
        "ltx2",
        "ltx23-ti2v-two-stage",
        "ltx23-one-stage",
        "ltx23-two-stage",
        "ltx23-two-stage-cfg-parallel",
        "joyai-edit",
        "firered-edit-1.0",
        "firered-edit-1.1",
    }:
        return 2
    return 1


def model_nightly_case_id(model_key: str) -> str:
    return MODELS[model_key].get("nightly_case_id", "-")


def _safe_cache_component(value: str) -> str:
    component = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in value
    ).strip(".")
    if not component:
        raise ValueError(f"Cannot derive a cache directory name from {value!r}")
    return component


def _resolve_seed_hub_cache(seed_root: Path) -> Path:
    seed_root = seed_root.expanduser().resolve()
    hub_root = seed_root / "hub"
    if hub_root.is_dir():
        return hub_root
    if seed_root.name == "hub" and seed_root.is_dir():
        return seed_root
    raise FileNotFoundError(
        "A seed model cache must be either a Hugging Face home containing "
        f"hub/ or the hub directory itself: {seed_root}"
    )


def _seed_hub_entry(source_entry: Path, target_entry: Path) -> None:
    """Build a writable cache overlay without copying immutable payloads."""
    if not source_entry.is_dir():
        if not target_entry.exists() and not target_entry.is_symlink():
            target_entry.symlink_to(source_entry.resolve())
        return

    target_entry.mkdir(exist_ok=True)
    for source_path in sorted(source_entry.rglob("*")):
        relative_path = source_path.relative_to(source_entry)
        target_path = target_entry / relative_path
        if target_path.exists() or target_path.is_symlink():
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.is_symlink():
            target_path.symlink_to(
                os.readlink(source_path), target_is_directory=source_path.is_dir()
            )
        elif source_path.is_dir():
            target_path.mkdir()
        elif relative_path.parts[0] in {"refs", "trees"}:
            shutil.copy2(source_path, target_path)
        else:
            target_path.symlink_to(source_path.resolve())


def _seed_model_cache(cache_dir: Path, seed_roots: list[Path]) -> None:
    """Expose read-only Hugging Face caches through copy-on-write overlays."""
    target_hub = cache_dir / "huggingface" / "hub"
    target_hub.mkdir(parents=True, exist_ok=True)
    (target_hub / ".locks").mkdir()

    for seed_root in seed_roots:
        source_hub = _resolve_seed_hub_cache(seed_root)
        if source_hub == target_hub or target_hub in source_hub.parents:
            raise ValueError(
                f"Refusing to seed the isolated cache from itself: {source_hub}"
            )
        for source_entry in sorted(source_hub.iterdir()):
            if source_entry.name == ".locks":
                continue
            target_entry = target_hub / source_entry.name
            _seed_hub_entry(source_entry, target_entry)


def _prepare_model_cache(
    cache_root: Path,
    model_key: str,
    label: str,
    seed_model_cache_roots: list[Path] | None = None,
) -> Path:
    cache_root = cache_root.expanduser().resolve()
    unsafe_roots = {Path("/"), Path.home().resolve(), REPO_ROOT.resolve()}
    if cache_root in unsafe_roots:
        raise ValueError(
            "Refusing to use a broad or shared directory as the isolated model "
            f"cache root: {cache_root}"
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    marker = cache_root / MODEL_CACHE_MARKER
    if not marker.exists():
        marker.write_text(
            "Owned by bench_diffusion_denoise.py. Only generated child caches "
            "may be removed.\n",
            encoding="utf-8",
        )

    cache_dir = cache_root / (
        f"{_safe_cache_component(model_key)}-{_safe_cache_component(label)}"
    )
    if cache_dir.exists():
        raise FileExistsError(
            "The isolated model cache already exists. Refusing to reuse or "
            f"delete it without inspection: {cache_dir}"
        )
    cache_dir.mkdir()
    if seed_model_cache_roots:
        _seed_model_cache(cache_dir, seed_model_cache_roots)
    return cache_dir


def _model_cache_env(cache_dir: Path) -> dict[str, str]:
    huggingface_root = cache_dir / "huggingface"
    huggingface_hub = huggingface_root / "hub"
    return {
        "HF_HOME": str(huggingface_root),
        "HF_ASSETS_CACHE": str(huggingface_root / "assets"),
        "HF_HUB_CACHE": str(huggingface_hub),
        "HF_MODULES_CACHE": str(huggingface_root / "modules"),
        "HF_XET_CACHE": str(huggingface_root / "xet"),
        "HUGGINGFACE_HUB_CACHE": str(huggingface_hub),
        "DIFFUSERS_CACHE": str(huggingface_hub),
        "TRANSFORMERS_CACHE": str(huggingface_hub),
        "MODELSCOPE_CACHE": str(cache_dir / "modelscope"),
        "MODELSCOPE_MODULES_CACHE": str(cache_dir / "modelscope" / "modules"),
    }


def _cache_stats(cache_dir: Path) -> dict[str, int]:
    file_count = 0
    weight_file_count = 0
    total_bytes = 0
    if not cache_dir.exists():
        return {
            "file_count": 0,
            "weight_file_count": 0,
            "total_bytes": 0,
        }

    for entry in cache_dir.rglob("*"):
        if not (entry.is_file() or entry.is_symlink()):
            continue
        file_count += 1
        total_bytes += entry.lstat().st_size
        if entry.suffix.lower() in MODEL_WEIGHT_SUFFIXES:
            weight_file_count += 1
    return {
        "file_count": file_count,
        "weight_file_count": weight_file_count,
        "total_bytes": total_bytes,
    }


def _cleanup_model_cache(
    cache_root: Path,
    cache_dir: Path,
    ledger_path: Path,
    model_key: str,
    label: str,
    exit_reason: str,
) -> dict[str, object]:
    cache_root = cache_root.expanduser().resolve()
    cache_dir = cache_dir.resolve()
    if not (cache_root / MODEL_CACHE_MARKER).is_file():
        raise RuntimeError(f"Missing isolated-cache ownership marker: {cache_root}")
    if cache_dir.parent != cache_root:
        raise RuntimeError(
            f"Refusing to remove cache outside the isolated root: {cache_dir}"
        )

    before = _cache_stats(cache_dir)
    shutil.rmtree(cache_dir)
    after = _cache_stats(cache_dir)
    record: dict[str, object] = {
        "model": model_key,
        "label": label,
        "exit_reason": exit_reason,
        "cache_dir": str(cache_dir),
        "cleaned_at_unix_s": time.time(),
        "before": before,
        "after": after,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _parse_cli_args(args: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if not isinstance(arg, str) or not arg.startswith("--"):
            i += 1
            continue
        if "=" in arg:
            key, value = arg[2:].split("=", 1)
            parsed[key] = value
        elif i + 1 < len(args) and not str(args[i + 1]).startswith("--"):
            parsed[arg[2:]] = str(args[i + 1])
            i += 1
        else:
            parsed[arg[2:]] = True
        i += 1
    return parsed


def _normalize_cli_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _expected_nightly_cli_args(case: dict) -> dict[str, str]:
    expected = {
        "width": str(case["width"]),
        "height": str(case["height"]),
    }

    for key, flag in (
        ("num_frames", "num-frames"),
        ("fps", "fps"),
        ("num_inference_steps", "num-inference-steps"),
        ("guidance_scale", "guidance-scale"),
    ):
        if key in case:
            expected[flag] = str(case[key])

    if case.get("num_gpus", 1) > 1:
        expected["num-gpus"] = str(case["num_gpus"])

    serve_args = shlex.split(case["frameworks"]["sglang"].get("serve_args", ""))
    parsed_serve_args = _parse_cli_args(serve_args)
    for flag, value in parsed_serve_args.items():
        # Nightly's comparison driver still owns its legacy ``--warmup``
        # switch.  It is not a valid ``sglang generate`` flag after the
        # warmup-mode migration, so exclude both spellings from preset drift
        # validation.
        if flag in {"warmup", "warmup-mode"}:
            continue
        expected[flag] = _normalize_cli_value(value)

    return expected


def validate_nightly_alignment() -> int:
    """Validate nightly presets against diffusion comparison_configs.json."""
    if not NIGHTLY_CONFIG_PATH.exists():
        print(f"Missing nightly config: {NIGHTLY_CONFIG_PATH}")
        return 1

    with open(NIGHTLY_CONFIG_PATH) as f:
        config = json.load(f)

    cases = {case["id"]: case for case in config["cases"]}
    errors: list[str] = []

    preset_case_ids = [
        MODELS[model_key].get("nightly_case_id") for model_key in NIGHTLY_PRESET_ORDER
    ]
    if preset_case_ids != list(cases):
        errors.append(
            "Nightly preset order differs from comparison_configs.json: "
            f"skill={preset_case_ids}, ci={list(cases)}"
        )

    for model_key in NIGHTLY_PRESET_ORDER:
        preset = MODELS[model_key]
        case_id = preset["nightly_case_id"]
        case = cases.get(case_id)
        if case is None:
            errors.append(f"{model_key}: missing CI case {case_id}")
            continue

        if preset["path"] != case["model"]:
            errors.append(f"{model_key}: model path differs")
        if preset["prompt"] != case["prompt"]:
            errors.append(f"{model_key}: prompt differs")
        if bool(preset.get("image_path")) != bool(case.get("reference_image")):
            errors.append(f"{model_key}: reference image presence differs")
        if preset.get("seed", 42) != case.get("seed"):
            errors.append(f"{model_key}: seed differs")
        if preset.get("env", {}) != case["frameworks"]["sglang"].get("extra_env", {}):
            errors.append(f"{model_key}: environment differs")

        if "config_overrides" in preset:
            expected_config = {
                key: value
                for key, value in case.get("sglang_request_extra", {}).items()
                if value is not None
            }
            if "num_inference_steps" in case:
                expected_config["num_inference_steps"] = case["num_inference_steps"]
            if preset["config_overrides"] != expected_config:
                errors.append(
                    f"{model_key}: generated request config differs\n"
                    f"  skill={preset['config_overrides']}\n"
                    f"  ci={expected_config}"
                )

        ignored_args = set(preset.get("nightly_cli_ignored", set()))
        actual_args = {
            key: _normalize_cli_value(value)
            for key, value in _parse_cli_args(preset["extra_args"]).items()
            if key not in ignored_args
        }
        expected_args = {
            key: value
            for key, value in _expected_nightly_cli_args(case).items()
            if key not in ignored_args
        }
        if actual_args != expected_args:
            errors.append(
                f"{model_key}: CLI args differ\n"
                f"  skill={actual_args}\n"
                f"  ci={expected_args}"
            )

    if errors:
        print("Nightly alignment check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Nightly alignment check passed: presets match "
        "scripts/ci/utils/diffusion/comparison_configs.json."
    )
    return 0


def print_model_catalog():
    """Print preset order, model path, and whether each preset maps to nightly."""
    print()
    print("=" * CATALOG_TABLE_WIDTH)
    print("MODEL PRESETS — Nightly-aligned, then current-source and skill-only extras")
    print("=" * CATALOG_TABLE_WIDTH)
    print(f"{'Preset':<32} {'Nightly':<30} {'Model Path':<66} {'GPUs':>4}")
    print("-" * CATALOG_TABLE_WIDTH)
    for model_key, cfg in MODELS.items():
        print(
            f"{model_key:<32} {model_nightly_case_id(model_key):<30} {cfg['path']:<66} {required_gpus_for_model(model_key):>4}"
        )
    print("-" * CATALOG_TABLE_WIDTH)
    print(
        "Nightly column shows the comparison_configs.json case id; '-' means no nightly mapping."
    )


def build_sglang_cmd(
    model_key: str,
    perf_dump_path: str | None = None,
    warmup: bool = True,
    torch_compile: bool = False,
    quality: str = "lossless",
    breakable_cuda_graph: bool = False,
    bcg_text_buckets: list[int] | None = None,
    seed: int = 42,
    save_output: bool = True,
    artifact_dir: Path | None = None,
) -> list[str]:
    """
    Build the `sglang generate` command for the given model.
    Matches the commands in benchmark-and-profile.md exactly.
    """
    if quality not in BENCHMARK_QUALITY_LEVELS:
        raise ValueError(
            f"quality must be one of {BENCHMARK_QUALITY_LEVELS}, got {quality!r}"
        )
    if torch_compile and breakable_cuda_graph:
        raise ValueError("torch.compile and breakable CUDA graph are comparators")
    if bcg_text_buckets is not None:
        if not breakable_cuda_graph:
            raise ValueError("bcg_text_buckets requires breakable_cuda_graph=True")
        if not bcg_text_buckets or any(bucket <= 0 for bucket in bcg_text_buckets):
            raise ValueError("bcg_text_buckets must contain positive integers")

    cfg = MODELS[model_key]

    cmd = [
        "sglang",
        "generate",
        "--backend=sglang",
        f"--model-path={cfg['path']}",
        f"--prompt={cfg['prompt']}",
    ]

    effective_seed = cfg.get("seed", seed)
    if effective_seed is not None:
        cmd.append(f"--seed={effective_seed}")

    if "negative_prompt" in cfg:
        cmd.append(f"--negative-prompt={cfg['negative_prompt']}")

    if "image_path" in cfg:
        cmd.append(f"--image-path={cfg['image_path']}")

    if "config_overrides" in cfg:
        config_root = (
            Path(artifact_dir)
            if artifact_dir is not None
            else get_output_dir("benchmarks", REPO_ROOT)
        )
        config_dir = ensure_dir(config_root / "generated_configs")
        config_path = config_dir / f"{model_key}.json"
        with open(config_path, "w") as f:
            json.dump(cfg["config_overrides"], f, indent=2, sort_keys=True)
        cmd.append(f"--config={config_path}")

    cmd.extend(cfg["extra_args"])
    cmd.append(f"--quality={quality}")

    if save_output:
        cmd.append("--save-output")
    if warmup:
        cmd.extend(["--warmup-mode", "request"])
    if breakable_cuda_graph:
        cmd.append("--enable-breakable-cuda-graph")
        parsed_args = _parse_cli_args(cmd)
        if "warmup-resolutions" not in parsed_args:
            warmup_resolutions = cfg.get("bcg_warmup_resolutions")
            if warmup_resolutions is None and all(
                name in parsed_args for name in ("width", "height")
            ):
                warmup_resolutions = [f"{parsed_args['width']}x{parsed_args['height']}"]
            if warmup_resolutions:
                cmd.append("--warmup-resolutions")
                cmd.extend(warmup_resolutions)
        if "warmup-num-frames" not in parsed_args and "num-frames" in parsed_args:
            cmd.extend(["--warmup-num-frames", str(parsed_args["num-frames"])])
        if bcg_text_buckets is not None:
            cmd.append("--bcg-text-buckets")
            cmd.extend(str(bucket) for bucket in bcg_text_buckets)
    if torch_compile and not cfg.get("force_eager", False):
        cmd.append("--enable-torch-compile")
    if perf_dump_path:
        cmd.extend(["--perf-dump-path", perf_dump_path])

    return cmd


def _run_benchmark_once_impl(
    model_key: str,
    label: str,
    output_dir: Path,
    warmup: bool = True,
    torch_compile: bool = False,
    quality: str = "lossless",
    breakable_cuda_graph: bool = False,
    bcg_text_buckets: list[int] | None = None,
    model_cache_dir: Path | None = None,
    cuda_visible_devices: str | None = None,
) -> dict:
    """Run a single benchmark pass and return results dict."""
    perf_path = output_dir / f"{model_key}_{label}.json"

    cmd = build_sglang_cmd(
        model_key,
        perf_dump_path=str(perf_path),
        warmup=warmup,
        torch_compile=torch_compile,
        quality=quality,
        breakable_cuda_graph=breakable_cuda_graph,
        bcg_text_buckets=bcg_text_buckets,
        artifact_dir=output_dir,
    )
    output_file_name = f"{model_key}-{label}"
    cmd.extend(
        ["--output-path", str(output_dir), "--output-file-name", output_file_name]
    )

    env = os.environ.copy()
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    # Perf dumps are consumed as stage-attributed denoise measurements.  Drain
    # the device queue at stage boundaries so asynchronous denoise work cannot
    # leak into a later stage (most visibly DecodingStage).  An explicit 0 in
    # the caller's environment still opts out for e2e-only experiments.
    env.setdefault("SGLANG_DIFFUSION_SYNC_STAGE_PROFILING", "1")
    cfg = MODELS[model_key]
    for key, value in cfg.get("env", {}).items():
        env.setdefault(key, str(value))
    if model_cache_dir is not None:
        env.update(_model_cache_env(model_cache_dir))
    if env.get("HF_TOKEN") and not env.get("HUGGINGFACE_HUB_TOKEN"):
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]

    if model_key in GATED_MODELS and not (
        env.get("HF_TOKEN") or env.get("HUGGINGFACE_HUB_TOKEN")
    ):
        print(f"\n{'=' * 64}")
        print(f"[{label.upper()}] {model_key}")
        print("  ERROR: this preset uses a gated Hugging Face repo.")
        print("  Export HF_TOKEN before running it, for example:")
        print("    export HF_TOKEN=<your_hf_token>")
        print("  Without a token, the top-level `sglang generate` model detection may")
        print("  fail early and report a misleading unsupported-model error.")
        return {"model": model_key, "label": label, "error": True, "elapsed_s": 0.0}

    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    elif not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in pick_idle_gpus(required_gpus_for_model(model_key))
        )

    print(f"\n{'=' * 64}")
    print(f"[{label.upper()}] {model_key}")
    print(f"  CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print("  " + " \\\n  ".join(cmd))
    print()

    t0 = time.time()
    process = subprocess.Popen(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    fallback_detected = False
    bcg_capture_detected = False
    bcg_invalid_signals: set[str] = set()
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="")
            lower_line = line.lower()
            if any(signal in lower_line for signal in DIFFUSERS_FALLBACK_SIGNALS):
                fallback_detected = True
            if BCG_CAPTURE_SIGNAL in lower_line:
                bcg_capture_detected = True
            if (
                quality == "high"
                and breakable_cuda_graph
                and bcg_capture_detected
                and "mounted " in lower_line
                and "for quality=high" in lower_line
            ):
                bcg_invalid_signals.add(BCG_LATE_QUALITY_FUSION_SIGNAL)
            bcg_invalid_signals.update(
                signal for signal in BCG_INVALID_SIGNALS if signal in lower_line
            )
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    returncode = process.wait()
    elapsed = time.time() - t0

    if fallback_detected:
        print(
            "  ERROR: model fell back to the diffusers backend. "
            "Fix native SGLang diffusion backend selection before collecting perf data."
        )
        return {"model": model_key, "label": label, "error": True, "elapsed_s": elapsed}

    if breakable_cuda_graph and (not bcg_capture_detected or bcg_invalid_signals):
        reason = (
            ", ".join(sorted(bcg_invalid_signals))
            if bcg_invalid_signals
            else "no '[Diffusion BCG] captured' marker"
        )
        print(
            "  ERROR: BCG evidence is invalid: "
            f"{reason}. Do not report this run as BCG performance."
        )
        return {
            "model": model_key,
            "label": label,
            "quality": quality,
            "breakable_cuda_graph": True,
            "bcg_capture_detected": bcg_capture_detected,
            "bcg_invalid_signals": sorted(bcg_invalid_signals),
            "error": True,
            "elapsed_s": elapsed,
        }

    if returncode != 0:
        print(f"  ERROR: exit code {returncode}")
        return {"model": model_key, "label": label, "error": True, "elapsed_s": elapsed}

    output_artifacts = sorted(
        path
        for path in output_dir.rglob(f"{output_file_name}*")
        if path.is_file() and path.suffix.lower() in GENERATED_OUTPUT_SUFFIXES
    )
    missing_artifacts = []
    if not perf_path.is_file():
        missing_artifacts.append("perf dump")
    if not output_artifacts:
        missing_artifacts.append("generated output")
    if missing_artifacts:
        print(
            "  ERROR: command returned zero without required benchmark artifacts: "
            + ", ".join(missing_artifacts)
        )
        return {
            "model": model_key,
            "label": label,
            "quality": quality,
            "breakable_cuda_graph": breakable_cuda_graph,
            "missing_artifacts": missing_artifacts,
            "error": True,
            "elapsed_s": elapsed,
        }

    metrics = {
        "model": model_key,
        "label": label,
        "quality": quality,
        "breakable_cuda_graph": breakable_cuda_graph,
        "bcg_capture_detected": bcg_capture_detected,
        "elapsed_s": elapsed,
        "output_artifacts": [str(path) for path in output_artifacts],
        "output_sha256": [_sha256_file(path) for path in output_artifacts],
        "error": False,
    }
    if perf_path.exists():
        try:
            with open(perf_path) as f:
                perf = json.load(f)

            # e2e latency: total_duration_ms (set by PerformanceLogger.dump_benchmark_report)
            total_ms = perf.get("total_duration_ms")
            metrics["e2e_latency_s"] = (
                float(total_ms) / 1000.0 if total_ms is not None else None
            )

            # denoise latency: sum all true denoise/refinement stages.
            # This accepts variants such as "MOVADenoisingStage",
            # "HeliosChunkedDenoisingStage", and the LTX-2 two-stage pair
            # "LTX2AVDenoisingStage" + "LTX2RefinementStage", while excluding
            # setup stages like "QwenImageLayeredBeforeDenoisingStage".
            denoise_latency_s = None
            denoise_stage_total_ms = 0.0
            for step in perf.get("steps", []):
                step_name = step.get("name")
                if (
                    isinstance(step_name, str)
                    and step.get("duration_ms") is not None
                    and step_name.endswith(("DenoisingStage", "RefinementStage"))
                    and "BeforeDenoisingStage" not in step_name
                ):
                    denoise_stage_total_ms += float(step["duration_ms"])

            if denoise_stage_total_ms > 0.0:
                denoise_latency_s = denoise_stage_total_ms / 1000.0

            # fallback: sum all per-step durations from denoise_steps_ms
            # denoise_steps_ms = [{"step": 0, "duration_ms": 100.5}, ...]
            if denoise_latency_s is None:
                denoise_steps = perf.get("denoise_steps_ms", [])
                if denoise_steps:
                    denoise_latency_s = (
                        sum(s.get("duration_ms", 0.0) for s in denoise_steps) / 1000.0
                    )
            metrics["denoise_latency_s"] = denoise_latency_s

            # peak memory: max peak_reserved_mb across all memory checkpoints (→ GB)
            # memory_checkpoints = {"after_DenoisingStage": {"peak_reserved_mb": 12288.0, ...}}
            peak_memory_gb = None
            for snapshot in perf.get("memory_checkpoints", {}).values():
                peak_mb = snapshot.get("peak_reserved_mb")
                if peak_mb is not None:
                    candidate = float(peak_mb) / 1024.0
                    if peak_memory_gb is None or candidate > peak_memory_gb:
                        peak_memory_gb = candidate
            metrics["peak_memory_gb"] = peak_memory_gb

        except (AttributeError, OSError, TypeError, ValueError) as e:
            print(f"  Warning: could not parse perf dump: {e}")

    return metrics


def _validate_quality_bcg_output_hashes(results: list[dict]) -> None:
    """Reject BCG rows whose generated artifacts differ from eager."""
    for quality in BENCHMARK_QUALITY_LEVELS:
        quality_results = [
            result for result in results if result.get("quality") == quality
        ]
        eager_results = [
            result
            for result in quality_results
            if not result.get("breakable_cuda_graph") and not result.get("error")
        ]
        eager_hashes = [
            tuple(result.get("output_sha256", ())) for result in eager_results
        ]
        if not eager_hashes or any(not hashes for hashes in eager_hashes):
            continue

        reference_hashes = eager_hashes[0]
        if any(hashes != reference_hashes for hashes in eager_hashes[1:]):
            reason = f"eager {quality} output hashes are unstable"
            for result in quality_results:
                result["error"] = True
                result["output_hash_error"] = reason
            print(f"  ERROR: {reason}; do not use this matrix as BCG evidence.")
            continue

        for result in quality_results:
            if not result.get("breakable_cuda_graph") or result.get("error"):
                continue
            output_hashes = tuple(result.get("output_sha256", ()))
            if not output_hashes or output_hashes == reference_hashes:
                continue
            reason = f"BCG {quality} output hash differs from eager"
            result["error"] = True
            result["output_hash_error"] = reason
            print(f"  ERROR: {reason}; do not report this row as BCG performance.")


def run_benchmark_once(
    model_key: str,
    label: str,
    output_dir: Path,
    warmup: bool = True,
    torch_compile: bool = False,
    quality: str = "lossless",
    breakable_cuda_graph: bool = False,
    bcg_text_buckets: list[int] | None = None,
    model_cache_root: Path | None = None,
    seed_model_cache_roots: list[Path] | None = None,
    cleanup_model_cache: bool = False,
    cleanup_ledger_path: Path | None = None,
) -> dict:
    """Run one preset and optionally clean its task-owned model cache."""
    cache_dir = None
    exit_reason = "error"
    if model_cache_root is not None:
        cache_dir = _prepare_model_cache(
            model_cache_root,
            model_key,
            label,
            seed_model_cache_roots=seed_model_cache_roots,
        )

    try:
        result = _run_benchmark_once_impl(
            model_key,
            label,
            output_dir,
            warmup=warmup,
            torch_compile=torch_compile,
            quality=quality,
            breakable_cuda_graph=breakable_cuda_graph,
            bcg_text_buckets=bcg_text_buckets,
            model_cache_dir=cache_dir,
        )
        exit_reason = "error" if result.get("error") else "success"
        return result
    except KeyboardInterrupt:
        exit_reason = "interrupted"
        raise
    finally:
        if cleanup_model_cache and cache_dir is not None:
            assert model_cache_root is not None
            ledger_path = cleanup_ledger_path or output_dir / "cleanup.jsonl"
            record = _cleanup_model_cache(
                model_cache_root,
                cache_dir,
                ledger_path,
                model_key,
                label,
                exit_reason,
            )
            before = record["before"]
            assert isinstance(before, dict)
            print(
                "  Cleaned isolated model cache: "
                f"{before['total_bytes']} bytes, "
                f"{before['weight_file_count']} weight files; ledger={ledger_path}"
            )


def run_quality_bcg_matrix(
    model_key: str,
    label: str,
    output_dir: Path,
    warmup: bool = True,
    bcg_text_buckets: list[int] | None = None,
    model_cache_root: Path | None = None,
    seed_model_cache_roots: list[Path] | None = None,
    cleanup_model_cache: bool = False,
    cleanup_ledger_path: Path | None = None,
) -> list[dict]:
    """Run the quality/BCG applicability matrix on one fixed GPU set.

    A high+BCG cell is intentionally retained as a compatibility check. It is
    invalid when request-scoped DiT fusions mount after graph capture.
    """
    cache_dir = None
    exit_reason = "error"
    if model_cache_root is not None:
        cache_dir = _prepare_model_cache(
            model_cache_root,
            model_key,
            f"{label}-quality-bcg-matrix",
            seed_model_cache_roots=seed_model_cache_roots,
        )

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        cuda_visible_devices = ",".join(
            str(index) for index in pick_idle_gpus(required_gpus_for_model(model_key))
        )

    results: list[dict] = []
    try:
        for mode_label, quality, breakable_cuda_graph in QUALITY_BCG_ABBA_MATRIX:
            result = _run_benchmark_once_impl(
                model_key,
                f"{label}-{mode_label}",
                output_dir,
                warmup=warmup,
                quality=quality,
                breakable_cuda_graph=breakable_cuda_graph,
                bcg_text_buckets=(bcg_text_buckets if breakable_cuda_graph else None),
                model_cache_dir=cache_dir,
                cuda_visible_devices=cuda_visible_devices,
            )
            results.append(result)
        _validate_quality_bcg_output_hashes(results)
        exit_reason = (
            "error" if any(result.get("error") for result in results) else "success"
        )
        return results
    except KeyboardInterrupt:
        exit_reason = "interrupted"
        raise
    finally:
        if cleanup_model_cache and cache_dir is not None:
            assert model_cache_root is not None
            ledger_path = cleanup_ledger_path or output_dir / "cleanup.jsonl"
            record = _cleanup_model_cache(
                model_cache_root,
                cache_dir,
                ledger_path,
                model_key,
                f"{label}-quality-bcg-matrix",
                exit_reason,
            )
            before = record["before"]
            assert isinstance(before, dict)
            print(
                "  Cleaned isolated model cache after the full matrix: "
                f"{before['total_bytes']} bytes, "
                f"{before['weight_file_count']} weight files; ledger={ledger_path}"
            )


def print_results_table(results: list[dict]):
    """Print a compact table for one or more benchmark runs."""
    print()
    print("=" * RESULTS_TABLE_WIDTH)
    print("BENCHMARK RESULTS — Denoise Latency (primary metric ★)")
    print("(Models and params match benchmark-and-profile.md)")
    print("=" * RESULTS_TABLE_WIDTH)

    print(
        f"{'Model':<24} {'Nightly':<28} {'Label':<31} {'Denoise(s)':>12} {'E2E(s)':>10} {'Peak Mem(GB)':>14}"
    )
    print("-" * RESULTS_TABLE_WIDTH)

    for result in results:
        denoise_s = result.get("denoise_latency_s")
        e2e_s = result.get("e2e_latency_s")
        peak_mem = result.get("peak_memory_gb")
        denoise_text = f"{denoise_s:.2f}" if isinstance(denoise_s, float) else "n/a"
        e2e_text = f"{e2e_s:.2f}" if isinstance(e2e_s, float) else "n/a"
        mem_text = f"{peak_mem:.1f}" if isinstance(peak_mem, float) else "n/a"
        print(
            f"{result['model']:<24} {model_nightly_case_id(result['model']):<28} {result['label']:<31} {denoise_text:>12} {e2e_text:>10} {mem_text:>14}"
        )

    print("-" * RESULTS_TABLE_WIDTH)
    print()
    print(
        "★ Denoise latency = sum of stages ending with DenoisingStage plus any RefinementStage."
    )
    print(
        "  Compare two runs with python/sglang/multimodal_gen/benchmarks/compare_perf.py."
    )


def main():
    parser = argparse.ArgumentParser(
        description="SGLang Diffusion denoise benchmark preset runner"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model to benchmark (default: flux)",
    )
    parser.add_argument(
        "--all", action="store_true", help=f"Benchmark all {len(MODELS)} models"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List preset order, nightly mapping, and exit",
    )
    parser.add_argument(
        "--validate-nightly-alignment",
        action="store_true",
        help="Validate nightly presets against scripts/ci/utils/diffusion/comparison_configs.json and exit.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="baseline",
        help="Result label and perf dump suffix (e.g. baseline, tuned, pr20962).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(get_output_dir("benchmarks", REPO_ROOT)),
        help="Directory for perf dump JSON files",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    parser.add_argument(
        "--quality",
        choices=BENCHMARK_QUALITY_LEVELS,
        default="lossless",
        help="Request quality for a single run (default: lossless).",
    )
    parser.add_argument(
        "--breakable-cuda-graph",
        action="store_true",
        help=(
            "Run a BCG comparator. The result is invalid unless capture is "
            "observed and no disable/failure/signature-miss marker appears."
        ),
    )
    parser.add_argument(
        "--bcg-text-buckets",
        type=int,
        nargs="+",
        help="Optional positive text buckets for a BCG run or matrix.",
    )
    parser.add_argument(
        "--quality-bcg-matrix",
        action="store_true",
        help=(
            "Run lossless/high Eager-vs-BCG as two ABBA pairs on one GPU set "
            "and one task-owned model cache."
        ),
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--torch-compile",
        action="store_true",
        help="Opt in to a torch.compile comparison. Presets run eager by default.",
    )
    compile_group.add_argument(
        "--no-torch-compile",
        action="store_true",
        help="Deprecated compatibility flag; eager is already the default.",
    )
    parser.add_argument(
        "--model-cache-root",
        type=str,
        help=(
            "Create a new isolated Hugging Face/ModelScope cache below this "
            "directory for each model run."
        ),
    )
    parser.add_argument(
        "--cleanup-model-cache",
        action="store_true",
        help=(
            "Remove the task-owned model cache in a finally block and append "
            "a cleanup ledger record. Requires --model-cache-root."
        ),
    )
    parser.add_argument(
        "--seed-model-cache-root",
        action="append",
        default=[],
        help=(
            "Seed each isolated cache with a copy-on-write overlay from this "
            "read-only Hugging Face home or hub directory. May be repeated."
        ),
    )
    parser.add_argument(
        "--cleanup-ledger",
        type=str,
        help="JSONL cleanup ledger path (default: <output-dir>/cleanup.jsonl).",
    )

    args = parser.parse_args()

    if args.list_models:
        print_model_catalog()
        return

    if args.validate_nightly_alignment:
        raise SystemExit(validate_nightly_alignment())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup = not args.no_warmup
    torch_compile = args.torch_compile and not args.no_torch_compile
    if args.quality_bcg_matrix and torch_compile:
        parser.error("--quality-bcg-matrix cannot be combined with --torch-compile")
    if args.breakable_cuda_graph and torch_compile:
        parser.error("--breakable-cuda-graph cannot be combined with --torch-compile")
    if args.bcg_text_buckets and not (
        args.breakable_cuda_graph or args.quality_bcg_matrix
    ):
        parser.error(
            "--bcg-text-buckets requires --breakable-cuda-graph or "
            "--quality-bcg-matrix"
        )
    if args.cleanup_model_cache and not args.model_cache_root:
        parser.error("--cleanup-model-cache requires --model-cache-root")
    if args.seed_model_cache_root and not args.model_cache_root:
        parser.error("--seed-model-cache-root requires --model-cache-root")
    model_cache_root = (
        Path(args.model_cache_root) if args.model_cache_root is not None else None
    )
    seed_model_cache_roots = [Path(path) for path in args.seed_model_cache_root]
    cleanup_ledger_path = (
        Path(args.cleanup_ledger) if args.cleanup_ledger is not None else None
    )

    models_to_run = list(MODELS.keys()) if args.all else [args.model or "flux"]
    results = []

    for model_key in models_to_run:
        if args.quality_bcg_matrix:
            results.extend(
                run_quality_bcg_matrix(
                    model_key,
                    args.label,
                    output_dir,
                    warmup=warmup,
                    bcg_text_buckets=args.bcg_text_buckets,
                    model_cache_root=model_cache_root,
                    seed_model_cache_roots=seed_model_cache_roots,
                    cleanup_model_cache=args.cleanup_model_cache,
                    cleanup_ledger_path=cleanup_ledger_path,
                )
            )
        else:
            results.append(
                run_benchmark_once(
                    model_key,
                    args.label,
                    output_dir,
                    warmup=warmup,
                    torch_compile=torch_compile,
                    quality=args.quality,
                    breakable_cuda_graph=args.breakable_cuda_graph,
                    bcg_text_buckets=args.bcg_text_buckets,
                    model_cache_root=model_cache_root,
                    seed_model_cache_roots=seed_model_cache_roots,
                    cleanup_model_cache=args.cleanup_model_cache,
                    cleanup_ledger_path=cleanup_ledger_path,
                )
            )

    if results:
        print_results_table(results)

    print(f"Perf dump JSONs → {output_dir}")
    print(
        "Compare across runs: follow benchmark-and-profile.md -> Perf dump & before/after compare."
    )


if __name__ == "__main__":
    main()
