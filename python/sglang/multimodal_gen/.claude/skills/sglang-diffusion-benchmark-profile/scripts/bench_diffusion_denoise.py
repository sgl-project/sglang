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

    # All 20 preset models
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
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diffusion_skill_env import (  # noqa: E402
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
GATED_MODELS = {"flux", "flux2"}
DIFFUSERS_FALLBACK_SIGNALS = (
    "falling back to diffusers backend",
    "using diffusers backend",
    "loaded diffusers pipeline",
)
CATALOG_TABLE_WIDTH = 105
RESULTS_TABLE_WIDTH = 105
NIGHTLY_PRESET_ORDER = (
    "flux",
    "flux2",
    "qwen",
    "qwen-edit",
    "zimage",
    "wan-t2v",
    "wan-ti2v",
    "ltx2",
    "ltx23-ti2v-two-stage",
    "wan-i2v",
)

# ---------------------------------------------------------------------------
# Model configs — kept in exact sync with benchmark-and-profile.md
# Nightly-aligned presets mirror scripts/ci/utils/diffusion/comparison_configs.json
# first, followed by skill-only extras.
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
            "--dit-layerwise-offload",
            "false",
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
            "--dit-layerwise-offload",
            "false",
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
    # 8. Nightly: ltx2_twostage_t2v
    "ltx2": {
        "nightly_case_id": "ltx2_twostage_t2v",
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
    # 9. Nightly: ltx2.3_twostage_ti2v_2gpus
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
    # 10. Nightly: wan22_i2v_a14b_720p
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
    # 11. Skill-only extra preset
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
    # 12. Skill-only extra preset
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
    # 13. Skill-only extra preset
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
    # 14. Skill-only extra preset
    "hunyuanvideo": {
        "path": "hunyuanvideo-community/HunyuanVideo",
        "prompt": "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.",
        "extra_args": [
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
            "--num-frames=65",
            "--width=848",
            "--height=480",
            "--num-inference-steps=30",
        ],
    },
    # 15. Skill-only extra preset
    # Requires: <repo>/inputs/diffusion_benchmark/figs/mova_single_person.jpg
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
    # 16. Skill-only extra preset
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
    # 16. Skill-only extra preset
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "joyai-edit": {
        "path": "jdopensource/JoyAI-Image-Edit-Diffusers",
        "prompt": "Make the cat wear a red hat",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--backend=sglang",
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
    # 17. Skill-only extra preset
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "firered-edit-1.0": {
        "path": "FireRedTeam/FireRed-Image-Edit-1.0",
        "prompt": "Make the cat wear a red hat",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--backend=sglang",
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
    # 18. Skill-only extra preset
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "firered-edit-1.1": {
        "path": "FireRedTeam/FireRed-Image-Edit-1.1",
        "prompt": "Make the cat wear a red hat",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--backend=sglang",
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
    # 19. Skill-only extra preset
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "hunyuan3d-shape": {
        "path": "tencent/Hunyuan3D-2",
        "prompt": "generate 3d mesh",
        "image_path": str(ASSET_DIR / "cat.png"),
        "config_overrides": {
            "paint_enable": False,
        },
        "extra_args": [
            "--backend=sglang",
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
        if flag in {"enable-torch-compile", "warmup"}:
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

        actual_args = {
            key: _normalize_cli_value(value)
            for key, value in _parse_cli_args(preset["extra_args"]).items()
        }
        expected_args = _expected_nightly_cli_args(case)
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
    print("MODEL PRESETS — Nightly-aligned first, skill-only extras after")
    print("=" * CATALOG_TABLE_WIDTH)
    print(f"{'Preset':<24} {'Nightly':<28} {'Model Path':<46} {'GPUs':>4}")
    print("-" * CATALOG_TABLE_WIDTH)
    for model_key, cfg in MODELS.items():
        print(
            f"{model_key:<24} {model_nightly_case_id(model_key):<28} {cfg['path']:<46} {required_gpus_for_model(model_key):>4}"
        )
    print("-" * CATALOG_TABLE_WIDTH)
    print(
        "Nightly column shows the comparison_configs.json case id; '-' means skill-only."
    )


def build_sglang_cmd(
    model_key: str,
    perf_dump_path: Optional[str] = None,
    warmup: bool = True,
    torch_compile: bool = True,
    seed: int = 42,
    save_output: bool = True,
) -> list[str]:
    """
    Build the `sglang generate` command for the given model.
    Matches the commands in benchmark-and-profile.md exactly.
    """
    cfg = MODELS[model_key]

    cmd = [
        "sglang",
        "generate",
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
        config_dir = ensure_dir(
            get_output_dir("benchmarks", REPO_ROOT) / "generated_configs"
        )
        config_path = config_dir / f"{model_key}.json"
        with open(config_path, "w") as f:
            json.dump(cfg["config_overrides"], f, indent=2, sort_keys=True)
        cmd.append(f"--config={config_path}")

    cmd.extend(cfg["extra_args"])

    if save_output:
        cmd.append("--save-output")
    if warmup:
        cmd.append("--warmup")
    if torch_compile:
        cmd.append("--enable-torch-compile")
    if perf_dump_path:
        cmd.extend(["--perf-dump-path", perf_dump_path])

    return cmd


def run_benchmark_once(
    model_key: str,
    label: str,
    output_dir: Path,
    warmup: bool = True,
    torch_compile: bool = True,
) -> dict:
    """Run a single benchmark pass and return results dict."""
    perf_path = output_dir / f"{model_key}_{label}.json"

    cmd = build_sglang_cmd(
        model_key,
        perf_dump_path=str(perf_path),
        warmup=warmup,
        torch_compile=torch_compile,
    )

    env = os.environ.copy()
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
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

    if not env.get("CUDA_VISIBLE_DEVICES"):
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
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        if any(signal in line.lower() for signal in DIFFUSERS_FALLBACK_SIGNALS):
            fallback_detected = True
    returncode = process.wait()
    elapsed = time.time() - t0

    if fallback_detected:
        print(
            "  ERROR: model fell back to the diffusers backend. "
            "Fix native SGLang diffusion backend selection before collecting perf data."
        )
        return {"model": model_key, "label": label, "error": True, "elapsed_s": elapsed}

    if returncode != 0:
        print(f"  ERROR: exit code {returncode}")
        return {"model": model_key, "label": label, "error": True, "elapsed_s": elapsed}

    metrics = {"model": model_key, "label": label, "elapsed_s": elapsed, "error": False}
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
                    and (
                        step_name.endswith("DenoisingStage")
                        or step_name.endswith("RefinementStage")
                    )
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

        except Exception as e:
            print(f"  Warning: could not parse perf dump: {e}")

    return metrics


def print_results_table(results: list[dict]):
    """Print a compact table for one or more benchmark runs."""
    print()
    print("=" * RESULTS_TABLE_WIDTH)
    print("BENCHMARK RESULTS — Denoise Latency (primary metric ★)")
    print("(Models and params match benchmark-and-profile.md)")
    print("=" * RESULTS_TABLE_WIDTH)

    print(
        f"{'Model':<24} {'Nightly':<28} {'Label':<12} {'Denoise(s)':>12} {'E2E(s)':>10} {'Peak Mem(GB)':>14}"
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
            f"{result['model']:<24} {model_nightly_case_id(result['model']):<28} {result['label']:<12} {denoise_text:>12} {e2e_text:>10} {mem_text:>14}"
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
        "--no-torch-compile",
        action="store_true",
        help="Keep torch.compile disabled for eager-mode comparisons.",
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
    torch_compile = not args.no_torch_compile

    models_to_run = list(MODELS.keys()) if args.all else [args.model or "flux"]
    results = []

    for model_key in models_to_run:
        results.append(
            run_benchmark_once(
                model_key,
                args.label,
                output_dir,
                warmup=warmup,
                torch_compile=torch_compile,
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
