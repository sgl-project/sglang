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

    # All 10 preset models
    python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py --all

For gated Hugging Face repos such as FLUX, export HF_TOKEN first:
    export HF_TOKEN=<your_hf_token>

Input images required for image-guided models:
    ASSET_DIR=$(python3 python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/diffusion_skill_env.py print-assets-dir --mkdir)
    wget -O "${ASSET_DIR}/cat.png" \
      https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
    wget -O "${ASSET_DIR}/astronaut.jpg" \
      https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg
    wget -O "${ASSET_DIR}/mova_single_person.jpg" \
      https://github.com/OpenMOSS/MOVA/raw/main/assets/single_person.jpg
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

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
GATED_MODELS = {"flux", "flux2"}

# ---------------------------------------------------------------------------
# Model configs — kept in exact sync with benchmark-and-profile.md
# Each entry produces the same `sglang generate` command as shown in that doc.
# ---------------------------------------------------------------------------
MODELS = {
    # 1. Qwen/Qwen-Image-2512 — Text-to-Image, 1024×1024, 50 steps
    "qwen": {
        "path": "Qwen/Qwen-Image-2512",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k",
        "negative_prompt": " ",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=50",
            "--guidance-scale=4.0",
            "--dit-cpu-offload",
            "false",
            "--text-encoder-cpu-offload",
            "false",
        ],
    },
    # 2. Qwen/Qwen-Image-Edit-2511 — Image Editing, 1024×1024, 50 steps
    # Requires: <repo>/inputs/diffusion_benchmark/figs/cat.png
    "qwen-edit": {
        "path": "Qwen/Qwen-Image-Edit-2511",
        "prompt": "Transform into anime style",
        "negative_prompt": " ",
        "image_path": str(ASSET_DIR / "cat.png"),
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=50",
            "--guidance-scale=4.0",
            "--dit-cpu-offload",
            "false",
            "--text-encoder-cpu-offload",
            "false",
        ],
    },
    # 3. black-forest-labs/FLUX.1-dev — Text-to-Image, 1024×1024, 50 steps
    "flux": {
        "path": "black-forest-labs/FLUX.1-dev",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets, highly detailed, 8k",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=50",
            "--guidance-scale=4.0",
        ],
    },
    # 4. black-forest-labs/FLUX.2-dev — Text-to-Image, 1024×1024
    "flux2": {
        "path": "black-forest-labs/FLUX.2-dev",
        "prompt": "A Logo With Bold Large Text: SGL Diffusion",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--dit-layerwise-offload",
            "false",
            "--dit-cpu-offload",
            "false",
            "--text-encoder-cpu-offload",
            "true",
            "--vae-cpu-offload",
            "false",
        ],
    },
    # 5. Tongyi-MAI/Z-Image-Turbo — Turbo Text-to-Image, 1024×1024, 9 steps
    "zimage": {
        "path": "Tongyi-MAI/Z-Image-Turbo",
        "prompt": "A fantasy landscape with mountains and a river, detailed, vibrant colors",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=9",
            "--guidance-scale=0.0",
            "--dit-cpu-offload",
            "false",
            "--text-encoder-cpu-offload",
            "false",
        ],
    },
    # 6. Wan-AI/Wan2.2-T2V-A14B-Diffusers — Text-to-Video, 720P, 4 GPUs, 81 frames, 2 steps
    "wan-t2v": {
        "path": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "prompt": "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon.",
        "negative_prompt": " ",
        "extra_args": [
            "--720p",
            "--num-inference-steps=2",
            "--num-frames=81",
            "--guidance-scale=5.0",
            "--num-gpus=4",
            "--ulysses-degree=4",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
        ],
    },
    # 7. Wan-AI/Wan2.2-TI2V-5B-Diffusers — Text-Image-to-Video, 720P, 1 GPU, 81 frames, 50 steps
    # Requires: <repo>/inputs/diffusion_benchmark/figs/astronaut.jpg
    "wan-ti2v": {
        "path": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "prompt": "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
        "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        "image_path": str(ASSET_DIR / "astronaut.jpg"),
        "extra_args": [
            "--num-frames",
            "81",
            "--720p",
            "--num-inference-steps",
            "50",
            "--guidance-scale",
            "5.0",
            "--dit-layerwise-offload",
            "false",
            "--dit-cpu-offload",
            "false",
            "--vae-cpu-offload",
            "false",
            "--text-encoder-cpu-offload",
            "false",
        ],
    },
    # 8. hunyuanvideo-community/HunyuanVideo — Text-to-Video, 848×480, 65 frames, 30 steps
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
    # 9. OpenMOSS-Team/MOVA-720p — Image-to-Video, 4 GPUs, 193 frames, 2 steps
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
    # 10. BestWishYsh/Helios-Base — Text-to-Video, 640×384, 33 frames
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
}


def required_gpus_for_model(model_key: str) -> int:
    if model_key == "wan-t2v":
        return 4
    if model_key == "mova-720p":
        return 4
    return 1


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
        "--log-level=info",
    ]

    if seed is not None:
        cmd.append(f"--seed={seed}")

    if "negative_prompt" in cfg:
        cmd.append(f"--negative-prompt={cfg['negative_prompt']}")

    if "image_path" in cfg:
        cmd.append(f"--image-path={cfg['image_path']}")

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
) -> dict:
    """Run a single benchmark pass and return results dict."""
    perf_path = output_dir / f"{model_key}_{label}.json"

    cmd = build_sglang_cmd(
        model_key,
        perf_dump_path=str(perf_path),
        warmup=warmup,
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
    result = subprocess.run(cmd, env=env, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  ERROR: exit code {result.returncode}")
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

            # denoise latency: accept the canonical "DenoisingStage" plus
            # model-specific variants such as "MOVADenoisingStage" and
            # "HeliosChunkedDenoisingStage".
            # steps = [{"name": "DenoisingStage", "duration_ms": 1234.5}, ...]
            denoise_latency_s = None
            for step in perf.get("steps", []):
                step_name = step.get("name")
                if (
                    isinstance(step_name, str)
                    and "DenoisingStage" in step_name
                    and step.get("duration_ms") is not None
                ):
                    denoise_latency_s = float(step["duration_ms"]) / 1000.0
                    break

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
    print("=" * 80)
    print("BENCHMARK RESULTS — Denoise Latency (primary metric ★)")
    print("(Models and params match benchmark-and-profile.md)")
    print("=" * 80)

    print(
        f"{'Model':<16} {'Label':<12} {'Denoise(s)':>12} {'E2E(s)':>10} {'Peak Mem(GB)':>14}"
    )
    print("-" * 64)

    for result in results:
        denoise_s = result.get("denoise_latency_s")
        e2e_s = result.get("e2e_latency_s")
        peak_mem = result.get("peak_memory_gb")
        denoise_text = f"{denoise_s:.2f}" if isinstance(denoise_s, float) else "n/a"
        e2e_text = f"{e2e_s:.2f}" if isinstance(e2e_s, float) else "n/a"
        mem_text = f"{peak_mem:.1f}" if isinstance(peak_mem, float) else "n/a"
        print(
            f"{result['model']:<16} {result['label']:<12} {denoise_text:>12} {e2e_text:>10} {mem_text:>14}"
        )

    print("-" * 64)
    print()
    print("★ Denoise latency = total DiT forward pass time across all inference steps.")
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
    parser.add_argument("--all", action="store_true", help="Benchmark all 10 models")
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

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup = not args.no_warmup

    models_to_run = list(MODELS.keys()) if args.all else [args.model or "flux"]
    results = []

    for model_key in models_to_run:
        results.append(run_benchmark_once(model_key, args.label, output_dir, warmup))

    if results:
        print_results_table(results)

    print(f"Perf dump JSONs → {output_dir}")
    print(
        "Compare across runs: follow benchmark-and-profile.md -> Perf dump & before/after compare."
    )


if __name__ == "__main__":
    main()
