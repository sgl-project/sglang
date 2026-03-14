"""
End-to-end denoise-stage benchmark for SGLang Diffusion with/without custom JIT CUDA kernels.

Measures denoise latency (primary metric ★) and peak GPU memory.
All model configs are kept in exact sync with diffusion-benchmark-and-profile.md.

Adapted from: https://github.com/huggingface/kernels/tree/main/skills/cuda-kernels

Usage:
    # Baseline — single model
    cd /data/bbuf/sglang
    python3 python/sglang/multimodal_gen/.claude/skills/diffusion-kernel/scripts/bench_diffusion_denoise.py --model flux

    # With custom JIT CUDA kernels
    python3 python/sglang/multimodal_gen/.claude/skills/diffusion-kernel/scripts/bench_diffusion_denoise.py --model flux --custom-kernels

    # Side-by-side comparison
    python3 python/sglang/multimodal_gen/.claude/skills/diffusion-kernel/scripts/bench_diffusion_denoise.py --model flux --compare

    # All 9 models, comparison
    python3 python/sglang/multimodal_gen/.claude/skills/diffusion-kernel/scripts/bench_diffusion_denoise.py --all --compare

Input images required for image-guided models:
    ASSET_DIR=$(python3 python/sglang/multimodal_gen/.claude/skills/diffusion-kernel/scripts/diffusion_skill_env.py print-assets-dir --mkdir)
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

# ---------------------------------------------------------------------------
# Model configs — kept in exact sync with diffusion-benchmark-and-profile.md
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
    # 6. Wan-AI/Wan2.2-T2V-A14B-Diffusers — Text-to-Video, 720P, 8 GPUs, 81 frames, 40 steps
    "wan-t2v": {
        "path": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "prompt": "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon.",
        "negative_prompt": " ",
        "extra_args": [
            "--720p",
            "--num-inference-steps=40",
            "--num-frames=81",
            "--guidance-scale=5.0",
            "--num-gpus=8",
            "--enable-cfg-parallel",
            "--ulysses-degree=4",
            "--dit-layerwise-offload",
            "true",
            "--dit-cpu-offload",
            "false",
            "--vae-cpu-offload",
            "false",
            "--text-encoder-cpu-offload",
            "true",
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
    # 9. OpenMOSS-Team/MOVA-720p — Image-to-Video, 4 GPUs, 193 frames, 24 steps
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
            "--num-inference-steps=24",
        ],
    },
}


def required_gpus_for_model(model_key: str) -> int:
    if model_key == "wan-t2v":
        return 8
    if model_key == "mova-720p":
        return 4
    return 1


def build_sglang_cmd(
    model_key: str,
    use_custom_kernels: bool,
    perf_dump_path: Optional[str] = None,
    warmup: bool = True,
    torch_compile: bool = True,
    seed: int = 42,
    save_output: bool = True,
) -> list[str]:
    """
    Build the `sglang generate` command for the given model.
    Matches the commands in diffusion-benchmark-and-profile.md exactly.
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
    use_custom_kernels: bool,
    output_dir: Path,
    warmup: bool = True,
) -> dict:
    """Run a single benchmark pass and return results dict."""
    label = "custom" if use_custom_kernels else "baseline"
    perf_path = output_dir / f"{model_key}_{label}.json"

    cmd = build_sglang_cmd(
        model_key,
        use_custom_kernels=use_custom_kernels,
        perf_dump_path=str(perf_path),
        warmup=warmup,
    )

    env = os.environ.copy()
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in pick_idle_gpus(required_gpus_for_model(model_key))
        )
    if use_custom_kernels:
        # NOTE: This env var is a convention for user-implemented kernel injection
        # logic. SGLang runtime does not read it by default — you must add handling
        # in your denoising stage or model code to check this var and apply patches.
        env["SGLANG_DIFFUSION_CUSTOM_CUDA_KERNELS"] = "1"

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

            # denoise latency: look in "steps" list for the "DenoisingStage" entry
            # steps = [{"name": "DenoisingStage", "duration_ms": 1234.5}, ...]
            denoise_latency_s = None
            for step in perf.get("steps", []):
                if (
                    step.get("name") == "DenoisingStage"
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
    """Print baseline vs custom kernel comparison table."""
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS — Denoise Latency (primary metric ★)")
    print("(Models and params match diffusion-benchmark-and-profile.md)")
    print("=" * 80)

    by_model: dict[str, dict] = {}
    for r in results:
        by_model.setdefault(r["model"], {})[r["label"]] = r

    print(
        f"{'Model':<16} {'Baseline(s)':>12} {'Custom(s)':>10} {'Speedup':>9} {'Peak Mem(GB)':>14}"
    )
    print("-" * 64)

    for model_key in MODELS:  # preserve order
        if model_key not in by_model:
            continue
        runs = by_model[model_key]
        base = runs.get("baseline", {})
        custom = runs.get("custom", {})

        base_lat = base.get("denoise_latency_s")
        custom_lat = custom.get("denoise_latency_s")
        peak_mem = base.get("peak_memory_gb") or custom.get("peak_memory_gb")

        speedup = f"{base_lat / custom_lat:.2f}x" if base_lat and custom_lat else "n/a"
        base_s = f"{base_lat:.2f}" if base_lat else "n/a"
        custom_s = f"{custom_lat:.2f}" if custom_lat else "n/a"
        mem_s = f"{peak_mem:.1f}" if isinstance(peak_mem, float) else "n/a"

        print(f"{model_key:<16} {base_s:>12} {custom_s:>10} {speedup:>9} {mem_s:>14}")

    print("-" * 64)
    print()
    print("★ Denoise latency = total DiT forward pass time across all inference steps.")
    print(
        "  See diffusion-benchmark-and-profile.md for full Level 1/2 profiling workflow."
    )


def inject_kernels_example():
    """
    Show the kernel injection pattern used when SGLANG_DIFFUSION_CUSTOM_CUDA_KERNELS=1.
    After implementing add-cuda-kernel.md, this logic lives in denoising.py or
    the model's transformer.py — NOT in this script.

    Call patch_rmsnorm(dit_model) BEFORE torch.compile and BEFORE any CPU offloading.
    """
    import torch.nn as nn

    try:
        from sglang.jit_kernel.diffusion.rmsnorm import diffusion_rmsnorm
    except ImportError:
        print(
            "diffusion.rmsnorm JIT kernel not available. "
            "Implement add-cuda-kernel.md first."
        )
        return

    def patch_rmsnorm(model: nn.Module, verbose: bool = False) -> int:
        """Monkey-patch all RMSNorm variants to use the JIT CUDA kernel."""
        patched = 0
        for name, module in model.named_modules():
            if "RMSNorm" not in type(module).__name__:
                continue
            eps = getattr(module, "eps", getattr(module, "variance_epsilon", 1e-6))
            has_weight = hasattr(module, "weight") and module.weight is not None

            if has_weight:

                def _make(mod, ep):
                    def fwd(x):
                        return diffusion_rmsnorm(x, weight=mod.weight, eps=ep)

                    return fwd

                module.forward = _make(module, eps)
            else:

                def _make_no_w(ep):
                    def fwd(x):
                        return diffusion_rmsnorm(x, weight=None, eps=ep)

                    return fwd

                module.forward = _make_no_w(eps)

            patched += 1
            if verbose:
                print(f"  Patched: {name} (weight={has_weight})")
        return patched

    return patch_rmsnorm


def main():
    parser = argparse.ArgumentParser(
        description="SGLang Diffusion denoise benchmark — baseline vs JIT CUDA kernels"
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Model to benchmark (default: flux)",
    )
    parser.add_argument("--all", action="store_true", help="Benchmark all 7 models")
    parser.add_argument(
        "--custom-kernels",
        action="store_true",
        help="Run with custom JIT CUDA kernels (SGLANG_DIFFUSION_CUSTOM_CUDA_KERNELS=1)",
    )
    parser.add_argument(
        "--no-custom-kernels",
        action="store_true",
        help="Run baseline (no custom kernels)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and custom, print comparison table",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(get_output_dir("benchmarks", REPO_ROOT)),
        help="Directory for perf dump JSON files",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup")
    parser.add_argument(
        "--show-injection-example",
        action="store_true",
        help="Print kernel injection pattern and exit",
    )

    args = parser.parse_args()

    if args.show_injection_example:
        patch_fn = inject_kernels_example()
        if patch_fn:
            print(
                "patch_rmsnorm function defined. "
                "Call it on the DiT model before torch.compile and CPU offloading."
            )
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warmup = not args.no_warmup

    models_to_run = list(MODELS.keys()) if args.all else [args.model or "flux"]
    results = []

    for model_key in models_to_run:
        if args.compare:
            results.append(run_benchmark_once(model_key, False, output_dir, warmup))
            results.append(run_benchmark_once(model_key, True, output_dir, warmup))
        elif args.custom_kernels:
            results.append(run_benchmark_once(model_key, True, output_dir, warmup))
        else:
            results.append(run_benchmark_once(model_key, False, output_dir, warmup))

    if results:
        print_results_table(results)

    print(f"Perf dump JSONs → {output_dir}")
    print(
        "Compare across runs: follow diffusion-benchmark-and-profile.md → Perf dump & before/after compare."
    )


if __name__ == "__main__":
    main()
