import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ["SGLANG_NVFP4_REPO_ROOT"]) if os.environ.get(
    "SGLANG_NVFP4_REPO_ROOT"
) else Path(__file__).resolve().parents[5]
SKILL_SCRIPTS_DIR = (
    REPO_ROOT
    / "python"
    / "sglang"
    / "multimodal_gen"
    / ".claude"
    / "skills"
    / "sglang-diffusion-benchmark-profile"
    / "scripts"
)

if str(SKILL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS_DIR))

from diffusion_skill_env import ensure_dir, get_output_dir, pick_idle_gpus

PRESETS: dict[str, dict[str, Any]] = {
    "flux2-nvfp4": {
        "model_path": "black-forest-labs/FLUX.2-dev",
        "transformer_weights_path": "black-forest-labs/FLUX.2-dev-NVFP4",
        "prompt": "A futuristic cyberpunk city at night, neon lights reflecting on wet streets",
        "extra_args": [
            "--width=1024",
            "--height=1024",
            "--num-inference-steps=1",
            "--guidance-scale=4.0",
            "--dit-layerwise-offload",
            "false",
        ],
        "required_gpus": 1,
        "required_fused_prefix_markers": ["to_qkv", "to_qkv_mlp_proj"],
    },
}


def build_generate_cmd(
    preset: str,
    *,
    seed: int = 42,
    save_output: bool = False,
    warmup: bool = False,
    torch_compile: bool = False,
) -> list[str]:
    cfg = PRESETS[preset]
    cmd = [
        "sglang",
        "generate",
        f"--model-path={cfg['model_path']}",
        f"--transformer-weights-path={cfg['transformer_weights_path']}",
        f"--prompt={cfg['prompt']}",
        "--log-level=info",
        f"--seed={seed}",
    ]
    cmd.extend(cfg["extra_args"])
    if save_output:
        cmd.append("--save-output")
    if warmup:
        cmd.append("--warmup")
    if torch_compile:
        cmd.append("--enable-torch-compile")
    return cmd


def build_sitecustomize() -> str:
    return textwrap.dedent(
        """
        import json
        import os
        import threading

        LOG_PATH = os.environ.get("SGLANG_NVFP4_SHAPE_LOG")
        FORCE_FLASHINFER = os.environ.get("SGLANG_NVFP4_FORCE_FLASHINFER") == "1"
        DUMMY_OUTPUT = os.environ.get("SGLANG_NVFP4_SHAPE_DUMMY_OUTPUT") == "1"

        if LOG_PATH:
            _LOCK = threading.Lock()

            def _append(record):
                with _LOCK:
                    with open(LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\\n")

            try:
                import flashinfer
                import torch
                from sglang.multimodal_gen.runtime.platforms import current_platform
                from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
                    ComfyUIFp4LinearMethod,
                    ModelOptFp4LinearMethod,
                )

                if FORCE_FLASHINFER:
                    current_platform.get_modelopt_fp4_quantize_op = (
                        lambda *args, **kwargs: flashinfer.fp4_quantize
                    )
                    current_platform.get_modelopt_fp4_gemm_op = (
                        lambda *args, **kwargs: (flashinfer.mm_fp4, "auto")
                    )

                _orig_generic_apply = ModelOptFp4LinearMethod.apply
                _orig_comfy_apply = ComfyUIFp4LinearMethod.apply

                def _record(impl_name, layer, x):
                    weight = getattr(layer, "weight", None)
                    if weight is None or getattr(weight, "dtype", None) != torch.uint8:
                        return
                    if not (
                        hasattr(layer, "weight_scale")
                        or hasattr(layer, "weight_scale_interleaved")
                        or hasattr(layer, "weight_scale_ck")
                    ):
                        return
                    x_2d = x.reshape(-1, x.shape[-1])
                    _append(
                        {
                            "impl": impl_name,
                            "prefix": getattr(layer, "prefix", ""),
                            "input_shape": list(x.shape),
                            "m": int(x_2d.shape[0]),
                            "n": int(getattr(layer, "output_size_per_partition", 0)),
                            "k": int(x_2d.shape[1]),
                            "weight_dtype": str(weight.dtype),
                        }
                    )

                def _dummy(layer, x):
                    output_shape = list(x.shape[:-1]) + [
                        int(getattr(layer, "output_size_per_partition", 0))
                    ]
                    return x.new_zeros(output_shape)

                def _generic_apply(self, layer, x, bias=None):
                    _record("generic", layer, x)
                    if DUMMY_OUTPUT:
                        return _dummy(layer, x)
                    return _orig_generic_apply(self, layer, x, bias)

                def _comfy_apply(self, layer, x, bias=None):
                    _record("comfy", layer, x)
                    if DUMMY_OUTPUT:
                        return _dummy(layer, x)
                    return _orig_comfy_apply(self, layer, x, bias)

                ModelOptFp4LinearMethod.apply = _generic_apply
                ComfyUIFp4LinearMethod.apply = _comfy_apply
                print(f"[probe_diffusion_nvfp4_shapes] logging to {LOG_PATH}", flush=True)
            except Exception as exc:
                print(
                    f"[probe_diffusion_nvfp4_shapes] patch failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
        """
    )


def summarize_rows(rows: list[dict[str, Any]], preset: str) -> dict[str, Any]:
    shape_counter = Counter((row["m"], row["n"], row["k"]) for row in rows)
    flop_counter = Counter(
        {
            shape: 2 * shape[0] * shape[1] * shape[2] * count
            for shape, count in shape_counter.items()
        }
    )
    prefix_counter = Counter(row["prefix"] for row in rows)
    total_calls = sum(shape_counter.values())
    total_flops = sum(flop_counter.values())

    top_shapes = []
    for shape, approx_flops in flop_counter.most_common(10):
        m, n, k = shape
        matching_prefixes = [
            row["prefix"] for row in rows if (row["m"], row["n"], row["k"]) == shape
        ]
        prefix_counts = Counter(matching_prefixes)
        top_shapes.append(
            {
                "m": m,
                "n": n,
                "k": k,
                "count": shape_counter[shape],
                "share_calls": shape_counter[shape] / total_calls if total_calls else 0.0,
                "approx_flops": approx_flops,
                "share_flops": approx_flops / total_flops if total_flops else 0.0,
                "example_prefix": prefix_counts.most_common(1)[0][0],
            }
        )

    fused_markers = PRESETS[preset].get("required_fused_prefix_markers", [])
    fused_prefixes = sorted(
        {
            prefix
            for prefix in prefix_counter
            if any(marker in prefix for marker in fused_markers)
        }
    )

    return {
        "preset": preset,
        "total_calls": total_calls,
        "unique_shapes": len(shape_counter),
        "top_shapes": top_shapes,
        "top_prefixes": [
            {"prefix": prefix, "count": count}
            for prefix, count in prefix_counter.most_common(20)
        ],
        "fused_prefixes": fused_prefixes,
    }


def write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["impl", "prefix", "input_shape", "m", "n", "k", "weight_dtype"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    summary: dict[str, Any], cmd: list[str], output_path: Path
) -> None:
    lines: list[str] = []
    lines.append("# Diffusion NVFP4 Shape Probe")
    lines.append("")
    lines.append(f"- Preset: `{summary['preset']}`")
    lines.append(f"- Total NVFP4 GEMM calls: `{summary['total_calls']}`")
    lines.append(f"- Unique shapes: `{summary['unique_shapes']}`")
    lines.append(f"- Command: `{' '.join(cmd)}`")
    lines.append("")
    lines.append("## Fused Prefixes Seen")
    lines.append("")
    if summary["fused_prefixes"]:
        for prefix in summary["fused_prefixes"]:
            lines.append(f"- `{prefix}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Top Shapes By Approx FLOPs")
    lines.append("")
    lines.append("| Shape `(M,N,K)` | Calls | Share Calls | Share FLOPs | Example Prefix |")
    lines.append("|---|---:|---:|---:|---|")
    for row in summary["top_shapes"]:
        lines.append(
            f"| `({row['m']}, {row['n']}, {row['k']})` | {row['count']} | {row['share_calls']:.1%} | {row['share_flops']:.1%} | `{row['example_prefix']}` |"
        )
    lines.append("")
    lines.append("## Top Prefixes By Call Count")
    lines.append("")
    lines.append("| Prefix | Calls |")
    lines.append("|---|---:|")
    for row in summary["top_prefixes"]:
        lines.append(f"| `{row['prefix']}` | {row['count']} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real diffusion generate command and record actual NVFP4 GEMM shapes."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="flux2-nvfp4",
        help="Preset command to run for shape collection.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(get_output_dir("benchmarks", REPO_ROOT) / "nvfp4_shapes"),
        help="Directory for logs, CSV, JSON, and Markdown outputs.",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Keep the generated image/video output from the probe run.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Add --warmup to the probe command.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Add --enable-torch-compile to the probe command.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed passed to sglang generate.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Override CUDA_VISIBLE_DEVICES. Defaults to an idle GPU.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output_dir))
    cmd = build_generate_cmd(
        args.preset,
        seed=args.seed,
        save_output=args.save_output,
        warmup=args.warmup,
        torch_compile=args.torch_compile,
    )

    env = os.environ.copy()
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    if env.get("HF_TOKEN") and not env.get("HUGGINGFACE_HUB_TOKEN"):
        env["HUGGINGFACE_HUB_TOKEN"] = env["HF_TOKEN"]

    if not (env.get("HF_TOKEN") or env.get("HUGGINGFACE_HUB_TOKEN")):
        raise RuntimeError("HF_TOKEN is required for the gated FLUX.2 NVFP4 preset.")

    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    elif not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in pick_idle_gpus(PRESETS[args.preset]["required_gpus"])
        )

    run_name = f"{args.preset}_seed{args.seed}"
    jsonl_path = output_dir / f"{run_name}.jsonl"
    csv_path = output_dir / f"{run_name}.csv"
    summary_json_path = output_dir / f"{run_name}_summary.json"
    summary_md_path = output_dir / f"{run_name}_summary.md"
    log_path = output_dir / f"{run_name}.log"

    if jsonl_path.exists():
        jsonl_path.unlink()

    with tempfile.TemporaryDirectory(prefix="sglang_nvfp4_probe_") as tmpdir:
        patch_dir = Path(tmpdir)
        (patch_dir / "sitecustomize.py").write_text(
            build_sitecustomize(), encoding="utf-8"
        )
        python_path_parts = [str(patch_dir), str(REPO_ROOT / "python")]
        if env.get("PYTHONPATH"):
            python_path_parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = ":".join(python_path_parts)
        env["SGLANG_NVFP4_SHAPE_LOG"] = str(jsonl_path)

        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

    if result.returncode != 0:
        raise RuntimeError(
            f"Probe command failed with exit code {result.returncode}. See {log_path}."
        )

    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"No NVFP4 shape rows were captured. See {log_path}.")

    summary = summarize_rows(rows, args.preset)
    if not summary["fused_prefixes"]:
        raise RuntimeError(
            "Did not observe any fused NVFP4 prefixes. "
            "Packed-QKV shape collection likely did not trigger as expected."
        )

    write_rows_csv(rows, csv_path)
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(summary, cmd, summary_md_path)

    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_md_path}")
    print(f"Wrote {log_path}")
    print(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}")
    print(f"Top fused prefixes: {summary['fused_prefixes'][:5]}")


if __name__ == "__main__":
    main()
