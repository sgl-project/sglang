# SPDX-License-Identifier: Apache-2.0
"""Offline promotion tool for the compile-trajectory gate (see RFC #35333,
:mod:`sglang.multimodal_gen.runtime.utils.compile_trajectory_gate`).

Runs one eager reference rollout and one regionally-compiled candidate
rollout of the same model/prompt/seed, captures per-step denoising latents
via ``return_trajectory_latents`` (the same plumbing
``tools/compare_diffusion_trajectory_similarity.py`` uses), scores them with
``run_trajectory_gate()``, and writes the resulting
``CompiledPlanManifest`` to a JSON file consumable by
``--compile-trajectory-gate-manifest`` at serve time.

Example:

    python -m sglang.multimodal_gen.tools.build_compile_trajectory_manifest \\
        --model-path /path/to/model \\
        --prompt "A futuristic cyberpunk city at night" \\
        --width 512 --height 512 --num-inference-steps 8 --seed 42 \\
        --cosine-min 0.999 --max-abs-max 0.05 \\
        --output-manifest /tmp/compile_trajectory_manifest.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import torch

from sglang.multimodal_gen.runtime.utils.compile_trajectory_gate import (
    CompiledPlanManifest,
    CompileWorkloadSignature,
    compute_tensor_metrics,
    passes_thresholds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--dit-precision",
        default="bf16",
        help=(
            "Must match the deployed model's actual PipelineConfig.dit_precision "
            "(default bf16 for most models) -- this feeds the workload signature "
            "DenoisingStage._build_compile_workload_signature computes from the "
            "live pipeline_config, so a mismatch here means the manifest never "
            "matches at serve time."
        ),
    )
    parser.add_argument(
        "--regional-compile",
        action="store_true",
        help=(
            "Compile only the matching submodules declared by the model's "
            "_compile_conditions instead of the whole transformer. Only some "
            "models declare _compile_conditions (e.g. ZImageTransformer2DModel "
            "does not); passing this for an unsupported model raises "
            "'no matching submodules' rather than falling back silently."
        ),
    )
    parser.add_argument(
        "--enable-cfg-parallel",
        action="store_true",
        help=(
            "Validate the plan under CFG-parallel instead of CFG-serial. Both "
            "reference and candidate rollouts use the same setting -- the gate "
            "compares eager vs. compiled, not serial vs. parallel."
        ),
    )
    parser.add_argument(
        "--cache-mode",
        default="none",
        choices=["none", "teacache", "step_reuse", "spectrum"],
        help=(
            "Label for the cache-dit mode under validation; must match the "
            "cache_mode DenoisingStage._build_compile_workload_signature derives "
            "from the actual request at serve time, or the manifest won't match "
            "any real request's signature."
        ),
    )
    parser.add_argument(
        "--cache-dit-config",
        default=None,
        help="Passed through to DiffGenerator as cache_dit_config, if --cache-mode != none.",
    )
    parser.add_argument(
        "--cosine-min",
        type=float,
        default=0.999,
        help="Minimum per-checkpoint cosine similarity required to promote the plan.",
    )
    parser.add_argument(
        "--max-abs-max",
        type=float,
        default=0.05,
        help="Maximum per-checkpoint max-abs error tolerated to promote the plan.",
    )
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing manifest file's JSON array instead of overwriting it.",
    )
    return parser.parse_args()


class RolloutResult(NamedTuple):
    """A rollout's steady-state (warm) generation result plus its benchmark
    numbers -- the RFC's "cold compile/warmup wall time" and "warm p50/p95
    latency" report, for one signature/plan.
    """

    result: Any
    cold_time_s: float
    warm_time_s: float | None
    peak_memory_mb: float | None


def _run_rollout(
    *,
    model_path: str,
    num_gpus: int,
    sampling_kwargs: dict[str, Any],
    compile_candidate: bool,
    warmup: bool,
    enable_cfg_parallel: bool = False,
    cache_dit_config: str | None = None,
    regional_compile: bool = True,
) -> RolloutResult:
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    server_kwargs: dict[str, Any] = {
        "local_mode": True,
        "model_path": model_path,
        "num_gpus": num_gpus,
        "enable_torch_compile": compile_candidate,
        # Whole-module compile if the model declares no _compile_conditions
        # (e.g. ZImageTransformer2DModel); regional compile raises "no
        # matching submodules" for those models instead of silently falling
        # back, so this must match what the model actually supports.
        "regional_compile": compile_candidate and regional_compile,
        "enable_cfg_parallel": enable_cfg_parallel,
    }
    if cache_dit_config is not None:
        server_kwargs["cache_dit_config"] = cache_dit_config

    with DiffGenerator.from_pretrained(**server_kwargs) as generator:
        cold_start = time.perf_counter()
        result = generator.generate(sampling_params_kwargs=sampling_kwargs)
        cold_time_s = time.perf_counter() - cold_start
        if isinstance(result, list):
            result = result[0]

        warm_time_s: float | None = None
        if warmup:
            warm_start = time.perf_counter()
            warm_result = generator.generate(sampling_params_kwargs=sampling_kwargs)
            warm_time_s = time.perf_counter() - warm_start
            if isinstance(warm_result, list):
                warm_result = warm_result[0]
            result = warm_result

    return RolloutResult(
        result=result,
        cold_time_s=cold_time_s,
        warm_time_s=warm_time_s,
        peak_memory_mb=getattr(result, "peak_memory_mb", None),
    )


def _checkpoint_names(num_steps: int) -> tuple[str, ...]:
    if num_steps == 1:
        return ("step_0",)
    return ("step_0", "terminal")


def build_manifest(args: argparse.Namespace) -> CompiledPlanManifest:
    sampling_kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "return_trajectory_latents": True,
        "record_decision_trace": True,
        "save_output": False,
    }
    if args.num_frames is not None:
        sampling_kwargs["num_frames"] = args.num_frames

    reference = _run_rollout(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        sampling_kwargs=sampling_kwargs,
        compile_candidate=False,
        warmup=False,
        enable_cfg_parallel=args.enable_cfg_parallel,
        cache_dit_config=args.cache_dit_config,
    )
    candidate = _run_rollout(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        sampling_kwargs=sampling_kwargs,
        compile_candidate=True,
        warmup=True,
        enable_cfg_parallel=args.enable_cfg_parallel,
        cache_dit_config=args.cache_dit_config,
        regional_compile=args.regional_compile,
    )

    ref_latents = torch.as_tensor(reference.result.trajectory_latents)
    cand_latents = torch.as_tensor(candidate.result.trajectory_latents)
    if ref_latents.shape != cand_latents.shape:
        raise ValueError(
            "Reference/candidate trajectory shape mismatch: "
            f"{tuple(ref_latents.shape)} vs {tuple(cand_latents.shape)}"
        )
    num_steps = ref_latents.shape[1]
    checkpoints = _checkpoint_names(num_steps)
    step_by_checkpoint = {"step_0": 0, "terminal": num_steps - 1}

    checkpoint_metrics: dict[str, dict[str, float]] = {}
    for checkpoint_name in checkpoints:
        step_index = step_by_checkpoint[checkpoint_name]
        checkpoint_metrics[checkpoint_name] = compute_tensor_metrics(
            ref_latents[:, step_index], cand_latents[:, step_index]
        )

    thresholds = {"cosine_similarity": args.cosine_min, "max_abs": args.max_abs_max}
    status = "validated"
    for checkpoint_name in checkpoints:
        if not passes_thresholds(checkpoint_metrics[checkpoint_name], thresholds):
            status = "rejected"
            break

    # Only meaningful when cache-dit/teacache is active (args.cache_mode != "none");
    # both traces come back empty otherwise, and an empty-vs-empty match would be a
    # vacuous pass, not a real decision-trace validation.
    reference_trace = list(getattr(reference.result, "decision_trace", None) or [])
    candidate_trace = list(getattr(candidate.result, "decision_trace", None) or [])
    decision_trace_matched: bool | None = None
    if reference_trace or candidate_trace:
        decision_trace_matched = reference_trace == candidate_trace
        if not decision_trace_matched:
            status = "rejected"

    signature = CompileWorkloadSignature(
        model_revision=args.model_path,
        dtype=args.dit_precision,
        backend="torch.compile",
        parallel_signature=f"sp{args.num_gpus}",
        # num_frames defaults to 1 (not None) at the Req/SamplingParams level
        # even for image models, so it must be included here the same way, or
        # this will never match DenoisingStage._build_compile_workload_signature's
        # (height, width, num_frames) tuple at serve time.
        latent_shape_regime=(args.height, args.width, args.num_frames or 1),
        num_inference_steps=args.num_inference_steps,
        cfg_mode=(
            "no_cfg"
            if args.guidance_scale <= 0.0
            else ("cfg_parallel" if args.enable_cfg_parallel else "cfg")
        ),
        cache_mode=args.cache_mode,
        state_schema_version="v1",
    )

    benchmark = {
        "cold_compile_time_s": candidate.cold_time_s,
        "eager_reference_time_s": reference.cold_time_s,
    }
    if candidate.warm_time_s is not None:
        benchmark["warm_steady_state_time_s"] = candidate.warm_time_s
    if candidate.peak_memory_mb is not None:
        benchmark["peak_memory_mb"] = candidate.peak_memory_mb

    return CompiledPlanManifest(
        signature=signature,
        regions=(),
        compile_options={"regional_compile": args.regional_compile},
        gate_digest=signature.digest(),
        status=status,
        checkpoint_metrics=checkpoint_metrics,
        benchmark=benchmark,
        decision_trace_matched=decision_trace_matched,
    )


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)

    output_path = Path(args.output_manifest)
    existing: list[dict[str, Any]] = []
    if args.append and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(existing + [manifest.to_dict()], indent=2), encoding="utf-8"
    )
    print(
        f"compile-trajectory-gate: wrote {manifest.status!r} manifest entry "
        f"(gate_digest={manifest.gate_digest}) to {output_path}"
    )


if __name__ == "__main__":
    main()
