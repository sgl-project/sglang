# SPDX-License-Identifier: Apache-2.0
"""Full-checkpoint Ovis oracle and native runner; use separate Python environments.

Run the native mode under torchrun for TP/SP/CFG. Results are CPU tensors, a
PNG, and JSON provenance. Supply an existing local model and allocated GPUs.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_REVISION = "41be1c5821a92c970d63d7eb595a2fd3fe32b22e"
REFERENCE_REVISION = "c6df88a511a98740646ee55577b590c9852650ce"
COMPARISON_ARGUMENTS = (
    "prompt",
    "second_prompt",
    "height",
    "width",
    "steps",
    "seed",
    "guidance",
    "text_length",
    "outputs",
    "vae_tiling",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("reference", "native", "compare"), required=True
    )
    parser.add_argument("--model-path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--comparison-name", default="comparison.json")
    parser.add_argument(
        "--prompt", default='A shop sign reading "HELLO 世界", beside a red bicycle.'
    )
    parser.add_argument(
        "--second-prompt",
        help="Exercise an internal two-prompt batch; this does not test the HTTP API",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--text-length", type=int, default=256)
    parser.add_argument("--outputs", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ulysses", type=int, default=1)
    parser.add_argument("--ring", type=int, default=1)
    parser.add_argument("--cfg", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--offload", choices=("none", "component", "layerwise"), default="none"
    )
    parser.add_argument("--vae-tiling", action="store_true")
    parser.add_argument("--vae-sp", action="store_true")
    parser.add_argument("--attention", default="torch_sdpa")
    args = parser.parse_args()
    if Path(
        args.comparison_name
    ).name != args.comparison_name or not args.comparison_name.endswith(".json"):
        parser.error("--comparison-name must be a JSON filename")
    if args.mode == "compare" and args.reference is None:
        parser.error("--reference is required in compare mode")
    if args.mode != "compare" and (
        args.model_path is None or not Path(args.model_path).is_dir()
    ):
        parser.error("--model-path must name an existing local model directory")
    if args.vae_sp and not args.vae_tiling:
        parser.error("--vae-sp requires --vae-tiling")
    return args


def prompts(args):
    if args.second_prompt is not None:
        return [args.prompt, args.second_prompt]
    return args.prompt


def sample_count(args):
    return (2 if args.second_prompt is not None else 1) * args.outputs


def initial_latents(args):
    return torch.randn(
        sample_count(args),
        16,
        args.height // 8,
        args.width // 8,
        generator=torch.Generator("cpu").manual_seed(args.seed),
        dtype=torch.bfloat16,
    )


def jsonable(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, set):
        return sorted(jsonable(x) for x in value)
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def source_provenance(path):
    """Hash tracked changes and untracked source files, including this runner."""
    path = Path(path).resolve()
    if path.is_file():
        path = path.parent

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], stderr=subprocess.DEVNULL
        )

    try:
        root = Path(git("rev-parse", "--show-toplevel").decode().strip())
        path = root
        commit = git("rev-parse", "HEAD").decode().strip()
        digest = hashlib.sha256(git("diff", "--binary", "HEAD"))
        untracked = sorted(
            name.decode()
            for name in git("ls-files", "--others", "--exclude-standard", "-z").split(
                b"\0"
            )
            if name
        )
        for name in untracked:
            digest.update(name.encode() + b"\0")
            digest.update((root / name).read_bytes())
        return {
            "repository": str(root),
            "commit": commit,
            "diff_sha256": digest.hexdigest(),
            "untracked_files": untracked,
        }
    except subprocess.CalledProcessError:
        return {"repository": None, "commit": None, "diff_sha256": None}


def validate_rank_sources(records):
    """Reject a distributed capture assembled from different source revisions."""
    sources = []
    baseline = None
    for index, record in enumerate(records):
        source = record.get("source")
        rank = record.get("rank_metrics", {}).get("rank", index)
        if not isinstance(source, dict) or any(
            not isinstance(source.get(key), str) or not source[key]
            for key in ("commit", "diff_sha256")
        ):
            raise ValueError(f"Rank {rank}: missing source commit or diff_sha256")
        identity = (source["commit"], source["diff_sha256"])
        if baseline is None:
            baseline = identity
        elif identity != baseline:
            raise ValueError(
                f"Rank {rank}: source commit/diff_sha256 differs from rank 0"
            )
        sources.append({"rank": rank, "source": source})
    if not sources:
        raise ValueError("No rank source records were gathered")
    return sources


def sdpa_flags():
    return {
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
    }


def resolved_encoder_tp_group(encoder):
    """Describe the loaded encoder's bound group and actual linear shards.

    ServerArgs.encoder_tp belongs to disaggregated serving; it does not select
    the group for this monolithic runner. Inspect loaded modules instead.
    """
    group = encoder._encoder_tp_group
    if group is None:
        raise ValueError("Loaded native Ovis encoder has no bound TP group")
    layer = encoder.layers[0]
    attention = layer.self_attn

    def linear(module):
        return {
            "class": f"{type(module).__module__}.{type(module).__qualname__}",
            "tp_size": module.tp_size,
            "tp_rank": module.tp_rank,
            "input_size": module.input_size,
            "output_size": module.output_size,
            "weight_shape": list(module.weight.shape),
            "weight_dtype": str(module.weight.dtype),
        }

    return {
        "ranks": list(group.ranks),
        "world_size": group.world_size,
        "rank_in_group": group.rank_in_group,
        "model_class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
        "layers": len(encoder.layers),
        "local_attention_heads": attention.num_heads,
        "local_kv_heads": attention.num_kv_heads,
        "head_dim": attention.head_dim,
        "qkv_proj": linear(attention.qkv_proj),
        "o_proj": linear(attention.o_proj),
        "gate_up_proj": linear(layer.mlp.gate_up_proj),
        "down_proj": linear(layer.mlp.down_proj),
    }


def capture_transformer(module, record, call_context, convention):
    """Capture the first step's branches and every actual time projection input."""
    pending = None
    rank = int(os.environ.get("RANK", 0))

    def before(module, positional, kwargs):
        nonlocal pending
        step, is_negative = call_context()
        pending = {
            "step": int(step),
            "is_cfg_negative": bool(is_negative),
            "rank": rank,
            "timestep_convention": convention,
            "timestep": kwargs["timestep"].detach().cpu().clone(),
        }
        if step == 0:
            pending["encoder_hidden_states"] = (
                kwargs["encoder_hidden_states"].detach().cpu().clone()
            )

    def project_time(module, positional):
        if pending is None:
            raise RuntimeError("Time projection ran outside a transformer call")
        pending["effective_timestep"] = positional[0].detach().cpu().clone()

    def after(module, positional, kwargs, output):
        nonlocal pending
        if pending is None or "effective_timestep" not in pending:
            raise RuntimeError("Transformer call did not capture its time projection")
        if pending["step"] == 0:
            if isinstance(output, tuple):
                output = output[0]
            elif hasattr(output, "sample"):
                output = output.sample
            pending["prediction"] = output.detach().cpu().clone()
        record["calls"].append(pending)
        pending = None

    return (
        module.register_forward_pre_hook(before, with_kwargs=True),
        module.time_proj.register_forward_pre_hook(project_time),
        module.register_forward_hook(after, with_kwargs=True),
    )


def finalize_captures(records, steps, uses_cfg):
    """Deduplicate replicated TP/SP calls and order positive before negative."""
    unique = {}
    for record in records:
        for call in record["calls"]:
            key = (call["step"], call["is_cfg_negative"])
            if key not in unique:
                unique[key] = {**call, "ranks": [call["rank"]]}
                continue
            previous = unique[key]
            for name in (
                "timestep",
                "effective_timestep",
                "encoder_hidden_states",
                "prediction",
            ):
                if name not in call and name not in previous:
                    continue
                if name not in call or name not in previous:
                    raise ValueError(f"Inconsistent capture {key}: missing {name}")
                a, b = call[name], previous[name]
                if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                    raise ValueError(
                        f"Replicated ranks disagree for {key}/{name}: "
                        f"{previous['ranks']} versus {call['rank']}"
                    )
            if previous["timestep_convention"] != call["timestep_convention"]:
                raise ValueError(f"Inconsistent timestep convention for {key}")
            previous["ranks"].append(call["rank"])
    branches = (False, True) if uses_cfg else (False,)
    expected = [(step, branch) for step in range(steps) for branch in branches]
    if sorted(unique) != expected:
        raise ValueError(
            f"Incomplete transformer captures: {sorted(unique)} != {expected}"
        )
    calls = [unique[key] for key in expected]
    for call in calls:
        for name, value in call.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError(f"Non-finite capture at step {call['step']}/{name}")
    first = [call for call in calls if call["step"] == 0]
    return {
        "capture_metadata": [
            {
                name: call[name]
                for name in ("step", "is_cfg_negative", "ranks", "timestep_convention")
            }
            for call in first
        ],
        "encoder_hidden_states": [call["encoder_hidden_states"] for call in first],
        "predictions": [call["prediction"] for call in first],
        "timestep": [call["timestep"] for call in first],
        "effective_timestep": [call["effective_timestep"] for call in first],
        "timestep_trace": [
            {
                name: call[name]
                for name in (
                    "step",
                    "is_cfg_negative",
                    "ranks",
                    "timestep_convention",
                    "timestep",
                    "effective_timestep",
                )
            }
            for call in calls
        ],
    }


@torch.no_grad()
def reference_run(args):
    import diffusers
    from diffusers import OvisImagePipeline
    from torch.nn.attention import SDPBackend, sdpa_kernel

    pipeline = OvisImagePipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    pipeline.transformer.set_attention_backend("native")
    if args.vae_tiling:
        pipeline.vae.enable_tiling()
    record = {"calls": []}
    state = {"step": 0, "branch_calls": 0}

    def call_context():
        branch = state["branch_calls"]
        state["branch_calls"] += 1
        return state["step"], branch == 1

    handles = capture_transformer(
        pipeline.transformer, record, call_context, "normalized"
    )
    latent = initial_latents(args)
    packed = pipeline._pack_latents(
        latent, sample_count(args), 16, args.height // 8, args.width // 8
    )
    record["initial_latents"] = packed
    trajectory = []
    trajectory_timesteps = []

    def callback(pipeline, step, timestep, kwargs):
        trajectory.append(kwargs["latents"].detach().cpu().clone())
        trajectory_timesteps.append(timestep.detach().cpu().clone())
        state.update(step=step + 1, branch_calls=0)
        return kwargs

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        # SGLang's CUDA platform disables CuDNN SDPA. Keep the independent
        # oracle on the same backend set without changing its process policy.
        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            reference_sdpa_flags = sdpa_flags()
            image = pipeline(
                prompt=prompts(args),
                negative_prompt="",
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                max_sequence_length=args.text_length,
                num_images_per_prompt=args.outputs,
                generator=torch.Generator("cpu").manual_seed(args.seed),
                latents=packed.cuda(),
                output_type="pt",
                callback_on_step_end=callback,
            ).images
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    seconds = time.perf_counter() - start
    record.update(finalize_captures([record], args.steps, args.guidance > 1))
    del record["calls"]
    record["trajectory"] = torch.stack(trajectory, dim=1)
    record["trajectory_timesteps"] = torch.stack(trajectory_timesteps)
    record["images"] = image.cpu()
    return record, {
        "seconds": seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "diffusers": diffusers.__version__,
        "source": source_provenance(diffusers.__file__),
        "resolved_config": jsonable(
            {
                "transformer": dict(pipeline.transformer.config),
                "text_encoder": pipeline.text_encoder.config.to_dict(),
                "vae": dict(pipeline.vae.config),
                "scheduler": dict(pipeline.scheduler.config),
                "attention_backend": "native",
                "sdpa_policy": "flash/efficient/math; CuDNN excluded to match SGLang",
                "sdpa_flags": reference_sdpa_flags,
                "parallelism": {"world_size": 1, "tp": 1, "sp": 1, "cfg": 1},
                "vae_tiling": pipeline.vae.use_tiling,
                "vae_sp": False,
                "prompt_batch": prompts(args),
                "sample_batch_size": sample_count(args),
            }
        ),
    }


@torch.no_grad()
def native_run(args):
    from sglang.multimodal_gen.configs.pipeline_configs.ovis_image import (
        OvisImagePipelineConfig,
    )
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        get_world_group,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context,
    )
    from sglang.multimodal_gen.runtime.pipelines.ovis_image import OvisImagePipeline
    from sglang.multimodal_gen.runtime.server_args import (
        ServerArgs,
        set_global_server_args,
    )
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    world = args.tp * args.ulysses * args.ring * args.cfg
    config = OvisImagePipelineConfig(vae_tiling=args.vae_tiling, vae_sp=args.vae_sp)
    config.vae_config.use_parallel_decode = args.vae_sp
    config.vae_config.parallel_decode_mode = "spatial_shard" if args.vae_sp else "auto"
    server = ServerArgs(
        model_path=args.model_path,
        num_gpus=world,
        tp_size=args.tp,
        pipeline_config=config,
        ulysses_degree=args.ulysses,
        ring_degree=args.ring,
        enable_cfg_parallel=args.cfg == 2,
        attention_backend=args.attention,
        warmup_mode="off",
        performance_mode="manual",
        use_fsdp_inference=False,
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        dit_cpu_offload=args.offload == "component",
        text_encoder_cpu_offload=args.offload == "component",
        vae_cpu_offload=args.offload == "component",
        dit_layerwise_offload=args.offload == "layerwise",
        layerwise_offload_components=["transformer", "text_encoder", "vae"]
        if args.offload == "layerwise"
        else [],
    )
    set_global_server_args(server)
    ensure_distributed_env_defaults()
    provenance = source_provenance(Path(__file__).resolve())
    handles = []
    try:
        maybe_init_distributed_environment_and_model_parallel(
            tp_size=args.tp,
            sp_size=args.ulysses * args.ring,
            cfg_degree=args.cfg,
            ulysses_degree=args.ulysses,
            ring_degree=args.ring,
        )
        from sglang.srt.runtime_context import publish
        from sglang.srt.server_args import ServerArgs as SrtServerArgs

        publish(
            SrtServerArgs(model_path="dummy", tp_size=args.tp),
            role="diffusion_gpu_worker",
        )
        pipeline = OvisImagePipeline(args.model_path, server)
        encoder_tp_group = resolved_encoder_tp_group(
            pipeline.get_module("text_encoder")
        )
        if server.has_layerwise_offload_components():
            from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
                get_global_component_residency_manager,
            )
            from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
                configure_layerwise_offload_modules,
            )

            configure_layerwise_offload_modules(
                pipeline.modules,
                server,
                pin_budget=get_global_component_residency_manager(
                    pipeline, server
                ).host_pin_budget,
                component_names=server.layerwise_offload_components,
            )
        sampling = SamplingParams.from_user_sampling_params_args(
            args.model_path,
            server_args=server,
            prompt=args.prompt,
            negative_prompt="",
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            max_sequence_length=args.text_length,
            num_outputs_per_prompt=args.outputs,
            seed=args.seed,
            generator_device="cpu",
            return_trajectory_latents=True,
            save_output=False,
        )
        batch = prepare_request(server_args=server, sampling_params=sampling)
        # prepare_request accepts image prompt strings at the public boundary.
        # Req forwards this assignment to SamplingParams; downstream stages
        # support an internal list batch, whose sample order is [A,A,B,B].
        batch.prompt = prompts(args)
        batch.latents = initial_latents(args)
        record = {"calls": []}

        def call_context():
            context = get_forward_context()
            return context.current_timestep, context.forward_batch.is_cfg_negative

        handles = capture_transformer(
            pipeline.get_module("transformer"), record, call_context, "scheduler_raw"
        )
        native_sdpa_flags = sdpa_flags()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = pipeline.forward(batch, server)
        torch.cuda.synchronize()
        metrics = {
            "seconds": time.perf_counter() - start,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        rank = torch.distributed.get_rank()
        record["rank_metrics"] = {
            "rank": rank,
            "sdpa_flags": native_sdpa_flags,
            "resolved_encoder_tp_group": encoder_tp_group,
            **metrics,
        }
        record["source"] = provenance
        records = [None] * torch.distributed.get_world_size() if rank == 0 else None
        torch.distributed.gather_object(
            record, records, dst=0, group=get_world_group().cpu_group
        )
        if rank == 0:
            metrics["rank_sources"] = validate_rank_sources(records)
            metrics["rank_metrics"] = [part["rank_metrics"] for part in records]
            metrics["seconds"] = max(
                part["seconds"] for part in metrics["rank_metrics"]
            )
            metrics["peak_memory_bytes"] = max(
                part["peak_memory_bytes"] for part in metrics["rank_metrics"]
            )
            record = finalize_captures(
                records, args.steps, batch.do_classifier_free_guidance
            )
            record["initial_latents"] = server.pipeline_config.maybe_pack_latents(
                initial_latents(args), sample_count(args), batch
            )
            record["trajectory"] = result.trajectory_latents
            record["trajectory_timesteps"] = result.trajectory_timesteps
            record["images"] = result.output.detach().cpu()
            metrics.update(
                source=provenance,
                resolved_config=jsonable(server),
            )
            metrics["resolved_config"]["sdpa_flags"] = native_sdpa_flags
            metrics["resolved_config"]["resolved_encoder_tp_group"] = encoder_tp_group
        else:
            record = {}
        return record, metrics
    finally:
        for handle in handles:
            handle.remove()
        if torch.distributed.is_initialized():
            cleanup_dist_env_and_memory()


def assert_close_error(actual, expected, *, atol, rtol):
    """Use the same dtype checks and tolerance predicate as the acceptance gate."""
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        return str(error)
    return None


def tensor_errors(actual, expected, label):
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise TypeError(f"{label} must contain tensors")
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: shapes differ: {actual.shape} != {expected.shape}")
    if not actual.numel():
        raise ValueError(f"{label}: empty tensor")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError(f"{label}: non-finite tensor")
    delta = actual.float() - expected.float()
    atol, rtol = (
        (1e-4, 1e-4)
        if actual.dtype == expected.dtype == torch.float32
        else (0.05, 0.02)
    )
    tolerance = atol + rtol * expected.float().abs()
    close_error = assert_close_error(actual, expected, atol=atol, rtol=rtol)
    return {
        "shape": list(actual.shape),
        "actual_dtype": str(actual.dtype),
        "reference_dtype": str(expected.dtype),
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "atol": atol,
        "rtol": rtol,
        "mismatched_elements": (delta.abs() > tolerance).sum().item(),
        "within_tolerance": close_error is None,
        "exact_equal": actual.dtype == expected.dtype and torch.equal(actual, expected),
    }


def matching_list(actual, expected, label):
    if not isinstance(actual, list) or not isinstance(expected, list):
        raise TypeError(f"{label} must be lists")
    if len(actual) != len(expected) or not actual:
        raise ValueError(
            f"{label}: lengths differ or are empty: {len(actual)}, {len(expected)}"
        )
    return zip(actual, expected)


def scheduler_units(call):
    timestep = call["timestep"].float()
    convention = call["timestep_convention"]
    if convention == "normalized":
        return timestep * 1000
    if convention == "scheduler_raw":
        return timestep
    raise ValueError(f"Unknown timestep convention: {convention}")


def load_comparison_metrics(directory):
    path = directory / "metrics.json"
    if not path.is_file():
        raise ValueError(f"Missing metrics.json: {path}; rerun this artifact")
    try:
        metrics = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read metrics.json: {path}: {error}") from error
    if not isinstance(metrics, dict) or not isinstance(metrics.get("arguments"), dict):
        raise ValueError(f"Invalid metrics.json arguments: {path}")
    missing = [key for key in COMPARISON_ARGUMENTS if key not in metrics["arguments"]]
    if missing or not metrics.get("model_revision"):
        raise ValueError(
            f"Incomplete metrics.json: {path}; missing "
            + ", ".join(
                missing + ([] if metrics.get("model_revision") else ["model_revision"])
            )
        )
    if metrics["arguments"].get("mode") not in ("native", "reference"):
        raise ValueError(f"Invalid metrics.json runtime mode: {path}")
    return metrics


def check_comparison_metadata(actual, expected, report):
    configuration = {}
    errors = []
    for key in COMPARISON_ARGUMENTS:
        a, b = actual["arguments"][key], expected["arguments"][key]
        configuration[key] = {"actual": a, "reference": b, "matches": a == b}
        if a != b:
            errors.append(f"Sampling argument {key} differs: {a!r} != {b!r}")
    a, b = actual["model_revision"], expected["model_revision"]
    configuration["model_revision"] = {"actual": a, "reference": b, "matches": a == b}
    if a != b:
        errors.append(f"Model revision differs: {a!r} != {b!r}")
    report["configuration"] = configuration
    report["source"] = {
        "actual": actual.get("source"),
        "reference": expected.get("source"),
    }
    # Native runs use the same SGLang repository even when worktree paths differ.
    # A Diffusers reference is a different repository and intentionally differs.
    if actual["arguments"]["mode"] == expected["arguments"]["mode"] == "native":
        for key in ("commit", "diff_sha256"):
            a = (actual.get("source") or {}).get(key)
            b = (expected.get("source") or {}).get(key)
            if not isinstance(a, str) or not a or not isinstance(b, str) or not b:
                errors.append(f"Native source {key} is missing")
            elif a != b:
                errors.append(f"Native source {key} differs: {a!r} != {b!r}")
    if errors:
        raise ValueError("; ".join(errors))


def comparison_details(args, report):
    actual_metrics = load_comparison_metrics(args.output)
    expected_metrics = load_comparison_metrics(args.reference)
    check_comparison_metadata(actual_metrics, expected_metrics, report)
    expected = torch.load(
        args.reference / "tensors.pt", weights_only=True, map_location="cpu"
    )
    actual = torch.load(
        args.output / "tensors.pt", weights_only=True, map_location="cpu"
    )
    if actual.get("capture_schema") != 2 or expected.get("capture_schema") != 2:
        raise ValueError("Capture schema changed; rerun both runtimes with this runner")
    for a, b in matching_list(
        actual["capture_metadata"], expected["capture_metadata"], "capture_metadata"
    ):
        for name in ("step", "is_cfg_negative"):
            if a[name] != b[name]:
                raise ValueError(f"Capture ordering differs for {name}: {a} != {b}")
    for key in (
        "initial_latents",
        "encoder_hidden_states",
        "predictions",
        "effective_timestep",
        "trajectory",
        "trajectory_timesteps",
        "images",
    ):
        pairs = (
            matching_list(actual[key], expected[key], key)
            if isinstance(actual[key], list)
            else [(actual[key], expected[key])]
        )
        report[key] = [
            tensor_errors(a, b, f"{key}[{i}]") for i, (a, b) in enumerate(pairs)
        ]
    report["timestep_trace"] = []
    for a, b in matching_list(
        actual["timestep_trace"], expected["timestep_trace"], "timestep_trace"
    ):
        for name in ("step", "is_cfg_negative"):
            if a[name] != b[name]:
                raise ValueError(f"Timestep ordering differs for {name}: {a} != {b}")
        label = f"step={a['step']}, negative={a['is_cfg_negative']}"
        tensor_errors(a["timestep"], b["timestep"], label)
        report["timestep_trace"].append(
            {
                "step": a["step"],
                "is_cfg_negative": a["is_cfg_negative"],
                "actual_input_convention": a["timestep_convention"],
                "reference_input_convention": b["timestep_convention"],
                "actual_input": a["timestep"].float().tolist(),
                "reference_input": b["timestep"].float().tolist(),
                "actual_effective": a["effective_timestep"].float().tolist(),
                "reference_effective": b["effective_timestep"].float().tolist(),
                "input_in_scheduler_units": tensor_errors(
                    scheduler_units(a), scheduler_units(b), label
                ),
                "effective": tensor_errors(
                    a["effective_timestep"], b["effective_timestep"], label
                ),
            }
        )
    a, b = actual["images"].float(), expected["images"].float()
    if a.ndim != 4 or a.shape[1] != 3:
        raise ValueError(f"Images must have shape [B, 3, H, W], got {a.shape}")
    mse = (a - b).square().mean().item()
    report["psnr"] = "inf" if mse == 0 else float(-10 * np.log10(mse))
    if torch.equal(a, b):
        report["ssim"] = 1.0
    else:
        from skimage.metrics import structural_similarity

        report["ssim"] = float(
            np.mean(
                [
                    structural_similarity(
                        x.permute(1, 2, 0).numpy(),
                        y.permute(1, 2, 0).numpy(),
                        channel_axis=2,
                        data_range=1,
                    )
                    for x, y in zip(a, b)
                ]
            )
        )
    checks = [
        (
            "initial_latents",
            actual["initial_latents"],
            expected["initial_latents"],
            0,
            0,
        )
    ]
    # Conditioning and first-step DiT predictions have component oracles.
    # Later BF16 trajectories and decoded images accumulate rounding differences;
    # retain their errors and PSNR/SSIM for review instead of inventing a bound.
    for key in ("encoder_hidden_states", "predictions"):
        for index, pair in enumerate(zip(actual[key], expected[key])):
            atol = report[key][index]["atol"]
            rtol = report[key][index]["rtol"]
            checks.append((f"{key}[{index}]", *pair, atol, rtol))
    for index, pair in enumerate(
        matching_list(
            actual["effective_timestep"],
            expected["effective_timestep"],
            "effective_timestep",
        )
    ):
        checks.append((f"effective_timestep[{index}]", *pair, 0, 0))
    for a, b in matching_list(
        actual["timestep_trace"], expected["timestep_trace"], "timestep_trace"
    ):
        checks.append(
            (
                f"effective_timestep[step={a['step']}, negative={a['is_cfg_negative']}]",
                a["effective_timestep"],
                b["effective_timestep"],
                0,
                0,
            )
        )
    return checks


def write_comparison_report(args, report):
    result = json.dumps(report, indent=2, allow_nan=False)
    (args.output / args.comparison_name).write_text(result + "\n")
    print(result)


def compare(args):
    report = {
        "reference": str(args.reference),
        "actual": str(args.output),
        "acceptance": False,
        "acceptance_errors": [],
    }
    try:
        checks = comparison_details(args, report)
    except Exception as error:
        report["acceptance_errors"].append(str(error))
        write_comparison_report(args, report)
        raise
    for label, actual, expected, atol, rtol in checks:
        error = assert_close_error(actual, expected, atol=atol, rtol=rtol)
        if error is not None:
            report["acceptance_errors"].append(f"{label}: {error}")
    report["acceptance"] = not report["acceptance_errors"]
    write_comparison_report(args, report)
    # Persist diagnostics and exact_equal before an acceptance assertion raises.
    for _, actual, expected, atol, rtol in checks:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return report


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "compare":
        compare(args)
        return
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    record, metrics = (
        reference_run(args) if args.mode == "reference" else native_run(args)
    )
    if int(os.environ.get("RANK", 0)) == 0:
        images = record["images"]
        if images.ndim == 5:
            images = images.squeeze(2)
        # Both runtimes' image processors normalize and clamp to [0, 1].
        record["images"] = images.float()
        record["capture_schema"] = 2
        torch.save(record, args.output / "tensors.pt")
        for i, image in enumerate(images):
            pixels = (
                (image.float().clamp(0, 1).permute(1, 2, 0).numpy() * 255)
                .round()
                .astype(np.uint8)
            )
            Image.fromarray(pixels).save(args.output / f"image-{i}.png")
        metrics.update(
            {
                "arguments": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(),
                "model_revision": MODEL_REVISION,
                "reference_revision": REFERENCE_REVISION,
                "capture_schema": 2,
            }
        )
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
