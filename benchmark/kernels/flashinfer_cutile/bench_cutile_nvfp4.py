"""Reproducible NVFP4 MoE layer checks and measurements on SM120/SM121.

This is a layer benchmark, not a model-serving benchmark. Marlin is selected
explicitly as a W4A16 reference; the normal SM12x model dispatch is not changed.
Run --help on a CPU host. GPU packages are imported only after argument parsing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOKENS = (1, 64, 412, 4096, 8192, 8193, 16384)


def percentile(values, percent):
    """Linear interpolation, including the endpoints; reject invalid timings."""
    if not values or not 0 <= percent <= 100:
        raise ValueError("percentile needs samples and a percentile in [0, 100]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("samples must be finite and nonnegative")
    position = (len(ordered) - 1) * percent / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_rank_samples(rank_samples):
    """Take the slowest rank for EACH call before computing percentiles.

    Taking max(rank p50) would hide stragglers that move between ranks.
    """
    if not rank_samples or not rank_samples[0]:
        raise ValueError("rank samples cannot be empty")
    if any(len(row) != len(rank_samples[0]) for row in rank_samples):
        raise ValueError("all ranks must report the same number of samples")
    for row in rank_samples:
        percentile(row, 50)
    samples = [max(values) for values in zip(*rank_samples)]
    return {
        "samples_ms": samples,
        "p50_ms": percentile(samples, 50),
        "p95_ms": percentile(samples, 95),
        "mean_ms": statistics.fmean(samples),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("latency", "workspace"), default="latency")
    parser.add_argument(
        "--cutile-api",
        choices=("runner-core", "fused"),
        default="runner-core",
        help="Explicit interface adapter for the target checkout; no automatic fallback",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-dir", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--checkpoint-revision", help="Checkpoint commit SHA (recorded claim)"
    )
    parser.add_argument("--layer-index", type=int, default=30)
    parser.add_argument("--experts", type=int, choices=(32, 128), default=32)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument("--capacity", type=int, default=16384)
    parser.add_argument("--input-std", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("cutile", "marlin", "direct"),
        default=["cutile", "marlin"],
    )
    parser.add_argument(
        "--execution", nargs="+", choices=("eager", "graph"), default=["eager", "graph"]
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reference-tokens", type=int, default=64)
    parser.add_argument("--max-relative-l2", type=float, default=0.05)
    parser.add_argument(
        "--distributed-timeout",
        type=float,
        default=600,
        help="NCCL initialization/collective timeout in seconds; run TP with torchrun",
    )
    parser.add_argument(
        "--streams",
        type=int,
        default=2,
        help="Sequential stream visits in workspace mode",
    )
    parser.add_argument(
        "--implementation-label",
        required=True,
        help="Human-readable implementation label; SHA is recorded separately",
    )
    parser.add_argument(
        "--flashinfer-variant",
        required=True,
        help="For example stock-0.7.3 or 0.7.3-plus-9129-<patch-SHA>",
    )
    parser.add_argument(
        "--flashinfer-source",
        type=Path,
        help="Optional FlashInfer source checkout to fingerprint",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Must not exist; never overwrite previous evidence",
    )
    args = parser.parse_args(argv)
    if args.samples < 100 or args.warmup < 1:
        parser.error("--samples must be >=100 and --warmup must be positive")
    if not args.tokens or any(n < 1 or n > args.capacity for n in args.tokens):
        parser.error("each token count must be positive and <= --capacity")
    if args.capacity < 1 or args.reference_tokens < 1 or args.streams < 1:
        parser.error("capacity, reference-tokens and streams must be positive")
    if not math.isfinite(args.input_std) or args.input_std <= 0:
        parser.error("--input-std must be finite and positive")
    if not math.isfinite(args.max_relative_l2) or not 0 < args.max_relative_l2 < 1:
        parser.error("--max-relative-l2 must be between zero and one")
    if not math.isfinite(args.distributed_timeout) or args.distributed_timeout <= 0:
        parser.error("--distributed-timeout must be finite and positive")
    if args.layer_index < 0 or args.seed < 0:
        parser.error("layer-index and seed must be nonnegative")
    if args.checkpoint_dir and not args.checkpoint_revision:
        parser.error("real weights require --checkpoint-revision for provenance")
    if len(set(args.backends)) != len(args.backends) or len(set(args.execution)) != len(
        args.execution
    ):
        parser.error("duplicate backends or execution modes are not allowed")
    return args


def load_test_helpers():
    """Load this checkout's GPU test helpers without importing a top-level test package."""
    # The repository's test directory is a namespace package. Python's own
    # regular "test" package can shadow it even with REPO_ROOT first on sys.path.
    # Use a private module name and a fixed path; leave the existing test package
    # untouched. Call only after argument parsing so --help stays lightweight.
    path = REPO_ROOT / "test/manual/layers/moe/test_flashinfer_cutile_nvfp4.py"
    spec = importlib.util.spec_from_file_location(
        "_sglang_cutile_nvfp4_test_helpers", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load benchmark helpers from {path}")
    helpers = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helpers
    spec.loader.exec_module(helpers)
    return helpers


def command_output(argv, cwd=None):
    """Read-only diagnostics; a missing optional executable is recorded."""
    try:
        result = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=30, check=False
        )
        return {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": str(error)}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_fingerprint(directory):
    head = command_output(
        ["git", "--no-optional-locks", "rev-parse", "HEAD"], directory
    )
    status = command_output(
        ["git", "--no-optional-locks", "status", "--porcelain"], directory
    )
    diff = command_output(
        ["git", "--no-optional-locks", "diff", "--binary", "HEAD"], directory
    )
    return {
        "head": head,
        "status": status,
        "tracked_diff_sha256": hashlib.sha256(
            diff.get("stdout", "").encode()
        ).hexdigest(),
        "diff_read_returncode": diff.get("returncode"),
        "note": "Untracked files are listed in status; harness files also have content hashes.",
    }


def create_output_dir(path):
    # mkdir(exist_ok=False) makes parallel/repeated runs fail rather than append
    # unrelated results into a directory that looks like a single experiment.
    path.mkdir(parents=True, exist_ok=False)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class BenchmarkCheckError(AssertionError):
    """Retain numerical evidence when a correctness gate stops a run."""

    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


class RunEvidence:
    """Write each rank's last stage outside timing, including failed checks."""

    def __init__(self, output, rank):
        self.path = output / f"status-rank{rank}.json"
        self.current = {"rank": rank}

    def stage(self, phase, **details):
        self.current.update(details)
        self.current["phase"] = phase
        write_json(self.path, {"complete": False, "running": True, **self.current})

    def fail(self, error):
        write_json(
            self.path,
            {
                "complete": False,
                "running": False,
                **self.current,
                "error": repr(error),
                "metrics": getattr(error, "metrics", None),
            },
        )

    def succeed(self):
        write_json(self.path, {"complete": True, "running": False, **self.current})


def distributed_timeout(args):
    return timedelta(seconds=args.distributed_timeout)


def exit_distributed_failure():
    """Let torchrun terminate peers without waiting in NCCL destructors.

    The caller has already saved evidence and printed the original traceback.
    Do not add a collective or destroy_process_group here: another rank can be
    stuck in a different operation. CPU tests can check this control flow, but
    actual watchdog/peer termination still requires GPU fault-injection tests.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(1)


def report_failure(evidence, error, *, distributed_worker):
    """Preserve the original exception even if saving evidence itself fails."""
    try:
        evidence.fail(error)
    except Exception as reporting_error:
        print(f"Could not save failure evidence: {reporting_error!r}", file=sys.stderr)
    if distributed_worker:
        traceback.print_exception(type(error), error, error.__traceback__)
        exit_distributed_failure()


def metadata(args, torch, rank, world_size):
    packages = {}
    for name in (
        "torch",
        "flashinfer-python",
        "cuda-tile",
        "nvidia-cuda-tileiras",
        "nvidia-cuda-nvcc",
        "nvidia-nvvm",
        "sgl-kernel",
        "safetensors",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import flashinfer

    import sglang
    from sglang.srt.layers.moe.moe_runner import flashinfer_cutile

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    files = [
        Path(__file__),
        REPO_ROOT / "test/manual/layers/moe/test_flashinfer_cutile_nvfp4.py",
        REPO_ROOT / "python/sglang/srt/layers/moe/moe_runner/flashinfer_cutile.py",
        REPO_ROOT / "python/sglang/srt/layers/moe/moe_runner/marlin.py",
    ]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "module_paths": {
            "sglang": sglang.__file__,
            "cutile_adapter": flashinfer_cutile.__file__,
            "flashinfer": flashinfer.__file__,
        },
        "python": sys.version,
        "platform": platform.platform(),
        "rank": rank,
        "world_size": world_size,
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "gpu": {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "sm_count": properties.multi_processor_count,
            "capability": list(torch.cuda.get_device_capability()),
        },
        "nvidia_smi": command_output(["nvidia-smi"]),
        "topology": command_output(["nvidia-smi", "topo", "-m"]),
        "git": git_fingerprint(REPO_ROOT),
        "file_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in files
        },
        "communication_env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("NCCL_", "TORCH_NCCL_")) or key == "CUDA_VISIBLE_DEVICES"
        },
        "protocol": {
            "reference": "FP32 matmuls with dequantized weights; W4A4 and BF16 activation variants",
            "routing": "fixed seed, random distinct top-k experts, normalized FP32 weights",
            "inputs": "four rotating BF16 input/routing addresses and four graphs; no copies in measurement",
            "collective": "NCCL all_reduce(SUM) included in layer timing when TP>1",
            "timing": "per-call CUDA events with end-event synchronization; not pipelined throughput",
            "comparison": "same machine, environment and inputs; alternate backend order by token case",
            "coverage": "synthetic routing on one MoE layer, not model-generated routing or full serving",
            "cutile_entrypoint": (
                "FlashInferCuTileRunnerCore.run_from_dispatch"
                if args.cutile_api == "runner-core"
                else "fused_experts_none_to_flashinfer_cutile"
            ),
        },
    }
    if args.flashinfer_source:
        result["flashinfer_git"] = git_fingerprint(args.flashinfer_source)
    if args.checkpoint_dir:
        result["checkpoint"] = {
            "claimed_revision": args.checkpoint_revision,
            "revision_verified": False,
            "file_sha256": {
                name: sha256_file(args.checkpoint_dir / name)
                for name in ("config.json", "model.safetensors.index.json")
            },
            "note": "Match local shards to the claimed revision before publishing; index hashes alone do not prove shard contents.",
        }
    return result


def runtime_scope(capacity):
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
    from sglang.srt.runtime_context import (
        get_context,
        get_exec,
        get_flags,
        reset_context,
    )

    stack = ExitStack()
    reset_context()
    stack.callback(reset_context)
    stack.enter_context(
        get_context().override_server_args(
            max_prefill_tokens=capacity, disable_cuda_graph=True
        )
    )
    stack.enter_context(get_exec().graph.override(cuda_graph_config=None))
    stack.enter_context(
        get_flags().moe.override(
            runner_backend=MoeRunnerBackend.FLASHINFER_CUTILE,
            a2a_backend=MoeA2ABackend.NONE,
        )
    )
    return stack


def clone_layer(layer, helpers):
    result = helpers._layer(
        {
            name: parameter.detach().clone()
            for name, parameter in layer.named_parameters()
        },
        top_k=layer.moe_runner_config.top_k,
    )
    result.moe_tp_size, result.moe_tp_rank = layer.moe_tp_size, layer.moe_tp_rank
    # Production FusedMoE layers carry the quantization group size. The small
    # standalone fixture must supply it too for the real Marlin preparation.
    result.quant_config = SimpleNamespace(group_size=16)
    return result


def build_backend(name, source, helpers, capacity, cutile_api="runner-core"):
    """Use independent weights because both preparation paths can replace them."""
    import torch

    layer = clone_layer(source, helpers)
    config = layer.moe_runner_config
    if name == "cutile":
        from sglang.srt.layers.moe.moe_runner import flashinfer_cutile

        if cutile_api == "runner-core":
            required = (
                "FlashInferCuTileRunnerCore",
                "prepare_flashinfer_cutile_nvfp4_weights",
            )
            if not all(hasattr(flashinfer_cutile, attr) for attr in required):
                raise ValueError(
                    "target checkout does not expose the selected runner-core API"
                )
            prepared = flashinfer_cutile.prepare_flashinfer_cutile_nvfp4_weights(
                layer, config
            )
            runner = flashinfer_cutile.FlashInferCuTileRunnerCore(config)

            def call(dispatch):
                return runner.run_from_dispatch(
                    dispatch, prepared, config
                ).hidden_states

        else:
            required = (
                "build_cutile_nvfp4_quant_info",
                "fused_experts_none_to_flashinfer_cutile",
            )
            if not all(hasattr(flashinfer_cutile, attr) for attr in required):
                raise ValueError(
                    "target checkout does not expose the selected fused API"
                )
            # This is an API shim, not a copied implementation. Both checkouts
            # consume the same checkpoint tensors, routing and timed call scope.
            prepared = flashinfer_cutile.build_cutile_nvfp4_quant_info(
                w13_packed=layer.w13_weight_packed,
                w13_block_scale=layer.w13_weight_scale,
                w13_weight_global_scale=layer.w13_weight_global_scale,
                w13_input_global_scale=layer.w13_input_global_scale,
                w2_packed=layer.w2_weight_packed,
                w2_block_scale=layer.w2_weight_scale,
                w2_weight_global_scale=layer.w2_weight_global_scale,
                w2_input_global_scale=layer.w2_input_global_scale,
                top_k=config.top_k,
                swiglu_alpha=helpers.ALPHA,
                swiglu_beta=helpers.BETA,
                swiglu_limit=helpers.LIMIT,
            )

            def call(dispatch):
                return flashinfer_cutile.fused_experts_none_to_flashinfer_cutile(
                    dispatch, prepared, config
                ).hidden_states

    elif name == "direct":
        from flashinfer.fused_moe import MoEActivationPack
        from flashinfer.fused_moe.core import RoutingInputMode

        runner, prepared = helpers._direct_flashinfer(layer, capacity=capacity)

        def call(dispatch):
            return runner(
                MoEActivationPack(
                    hidden_states_q=dispatch.hidden_states,
                    hidden_states_scale=None,
                    topk_ids=dispatch.topk_output.topk_ids,
                    topk_weights=dispatch.topk_output.topk_weights,
                    routing_input_mode=RoutingInputMode.PackedPrecomputed,
                ),
                prepared,
            )

    else:
        from sglang.srt.layers.moe.moe_runner.marlin import fused_experts_none_to_marlin
        from sglang.srt.layers.quantization.minimax_m3_nvfp4 import MiniMaxM3Nvfp4MoE

        scales = layer.w13_weight_global_scale
        if not torch.equal(scales[:, 0], scales[:, 1]):
            raise ValueError(
                "Marlin baseline needs equal gate/up global scales; do not silently drop one."
            )
        # Bypass only backend selection, not preparation or execution: this
        # benchmark explicitly selects the existing Marlin W4A16 implementation.
        # Constructing the normal scheme on SM12x would select a different path.
        scheme = object.__new__(MiniMaxM3Nvfp4MoE)
        scheme.moe_runner_config = config
        layer.dispatcher = None
        scheme._process_marlin_weights(layer)
        prepared = scheme.get_marlin_quant_info(layer)

        def call(dispatch):
            return fused_experts_none_to_marlin(
                dispatch, prepared, config
            ).hidden_states

    # Retain source geometry/weights until all captures using this callable die.
    call.layer = layer
    return call


def make_inputs(layer, tokens, seed, input_std, *, sets=4):
    import torch

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    generator = torch.Generator(device="cuda").manual_seed(seed)
    config = layer.moe_runner_config
    values = []
    for _ in range(sets):
        x = (
            torch.randn(
                tokens,
                config.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            * input_std
        )
        # Sorting random scores gives distinct expert IDs per token. Using
        # randint would accidentally test duplicate routes instead of top-k.
        scores = torch.rand(
            tokens, config.num_experts, device="cuda", generator=generator
        )
        selected, ids = scores.topk(config.top_k, dim=-1)
        weights = selected.softmax(dim=-1).float().contiguous()
        values.append(
            StandardDispatchOutput(
                hidden_states=x.contiguous(),
                hidden_states_scale=None,
                topk_output=StandardTopKOutput(
                    topk_ids=ids.to(torch.int32).contiguous(),
                    topk_weights=weights,
                    # Retain the actual pre-top-k scores for Marlin's input
                    # contract; do not fabricate logits or change the routes.
                    router_logits=scores,
                ),
            )
        )
    return values


def copy_input(destination, source):
    destination.hidden_states.copy_(source.hidden_states)
    destination.topk_output.topk_ids.copy_(source.topk_output.topk_ids)
    destination.topk_output.topk_weights.copy_(source.topk_output.topk_weights)
    # Keep captured storage addresses stable while updating the whole route.
    destination.topk_output.router_logits.copy_(source.topk_output.router_logits)


def slice_input(dispatch, tokens):
    return dispatch._replace(
        hidden_states=dispatch.hidden_states[:tokens],
        topk_output=dispatch.topk_output._replace(
            topk_ids=dispatch.topk_output.topk_ids[:tokens],
            topk_weights=dispatch.topk_output.topk_weights[:tokens],
            router_logits=dispatch.topk_output.router_logits[:tokens],
        ),
    )


def reduce_output(output, dist, world_size):
    if world_size > 1:
        dist.all_reduce(output)
    return output


def error_metrics(actual, expected):
    import torch

    delta = actual.float() - expected.float()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("nonfinite result or reference")
    return {
        "relative_l2": (delta.norm() / expected.float().norm().clamp_min(1e-8)).item(),
        "max_abs": delta.abs().max().item(),
        "rmse": delta.square().mean().sqrt().item(),
    }


def bit_equal(actual, expected):
    """Compare storage bits as well as values, including the sign of zero."""
    import torch

    return (
        actual.dtype == expected.dtype
        and actual.shape == expected.shape
        and torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    )


def validate_output(actual, expected_shape):
    """Check the whole output even when the expensive reference is a prefix."""
    import torch

    if tuple(actual.shape) != tuple(expected_shape):
        raise BenchmarkCheckError(
            "output shape mismatch",
            {
                "actual_shape": list(actual.shape),
                "expected_shape": list(expected_shape),
            },
        )
    finite = torch.isfinite(actual)
    if not finite.all().item():
        raise BenchmarkCheckError(
            "nonfinite output outside or inside the reference prefix",
            {"nonfinite_elements": (~finite).sum().item()},
        )


def repeat_metrics(name, actual, expected):
    """Keep the original gates and report every check's numerical difference."""
    validate_output(actual, expected.shape)
    metrics = {
        **error_metrics(actual, expected),
        "bit_identity": bit_equal(actual, expected),
    }
    if name == "marlin":
        # Atomic reductions permit bounded numerical noise, not arbitrary error.
        if metrics["relative_l2"] > 0.001:
            raise BenchmarkCheckError(
                "Marlin repeat/replay differs beyond reduction noise",
                {**metrics, "relative_l2_limit": 0.001},
            )
    elif not metrics["bit_identity"]:
        raise BenchmarkCheckError(
            f"{name}: repeat/replay is not bit-identical", metrics
        )
    return metrics


def assert_rank_agreement(output, dist, world_size):

    if world_size > 1:
        rank_zero = output.clone()
        dist.broadcast(rank_zero, src=0)
        if not bit_equal(output, rank_zero):
            raise BenchmarkCheckError(
                "NCCL-reduced outputs differ between ranks",
                {**error_metrics(output, rank_zero), "bit_identity": False},
            )


def reference_cases(layer, inputs, args, helpers, dist, world_size):
    """Bound FP32 reference cost; exact adapter/direct checks cover full tensors."""
    references = []
    for dispatch in inputs:
        small = slice_input(
            dispatch, min(args.reference_tokens, dispatch.hidden_states.shape[0])
        )
        x, ids, weights = (
            small.hidden_states,
            small.topk_output.topk_ids,
            small.topk_output.topk_weights,
        )
        quantized, stats = helpers._reference(layer, x, ids, weights, return_stats=True)
        bf16 = helpers._reference(layer, x, ids, weights, quantize_activations=False)
        reduce_output(quantized, dist, world_size)
        reduce_output(bf16, dist, world_size)
        references.append((quantized, bf16, stats))
    return references


def measure_backend(
    name, call, inputs, references, args, dist, world_size, stream, evidence, on_result
):
    import torch

    static = make_inputs(
        call.layer,
        inputs[0].hidden_states.shape[0],
        args.seed + 7001,
        args.input_std,
        sets=1,
    )[0]

    def run_static():
        return reduce_output(call(static), dist, world_size)

    expected, correctness = [], []
    for index, (source, (quantized, bf16, stats)) in enumerate(zip(inputs, references)):
        evidence.stage("reference-check", execution="eager", input_index=index)
        copy_input(static, source)
        actual = run_static().clone()
        # Stable NaN bit patterns can pass bit_equal; sample-based references
        # must never hide malformed or nonfinite values later in the batch.
        validate_output(actual, source.hidden_states.shape)
        assert_rank_agreement(actual, dist, world_size)
        expected.append(actual)
        n = quantized.shape[0]
        q_error = error_metrics(actual[:n], quantized)
        bf_error = error_metrics(actual[:n], bf16)
        checked = bf_error if name == "marlin" else q_error
        if checked["relative_l2"] > args.max_relative_l2:
            raise BenchmarkCheckError(
                f"{name} reference failure",
                {
                    **checked,
                    "reference": "bf16" if name == "marlin" else "w4a4",
                    "relative_l2_limit": args.max_relative_l2,
                },
            )
        correctness.append(
            {
                "w4a4_reference": q_error,
                "bf16_reference": bf_error,
                "reference_tokens": n,
                "clamp_stats_local_rank": stats,
            }
        )

    copy_input(static, inputs[0])
    repeat_identity = True
    repeat_checks = []
    for check_index in range(20):
        evidence.stage(
            "repeat-check", execution="eager", input_index=0, check_index=check_index
        )
        metrics = repeat_metrics(name, run_static(), expected[0])
        repeat_checks.append(metrics)
        repeat_identity = metrics["bit_identity"] and repeat_identity

    # Performance uses four distinct input addresses, not a copy into one
    # repeatedly warmed buffer. Do not describe this as guaranteed cold-cache
    # timing: the GPU's actual cache behavior still depends on geometry.
    def make_run(dispatch):
        def run():
            return reduce_output(call(dispatch), dist, world_size)

        return run

    timed_runs = [make_run(dispatch) for dispatch in inputs]
    rows = []
    for execution in args.execution:
        evidence.stage(
            "warmup", execution=execution, input_index=None, check_index=None
        )
        for run in timed_runs:
            for _ in range(args.warmup):
                run()
        stream.synchronize()
        graphs, graph_outputs = [], []
        replay_identity = None
        graph_checks = []
        if execution == "graph":
            for index, run in enumerate(timed_runs):
                evidence.stage("capture", input_index=index)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = run()
                graphs.append(graph)
                graph_outputs.append(captured)
            # Changing values AND routing at captured addresses is a separate
            # correctness experiment. Restore input 0 before measurement.
            copy_input(static, inputs[0])
            replay_identity = True
            for index, (source, target) in enumerate(zip(inputs[1:], expected[1:]), 1):
                evidence.stage("changed-input-replay", input_index=index)
                copy_input(inputs[0], source)
                graphs[0].replay()
                metrics = repeat_metrics(name, graph_outputs[0], target)
                graph_checks.append(
                    {"phase": "changed-input-replay", "input_index": index, **metrics}
                )
                replay_identity = metrics["bit_identity"] and replay_identity
            copy_input(inputs[0], static)
            for index, (graph, captured, target) in enumerate(
                zip(graphs, graph_outputs, expected)
            ):
                evidence.stage("restored-input-replay", input_index=index)
                graph.replay()
                metrics = repeat_metrics(name, captured, target)
                graph_checks.append(
                    {"phase": "restored-input-replay", "input_index": index, **metrics}
                )
                replay_identity = metrics["bit_identity"] and replay_identity
            executors = [graph.replay for graph in graphs]
        else:
            executors = timed_runs

        evidence.stage("timing", input_index=None)
        samples, host_samples = [], []
        for sample in range(args.samples):
            execute = executors[sample % len(executors)]
            stream.synchronize()
            if world_size > 1:
                dist.barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            host_start = time.perf_counter()
            start.record(stream)
            execute()
            end.record(stream)
            end.synchronize()
            host_samples.append((time.perf_counter() - host_start) * 1000)
            samples.append(start.elapsed_time(end))

        evidence.stage("gather-samples")
        all_samples, all_host = [None] * world_size, [None] * world_size
        if world_size > 1:
            dist.all_gather_object(all_samples, samples)
            dist.all_gather_object(all_host, host_samples)
        else:
            all_samples[0], all_host[0] = samples, host_samples
        rows.append(
            {
                "backend": name,
                "execution": execution,
                "tokens": static.hidden_states.shape[0],
                "tp_size": world_size,
                "gpu": summarize_rank_samples(all_samples),
                "host": summarize_rank_samples(all_host),
                "rank_samples_ms": all_samples,
                "rank_host_samples_ms": all_host,
                "correctness": correctness,
                "repeat_bit_identity": repeat_identity,
                "changed_input_graph_bit_identity": replay_identity,
                "repeat_checks": repeat_checks,
                "repeat_max_relative_l2": max(
                    item["relative_l2"] for item in repeat_checks
                ),
                "graph_checks": graph_checks,
                "graph_max_relative_l2": max(
                    (item["relative_l2"] for item in graph_checks), default=None
                ),
            }
        )
        # Persist each completed execution immediately: a later Graph failure
        # must not discard valid eager samples or mark the whole run successful.
        evidence.stage("save-execution")
        on_result(rows[-1])
        stream.synchronize()
        # No graph-referenced workspace is evicted. First destroy all captures,
        # then allow the outer runtime scope to release its execution resources.
        executors = []
        for graph in graphs:
            graph.reset()
        graph_outputs.clear()
        graphs.clear()
    return rows, expected


def write_latency_csv(path, rows):
    fields = (
        "backend",
        "execution",
        "tokens",
        "tp_size",
        "p50_ms",
        "p95_ms",
        "mean_ms",
        "host_p50_ms",
        "host_p95_ms",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fields[:4]},
                    **{key: row["gpu"][key] for key in fields[4:7]},
                    "host_p50_ms": row["host"]["p50_ms"],
                    "host_p95_ms": row["host"]["p95_ms"],
                }
            )


def persist_latency_case(output, rows, row):
    """Checkpoint completed modes without claiming the entire run passed."""
    rows.append(row)
    write_json(output / "latency.json", {"complete": False, "cases": rows})
    write_latency_csv(output / "latency.csv", rows)


def input_prefix_hashes(inputs, count=64):
    """Fingerprint x, IDs and weights outside timing; bound host-copy cost."""
    import torch

    values = []
    for dispatch in inputs:
        digest = hashlib.sha256()
        for tensor in (
            dispatch.hidden_states,
            dispatch.topk_output.topk_ids,
            dispatch.topk_output.topk_weights,
        ):
            prefix = tensor[:count].contiguous()
            digest.update(str((tuple(prefix.shape), prefix.dtype)).encode())
            digest.update(prefix.view(torch.uint8).cpu().numpy().tobytes())
        values.append(digest.hexdigest())
    return values


def telemetry():
    return command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )


def latency_run(layer, args, helpers, dist, rank, world_size, output, evidence):
    import torch

    backends = {}
    for name in args.backends:
        evidence.stage("prepare-backend", backend=name, execution=None, tokens=None)
        backends[name] = build_backend(
            name, layer, helpers, args.capacity, args.cutile_api
        )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    rows, direct_comparisons = [], []
    with torch.cuda.stream(stream):
        for case_index, tokens in enumerate(args.tokens):
            evidence.stage(
                "prepare-inputs",
                case_index=case_index,
                tokens=tokens,
                backend=None,
                execution=None,
            )
            inputs = make_inputs(layer, tokens, args.seed + tokens, args.input_std)
            hashes = input_prefix_hashes(inputs)
            rank_hashes = [None] * world_size
            if world_size > 1:
                dist.all_gather_object(rank_hashes, hashes)
                if any(value != hashes for value in rank_hashes):
                    raise AssertionError(
                        "TP ranks generated different input/routing prefixes"
                    )
            evidence.stage("compute-reference")
            references = reference_cases(layer, inputs, args, helpers, dist, world_size)
            # Alternate order to reduce systematic warming/thermal bias. Runs
            # across different checkouts must also be alternated externally.
            names = (
                args.backends if case_index % 2 == 0 else list(reversed(args.backends))
            )
            outputs = {}
            for name in names:
                evidence.stage("backend-checks", backend=name, execution="eager")
                before_telemetry = telemetry() if rank == 0 else None

                def on_result(row):
                    row["input_prefix_sha256"] = hashes
                    row["hashed_tokens"] = min(64, tokens)
                    if rank == 0:
                        row["telemetry_before"] = before_telemetry
                        row["telemetry_after"] = telemetry()
                        persist_latency_case(output, rows, row)
                        print(
                            json.dumps(
                                {
                                    "backend": name,
                                    "tokens": tokens,
                                    "execution": row["execution"],
                                    "p50_ms": row["gpu"]["p50_ms"],
                                    "p95_ms": row["gpu"]["p95_ms"],
                                }
                            ),
                            flush=True,
                        )
                    else:
                        rows.append(row)

                _, outputs[name] = measure_backend(
                    name,
                    backends[name],
                    inputs,
                    references,
                    args,
                    dist,
                    world_size,
                    stream,
                    evidence,
                    on_result,
                )
            if "cutile" in outputs and "direct" in outputs:
                evidence.stage(
                    "direct-comparison", backend="cutile/direct", execution="eager"
                )
                comparisons = []
                for integrated, direct in zip(outputs["cutile"], outputs["direct"]):
                    metrics = error_metrics(integrated, direct)
                    identity = bit_equal(integrated, direct)
                    # The fused API can chunk >8192 tokens. Chunking changes
                    # GEMM shapes, so require numerical agreement, not the
                    # bit identity expected from the same unchunked kernel.
                    if args.cutile_api == "fused" and tokens > 8192:
                        if metrics["relative_l2"] > args.max_relative_l2:
                            raise BenchmarkCheckError(
                                "chunked/direct mismatch", metrics
                            )
                    elif not identity:
                        raise BenchmarkCheckError(
                            "adapter/direct same-kernel outputs differ", metrics
                        )
                    comparisons.append({**metrics, "bit_identity": identity})
                direct_comparisons.append(
                    {"tokens": tokens, "comparisons": comparisons}
                )
            del outputs, references, inputs
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if rank == 0:
        write_json(
            output / "latency.json",
            {
                "complete": True,
                "cases": rows,
                "direct_adapter_comparison": direct_comparisons,
            },
        )


def workspace_run(layer, args, helpers, dist, rank, world_size, output, evidence):
    import torch

    if world_size != 1:
        raise ValueError(
            "workspace mode uses one process/device; run it separately on each GPU"
        )
    evidence.stage("prepare-backend", backend="cutile", execution="eager")
    call = build_backend("cutile", layer, helpers, args.capacity, args.cutile_api)
    inputs = make_inputs(layer, args.capacity, args.seed, args.input_std, sets=1)[0]
    streams = [torch.cuda.Stream() for _ in range(args.streams)]
    torch.cuda.synchronize()
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    rows = []
    # Keep context, runner, capacity and weight pack alive for ALL steps. Visit
    # user-specified bucket sequence (including repeats) on each stream without
    # empty_cache, context reset, or releasing earlier executions.
    for stream_index, stream in enumerate(streams):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for step, tokens in enumerate(args.tokens):
                evidence.stage(
                    "workspace-step", stream=stream_index, step=step, tokens=tokens
                )
                dispatch = slice_input(inputs, tokens)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                result = call(dispatch)
                torch.cuda.synchronize()
                del result
                row = {
                    "stream": stream_index,
                    "step": step,
                    "tokens": tokens,
                    "capacity": args.capacity,
                    "allocated_bytes": torch.cuda.memory_allocated(),
                    "reserved_bytes": torch.cuda.memory_reserved(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "step_allocated_delta_bytes": torch.cuda.memory_allocated()
                    - before,
                    "cumulative_allocated_delta_bytes": torch.cuda.memory_allocated()
                    - initial_allocated,
                    "cumulative_reserved_delta_bytes": torch.cuda.memory_reserved()
                    - initial_reserved,
                }
                rows.append(row)
                write_json(
                    output / "workspace.json", {"complete": False, "steps": rows}
                )
                print(json.dumps(row), flush=True)
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with (output / "workspace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "workspace.json",
        {
            "complete": True,
            "steps": rows,
            "initial_allocated_bytes": initial_allocated,
            "initial_reserved_bytes": initial_reserved,
            "note": "Sequential stream visits expose cumulative residency, not concurrent-stream throughput. No live graph workspace is evicted. Fixtures retain checkpoint and prepared weights; total allocation is not a production model weight budget.",
        },
    )


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.mode == "workspace" and int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("workspace mode is a single-process experiment")
    # These imports must remain below parse_args so --help works without torch,
    # FlashInfer, a CUDA driver, or a complete SGLang development environment.
    import torch
    import torch.distributed as dist

    sys.path[0:0] = [str(REPO_ROOT / "python"), str(REPO_ROOT)]
    import sglang
    from sglang.srt.layers.moe.moe_runner import flashinfer_cutile

    # An old editable install must not silently benchmark another checkout
    # while metadata reports this checkout's SHA.
    for module in (sglang, flashinfer_cutile):
        if not Path(module.__file__).resolve().is_relative_to(REPO_ROOT):
            raise RuntimeError(
                f"module loaded outside target checkout: {module.__file__}"
            )
    helpers = load_test_helpers()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in (1, 2, 4):
        raise ValueError("this protocol supports TP1, TP2 and TP4")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution requires CUDA; --help is available on CPU")
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability() not in ((12, 0), (12, 1)):
        raise RuntimeError("requires SM120 or SM121")
    if world_size > 1:
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
            timeout=distributed_timeout(args),
        )
    output = args.output_dir
    directory_error = [None]
    if rank == 0:
        try:
            create_output_dir(output)
        except OSError as error:
            directory_error[0] = str(error)
    if world_size > 1:
        dist.broadcast_object_list(directory_error)
    if directory_error[0]:
        raise FileExistsError(directory_error[0])
    evidence = RunEvidence(output, rank)
    distributed_worker = world_size > 1 and dist.is_initialized()
    try:
        evidence.stage("load-weights", run_mode=args.mode)
        with torch.inference_mode(), runtime_scope(args.capacity):
            torch.manual_seed(args.seed)
            # Full synthetic weights use the same seed on every rank before
            # slicing I. Expert count E stays unchanged under tensor parallelism.
            if args.synthetic:
                full = helpers._synthetic_layer(
                    experts=args.experts,
                    hidden_size=6144,
                    intermediate_size=3072,
                    top_k=4,
                )
                layer = helpers._slice_layer_for_tp(
                    full, tp_rank=rank, tp_size=world_size
                )
                del full
            else:
                layer = helpers._load_checkpoint_layer(
                    args.checkpoint_dir,
                    experts=args.experts,
                    layer_index=args.layer_index,
                    tp_rank=rank,
                    tp_size=world_size,
                )
            evidence.stage("metadata")
            info = metadata(args, torch, rank, world_size)
            info["geometry"] = {
                "experts_per_rank": layer.num_experts,
                "hidden_size": layer.hidden_size,
                "intermediate_per_rank": layer.intermediate_size_per_partition,
                "top_k": layer.moe_runner_config.top_k,
            }
            # Each rank writes a separate file; these carry device/topology
            # evidence even if a later NCCL or kernel operation fails.
            write_json(output / f"metadata-rank{rank}.json", info)
            if args.mode == "latency":
                latency_run(
                    layer, args, helpers, dist, rank, world_size, output, evidence
                )
            else:
                workspace_run(
                    layer, args, helpers, dist, rank, world_size, output, evidence
                )
            evidence.stage("final-synchronize")
            torch.cuda.synchronize()
        if distributed_worker:
            evidence.stage("distributed-shutdown")
            dist.destroy_process_group()
        evidence.succeed()
    except Exception as error:
        # Do not attempt collective shutdown on an unhealthy communicator.
        # A hard worker exit lets torchrun terminate ranks blocked elsewhere;
        # the original exception is retained in both status and the traceback.
        report_failure(evidence, error, distributed_worker=distributed_worker)
        raise


if __name__ == "__main__":
    main()
