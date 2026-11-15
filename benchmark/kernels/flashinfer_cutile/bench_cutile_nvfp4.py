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
        "p90_ms": percentile(samples, 90),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
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
        "--input-protocol",
        choices=("historical", "peer-compatible"),
        default="historical",
    )
    parser.add_argument(
        "--reference-protocol",
        choices=("corrected", "both"),
        default="corrected",
        help="Both adds a peer-math diagnostic; the corrected gate is unchanged",
    )
    parser.add_argument("--marlin-api", choices=("runner", "direct"), default="runner")
    parser.add_argument(
        "--compare-tp1",
        action="store_true",
        help="Compare real NCCL-reduced outputs with a full-weight TP1 call",
    )
    parser.add_argument("--workspace-warm-repeats", type=int, default=0)
    parser.add_argument("--require-all-buckets", action="store_true")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("cutile", "marlin", "direct", "direct-chunked"),
        default=["cutile", "marlin"],
    )
    parser.add_argument(
        "--execution", nargs="+", choices=("eager", "graph"), default=["eager", "graph"]
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reference-tokens", type=int, default=64)
    parser.add_argument(
        "--reference-sampling",
        choices=("prefix", "chunk-boundaries"),
        default="prefix",
        help="Keep historical prefix protocol or sample every chunk boundary and tail",
    )
    parser.add_argument(
        "--workspace-backend",
        choices=("cutile", "direct"),
        default="cutile",
        help="Direct is an unchunked public-API control and requires one stream",
    )
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
    if (
        args.mode == "workspace"
        and args.workspace_backend == "direct"
        and args.streams != 1
    ):
        parser.error(
            "direct workspace control requires --streams 1; it has one execution"
        )
    if args.workspace_warm_repeats < 0:
        parser.error("--workspace-warm-repeats must be nonnegative")
    if args.compare_tp1 and (args.synthetic or args.mode != "latency"):
        parser.error("--compare-tp1 requires real checkpoint latency mode")
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
            "reference_sampling": args.reference_sampling,
            "chunked_oracle": "direct-chunked: independent public API with 8192-token calls",
            "workspace_backend": args.workspace_backend,
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


def build_backend(
    name, source, helpers, capacity, cutile_api="runner-core", marlin_api="runner"
):
    """Use independent weights because both preparation paths can replace them."""
    import torch

    layer = clone_layer(source, helpers)
    config = layer.moe_runner_config
    runner = None
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

    elif name in ("direct", "direct-chunked"):
        chunk_size = 8192 if name == "direct-chunked" else None
        runner, prepared = helpers._direct_flashinfer(
            layer, capacity=min(capacity, 8192) if chunk_size else capacity
        )

        def call(dispatch):
            return helpers._direct_dispatch(
                runner, prepared, dispatch, chunk_size=chunk_size
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

        if marlin_api == "runner":

            def call(dispatch):
                return fused_experts_none_to_marlin(
                    dispatch, prepared, config
                ).hidden_states
        elif marlin_api == "direct":
            from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
                fused_marlin_moe,
            )
            from sglang.srt.layers.quantization.marlin_utils import (
                marlin_make_workspace,
            )

            def call(dispatch):
                topk = dispatch.topk_output
                return fused_marlin_moe(
                    dispatch.hidden_states,
                    prepared.w13_qweight,
                    prepared.w2_qweight,
                    prepared.w13_scales,
                    prepared.w2_scales,
                    topk.router_logits,
                    topk.topk_weights,
                    topk.topk_ids,
                    global_num_experts=config.num_experts,
                    w1_global_scale=prepared.w13_global_scale,
                    w2_global_scale=prepared.w2_global_scale,
                    workspace=marlin_make_workspace(
                        dispatch.hidden_states.device, max_blocks_per_sm=4
                    ),
                    num_bits=4,
                    clamp_limit=helpers.LIMIT,
                    gemm1_alpha=helpers.ALPHA,
                    activation="silu",
                    is_gated=True,
                )
        else:
            raise ValueError("unknown Marlin API")

    def peer_reference(dispatch, indices):
        selected = helpers._select_dispatch_rows(dispatch, indices)
        return helpers._reference(
            source,
            selected.hidden_states,
            selected.topk_output.topk_ids,
            selected.topk_output.topk_weights,
            reference_mode="peer-diagnostic",
        )

    call.peer_reference = peer_reference
    call.source_layer = source
    call.prepared = prepared
    call.runner = runner if name in ("cutile", "direct", "direct-chunked") else None

    # Retain source geometry/weights until all captures using this callable die.
    call.layer = layer
    # Record the implementation actually imported, not the experiment label.
    # A pre-chunking checkout can use this same harness for a matched control.
    if name == "cutile":
        call.chunk_size = (
            getattr(flashinfer_cutile, "_MAX_TOKENS_PER_CHUNK", None)
            if cutile_api == "runner-core"
            else 8192
        )
    else:
        call.chunk_size = 8192 if name == "direct-chunked" else None
    return call


def make_inputs(
    layer,
    tokens,
    seed,
    input_std,
    *,
    sets=4,
    input_protocol="historical",
    device="cuda",
):
    import torch

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    if input_protocol not in ("historical", "peer-compatible"):
        raise ValueError("unknown input protocol")
    if input_protocol == "peer-compatible":
        # Preserve the public script's draw order and seed per input set.
        # Dense scores are derived AFTER the draws, not sampled independently.
        cfg = layer.moe_runner_config
        values = []
        for copy_index in range(sets):
            generator = torch.Generator().manual_seed(tokens * 1009 + copy_index)
            x = (
                torch.randn(tokens, cfg.hidden_size, generator=generator) * input_std
            ).bfloat16()
            ids = torch.stack(
                [
                    torch.randperm(cfg.num_experts, generator=generator)[: cfg.top_k]
                    for _ in range(tokens)
                ]
            ).int()
            weights = torch.rand(tokens, cfg.top_k, generator=generator)
            scores = torch.zeros(tokens, cfg.num_experts).scatter_(
                1, ids.long(), weights
            )
            values.append(
                StandardDispatchOutput(
                    hidden_states=x.to(device),
                    hidden_states_scale=None,
                    topk_output=StandardTopKOutput(
                        topk_ids=ids.to(device),
                        topk_weights=weights.to(device),
                        router_logits=scores.to(device),
                    ),
                )
            )
        return values
    generator = torch.Generator(device=device).manual_seed(seed)
    config = layer.moe_runner_config
    values = []
    for _ in range(sets):
        x = (
            torch.randn(
                tokens,
                config.hidden_size,
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            * input_std
        )
        # Sorting random scores gives distinct expert IDs per token. Using
        # randint would accidentally test duplicate routes instead of top-k.
        scores = torch.rand(
            tokens, config.num_experts, device=device, generator=generator
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
        tokens = dispatch.hidden_states.shape[0]
        if args.reference_sampling == "chunk-boundaries":
            indices = helpers._reference_token_indices(tokens, args.reference_tokens)
            small = helpers._select_dispatch_rows(dispatch, indices)
        else:
            indices = list(range(min(args.reference_tokens, tokens)))
            small = slice_input(dispatch, len(indices))
        x, ids, weights = (
            small.hidden_states,
            small.topk_output.topk_ids,
            small.topk_output.topk_weights,
        )
        quantized, stats = helpers._reference(layer, x, ids, weights, return_stats=True)
        bf16 = helpers._reference(layer, x, ids, weights, quantize_activations=False)
        reduce_output(quantized, dist, world_size)
        reduce_output(bf16, dist, world_size)
        references.append((quantized, bf16, {**stats, "reference_indices": indices}))
    return references


def select_reference_output(actual, reference, stats):
    """Apply the same row selection to the output and its FP32 references."""
    indices = stats.get("reference_indices")
    return actual[: reference.shape[0]] if indices is None else actual[indices]


def compare_direct_outputs(integrated, direct, *, same_partition, relative_l2_limit):
    """Do not confuse cross-shape accuracy with same-kernel bit identity."""
    metrics = {
        **error_metrics(integrated, direct),
        "bit_identity": bit_equal(integrated, direct),
        "same_partition": same_partition,
    }
    if same_partition:
        if not metrics["bit_identity"]:
            raise BenchmarkCheckError(
                "adapter/direct same-partition outputs differ", metrics
            )
    elif metrics["relative_l2"] > relative_l2_limit:
        raise BenchmarkCheckError(
            "chunked/unchunked mismatch",
            {**metrics, "relative_l2_limit": relative_l2_limit},
        )
    return metrics


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
        if index == 0:
            stream.synchronize()
            first_resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            first_start = time.perf_counter()
        actual = run_static().clone()
        if index == 0:
            stream.synchronize()
            first_call_seconds = time.perf_counter() - first_start
            first_peak_bytes = torch.cuda.max_memory_allocated() - first_resident
        # Stable NaN bit patterns can pass bit_equal; sample-based references
        # must never hide malformed or nonfinite values later in the batch.
        validate_output(actual, source.hidden_states.shape)
        assert_rank_agreement(actual, dist, world_size)
        expected.append(actual)
        n = quantized.shape[0]
        sampled = select_reference_output(actual, quantized, stats)
        q_error = error_metrics(sampled, quantized)
        bf_error = error_metrics(sampled, bf16)
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
        peer_error = None
        if getattr(args, "reference_protocol", "corrected") == "both":
            peer_reference = call.peer_reference(
                source, stats.get("reference_indices", list(range(n)))
            )
            reduce_output(peer_reference, dist, world_size)
            peer_error = error_metrics(sampled, peer_reference)
        correctness.append(
            {
                "peer_math_diagnostic": peer_error,
                "w4a4_reference": q_error,
                "bf16_reference": bf_error,
                "reference_tokens": n,
                "reference_indices": stats.get("reference_indices", list(range(n))),
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
                "first_call_seconds": first_call_seconds,
                "first_call_peak_above_resident_bytes": first_peak_bytes,
                "first_call_resident_bytes": first_resident,
                "first_call_scope": "backend call plus output clone; references already computed",
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


def input_full_hashes(inputs):
    """Hash all input rows, including routes/logits beyond the sampled reference."""
    import torch

    hashes = []
    for dispatch in inputs:
        values = {}
        for name, tensor in (
            ("hidden_states", dispatch.hidden_states),
            ("topk_ids", dispatch.topk_output.topk_ids),
            ("topk_weights", dispatch.topk_output.topk_weights),
            ("router_logits", dispatch.topk_output.router_logits),
        ):
            if tensor is not None:
                raw = tensor.detach().contiguous().view(dtype=torch.uint8)
                values[name] = hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()
        hashes.append(values)
    return hashes


def tensor_sha256(tensor):
    """Fingerprint the complete output, outside measured/captured regions."""
    import torch

    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def execution_workspace_inventory(root):
    """Read actual FlashInfer caches; do not infer residency from input sizes.

    The public MoELayer owns backend runners below it. Keep this diagnostic
    traversal bounded and never follow tensor payloads or Python modules.
    """
    import torch

    seen, caches = set(), []

    def visit(value, path, depth):
        if id(value) in seen or depth > 8 or isinstance(value, torch.Tensor):
            return
        seen.add(id(value))
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, (tuple, list)):
            items = enumerate(value)
        elif hasattr(value, "__dict__"):
            items = vars(value).items()
        else:
            return
        for key, child in items:
            child_path = f"{path}.{key}"
            if key in ("_workspaces", "_workspace_cache") and isinstance(child, dict):
                tensors = []

                def gather(obj):
                    if isinstance(obj, torch.Tensor):
                        tensors.append(obj)
                    elif isinstance(obj, dict):
                        for item in obj.values():
                            gather(item)
                    elif isinstance(obj, (list, tuple)):
                        for item in obj:
                            gather(item)

                gather(child)
                storages = {
                    (
                        str(t.device),
                        t.untyped_storage().data_ptr(),
                    ): t.untyped_storage().nbytes()
                    for t in tensors
                }
                caches.append(
                    {
                        "path": child_path,
                        "keys": [repr(k) for k in child],
                        "unique_storage_bytes": sum(storages.values()),
                    }
                )
            if isinstance(child, (dict, list, tuple)) or (
                type(child).__module__.startswith("flashinfer")
                and hasattr(child, "__dict__")
            ):
                visit(child, child_path, depth + 1)

    visit(root, "execution", 0)
    return caches


def actual_tp1_comparison(
    args, helpers, dispatches, reduced_outputs, rank, dist, world_size
):
    """Compare actual distributed BF16 NCCL results with a full-weight TP1 call.

    Reference error is not subtracted to approximate this quantity. All ranks
    finish the distributed timing first; only rank zero runs the full layer.
    """

    if world_size == 1:
        raise ValueError("--compare-tp1 requires torchrun with TP2 or TP4")
    dist.barrier()
    records = []
    if rank == 0:
        full = helpers._load_checkpoint_layer(
            args.checkpoint_dir, experts=args.experts, layer_index=args.layer_index
        )
        single = build_backend("cutile", full, helpers, args.capacity, args.cutile_api)
        for index, (dispatch, distributed) in enumerate(
            zip(dispatches, reduced_outputs)
        ):
            expected = single(dispatch).clone()
            validate_output(expected, distributed.shape)
            metrics = error_metrics(distributed, expected)
            if metrics["relative_l2"] >= 0.02:
                raise BenchmarkCheckError(
                    "Actual NCCL TP output differs from TP1 beyond the existing TP bound",
                    {**metrics, "relative_l2_limit": 0.02},
                )
            indices = helpers._reference_token_indices(
                distributed.shape[0], min(args.reference_tokens, distributed.shape[0])
            )
            selected = helpers._select_dispatch_rows(dispatch, indices)
            reference = helpers._reference(
                full,
                selected.hidden_states,
                selected.topk_output.topk_ids,
                selected.topk_output.topk_weights,
            )
            records.append(
                dict(
                    input_index=index,
                    **metrics,
                    distributed_nccl=True,
                    tp_size=world_size,
                    reduction_dtype="bfloat16",
                    relative_l2_limit=0.02,
                    reference_indices=indices,
                    tp1_to_reference=error_metrics(expected[indices], reference),
                    distributed_to_reference=error_metrics(
                        distributed[indices], reference
                    ),
                    tp1_output_sha256=tensor_sha256(expected),
                    distributed_output_sha256=tensor_sha256(distributed),
                )
            )
        del single, full
    dist.barrier()
    return records


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
        torch.cuda.synchronize()
        preparation_start = time.perf_counter()
        backends[name] = build_backend(
            name,
            layer,
            helpers,
            args.capacity,
            args.cutile_api,
            marlin_api=getattr(args, "marlin_api", "runner"),
        )
        torch.cuda.synchronize()
        backends[name].preparation_seconds = time.perf_counter() - preparation_start
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    rows, direct_comparisons, tp1_comparisons = [], [], []
    with torch.cuda.stream(stream):
        for case_index, tokens in enumerate(args.tokens):
            evidence.stage(
                "prepare-inputs",
                case_index=case_index,
                tokens=tokens,
                backend=None,
                execution=None,
            )
            inputs = make_inputs(
                layer,
                tokens,
                args.seed + tokens,
                args.input_std,
                input_protocol=getattr(args, "input_protocol", "historical"),
            )
            hashes = input_prefix_hashes(inputs)
            complete_hashes = input_full_hashes(inputs)
            rank_hashes = [None] * world_size
            if world_size > 1:
                all_hashes = {"prefix": hashes, "full": complete_hashes}
                dist.all_gather_object(rank_hashes, all_hashes)
                if any(value != all_hashes for value in rank_hashes):
                    raise AssertionError(
                        "TP ranks generated different input/routing tensors"
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
                    row["preparation_seconds"] = backends[name].preparation_seconds
                    row["input_prefix_sha256"] = hashes
                    row["hashed_tokens"] = min(64, tokens)
                    row["input_full_sha256"] = complete_hashes
                    row["input_protocol"] = getattr(
                        args, "input_protocol", "historical"
                    )
                    row["marlin_api"] = getattr(args, "marlin_api", "runner")
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
            for oracle in ("direct-chunked", "direct"):
                if "cutile" not in outputs or oracle not in outputs:
                    continue
                evidence.stage(
                    "direct-comparison", backend=f"cutile/{oracle}", execution="eager"
                )
                # Both adapter APIs split >8192. Only direct-chunked has the
                # same per-call shape above that boundary. Retain the strict
                # bit gate there; report the unchunked numerical delta separately.
                adapter_chunk = backends["cutile"].chunk_size or tokens
                oracle_chunk = backends[oracle].chunk_size or tokens
                same_partition = min(tokens, adapter_chunk) == min(tokens, oracle_chunk)
                comparisons = [
                    compare_direct_outputs(
                        integrated,
                        direct,
                        same_partition=same_partition,
                        relative_l2_limit=args.max_relative_l2,
                    )
                    for integrated, direct in zip(outputs["cutile"], outputs[oracle])
                ]
                direct_comparisons.append(
                    {"tokens": tokens, "oracle": oracle, "comparisons": comparisons}
                )
            if getattr(args, "compare_tp1", False):
                if "cutile" not in outputs:
                    raise ValueError("--compare-tp1 requires the cutile backend")
                compared = actual_tp1_comparison(
                    args, helpers, inputs, outputs["cutile"], rank, dist, world_size
                )
                tp1_comparisons.append(dict(tokens=tokens, comparisons=compared))
                if rank == 0:
                    write_json(
                        output / "actual-tp1-comparison.json",
                        {"complete": False, "cases": tp1_comparisons},
                    )
            if "cutile" in backends and backends["cutile"].runner is not None:
                execution = backends["cutile"].runner._get_execution(
                    layer.w13_weight_packed.device, tokens
                )
                write_json(
                    output / f"workspace-owner-rank{rank}-t{tokens}.json",
                    {
                        "rank": rank,
                        "tp_size": world_size,
                        "tokens": tokens,
                        "chunk_size": backends["cutile"].chunk_size,
                        "execution_owner_id": id(execution),
                        "flashinfer_workspace_caches": execution_workspace_inventory(
                            execution
                        ),
                        "flashinfer_ceiling": min(
                            args.capacity, backends["cutile"].chunk_size
                        ),
                        "warmed_token_counts": sorted(execution.warmed_token_counts),
                        "allocated_bytes": torch.cuda.memory_allocated(),
                        "reserved_bytes": torch.cuda.memory_reserved(),
                    },
                )
            del outputs, references, inputs
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    if rank == 0:
        if getattr(args, "compare_tp1", False):
            write_json(
                output / "actual-tp1-comparison.json",
                {"complete": True, "cases": tp1_comparisons},
            )
        write_json(
            output / "latency.json",
            {
                "complete": True,
                "cases": rows,
                "direct_adapter_comparison": direct_comparisons,
                "actual_tp1_comparison": tp1_comparisons,
            },
        )


def workspace_bucket_coverage(
    tokens, ceiling, chunk_size, generate_buckets, map_bucket
):
    """Account for every chunk, including an otherwise easy-to-miss tail."""
    expected = sorted(generate_buckets(ceiling))
    visited = set()
    per_call = {}
    for count in tokens:
        parts = (
            [min(chunk_size, count - start) for start in range(0, count, chunk_size)]
            if chunk_size
            else [count]
        )
        per_call[count] = [map_bucket(part, ceiling) for part in parts]
        visited.update(per_call[count])
    return expected, sorted(visited), per_call


def workspace_run(layer, args, helpers, dist, rank, world_size, output, evidence):
    import torch

    if world_size != 1:
        raise ValueError(
            "workspace mode uses one process/device; run it separately on each GPU"
        )
    evidence.stage("prepare-backend", backend=args.workspace_backend, execution="eager")
    call = build_backend(
        args.workspace_backend, layer, helpers, args.capacity, args.cutile_api
    )
    from flashinfer.fused_moe.utils import (
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket,
    )

    ceiling = min(args.capacity, call.chunk_size or args.capacity)
    expected_buckets, visited_buckets, per_call_buckets = workspace_bucket_coverage(
        args.tokens,
        ceiling,
        call.chunk_size,
        get_hybrid_num_tokens_buckets,
        map_to_hybrid_bucket,
    )
    if (
        getattr(args, "require_all_buckets", False)
        and visited_buckets != expected_buckets
    ):
        raise ValueError(
            f"Workspace sequence does not cover all buckets: "
            f"visited={visited_buckets}, expected={expected_buckets}"
        )
    inputs = make_inputs(
        layer,
        args.capacity,
        args.seed,
        args.input_std,
        sets=1,
        input_protocol=getattr(args, "input_protocol", "historical"),
    )[0]
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
                output_live_allocated = torch.cuda.memory_allocated()
                output_bytes = result.numel() * result.element_size()
                del result
                row = {
                    "mapped_buckets": per_call_buckets[tokens],
                    "flashinfer_ceiling": ceiling,
                    "output_live_allocated_bytes": output_live_allocated,
                    "output_bytes": output_bytes,
                    "stream": stream_index,
                    "step": step,
                    "tokens": tokens,
                    "capacity": args.capacity,
                    "backend": args.workspace_backend,
                    "chunk_size": call.chunk_size,
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
                warm = []
                held = torch.cuda.memory_allocated()
                for repeat in range(getattr(args, "workspace_warm_repeats", 0)):
                    torch.cuda.reset_peak_memory_stats()
                    warmed_result = call(dispatch)
                    torch.cuda.synchronize()
                    warm_peak = torch.cuda.max_memory_allocated()
                    del warmed_result
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_allocated()
                    if after != held:
                        raise BenchmarkCheckError(
                            "Warm workspace allocated bytes changed",
                            {"before": held, "after": after, "repeat": repeat},
                        )
                    warm.append(
                        dict(
                            repeat=repeat,
                            allocated_bytes=after,
                            peak_allocated_bytes=warm_peak,
                        )
                    )
                row["warm_repeats"] = warm
                execution = (
                    call.runner._get_execution(layer.w13_weight_packed.device, tokens)
                    if args.workspace_backend == "cutile"
                    else call.runner
                )
                row["execution_owner_id"] = id(execution)
                row["flashinfer_workspace_caches"] = execution_workspace_inventory(
                    execution
                )
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
            "expected_buckets": expected_buckets,
            "visited_buckets": visited_buckets,
            "all_buckets_visited": visited_buckets == expected_buckets,
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
