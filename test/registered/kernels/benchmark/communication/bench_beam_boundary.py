"""Benchmark Beam's tensor-parallel layer boundary: all-reduce + sandwich norm.

Every provider takes this rank's partial sums `[tokens, 5120]` BF16 and returns
`(next_norm(combined), combined)`: the reduced branch is post-normed, combined
with the depth-scaled residual, and normalized for the next sublayer.

Providers:
    nccl         NCCL all-reduce (the fallback above the custom all-reduce buffer)
    ca-auto      CustomAllReduceV2 with its tuned algorithm choice
    ca-1shot     CustomAllReduceV2 forced to 1-shot push
    ca-2shot     CustomAllReduceV2 forced to 2-shot pull
    ca-auto-pdl  ca-auto with the norm launched with PDL
    fi-auto      FlashInfer MNNVL all-reduce (kAllReduce, its own 1/2-shot choice)
    fi-auto-pdl  fi-auto with the norm launched with PDL
    fused        beam_all_reduce_norm: 1-shot push with the epilogue fused
    ca-2shot-ar  ca-2shot without the norm, the floor for a fused 2-shot epilogue
    fi-ar        fi-auto without the norm
All but `fused` and the `-ar` providers finish with the Triton sandwich norm.

Modes: `graph` replays ~100 calls captured in one CUDA graph, as decode runs;
`eager` times back-to-back launches with the host running ahead, as prefill
runs: CustomAllReduceV2 then takes its eager thresholds and staging copies.
`l2=cold` rotates input copies past L2. `l2=hot` reuses one copy: the branch
arrives straight from the o_proj GEMM, so hot L2 is the serving best case,
kept to bound how much L2 reuse matters.

`ca-auto` skips sizes its tuned thresholds hand to NCCL. Its buffers are not
capped at SGLANG_CUSTOM_ALL_REDUCE_V2_MAX_SIZE_KB, so above that cap it shows
what lifting the cap would give.

Usage, from this directory::

    python bench_beam_boundary.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import math
import os
from collections.abc import Callable

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import get_benchmark_range, multigpu_bench_main
from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.kernels.ops.communication.beam_all_reduce import beam_all_reduce_norm
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.layernorm import BeamSandwichResidualRMSNormOp
from sglang.kernels.ops.layernorm.beam_fused import fused_sandwich_residual_rmsnorm
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=300,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires multi-GPU, self-skips in CI",
)

DTYPE = torch.bfloat16
HIDDEN = 5120
EPS = 1e-6
# Depth-scaled residual weights of decoder layer 10's attention boundary.
_DEPTH = 2 * (10 + 1)
SCALES = (1.0 / math.sqrt(_DEPTH), math.sqrt(_DEPTH - 1) / math.sqrt(_DEPTH))
TOKENS = get_benchmark_range(
    [1, 32, 64, 96, 128, 256, 384, 512, 640, 768, 1024, 4096, 8192, 16384],
    [1, 64, 1024],
)
MAX_BYTES = max(TOKENS) * HIDDEN * DTYPE.itemsize
MODES = [("graph", "cold"), ("graph", "hot"), ("eager", "cold")]
PROVIDERS = [
    "nccl",
    "ca-auto",
    "ca-1shot",
    "ca-2shot",
    "ca-auto-pdl",
    "fi-auto",
    "fi-auto-pdl",
    "fused",
    "ca-2shot-ar",
    "fi-ar",
]
_FORCED_ALGOS = {
    "ca-1shot": AllReduceAlgo.ONE_SHOT_PUSH,
    "ca-2shot": AllReduceAlgo.TWO_SHOT_PULL,
}

# ---------------------------------------------------------------------------
# Per-rank distributed init (run once per torchrun worker)
# ---------------------------------------------------------------------------


@cache_once
def _init_cpu_group() -> dist.ProcessGroup:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    if local_rank == 0:
        props = torch.cuda.get_device_properties(local_rank)
        print(
            f"{props.name}: {props.multi_processor_count} SMs, "
            f"L2 {props.L2_cache_size / 2**20:.0f} MiB"
        )
    return coord.cpu_group


@cache_once
def _init_gpu_group() -> dist.ProcessGroup:
    _init_cpu_group()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    gpu_group = dist.new_group(backend="nccl", device_id=device)
    assert isinstance(gpu_group, dist.ProcessGroup)
    return gpu_group


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


def _new_comm(**sizes) -> CustomAllReduceV2:
    comm = CustomAllReduceV2(_init_cpu_group(), _device(), **sizes)
    if comm.disabled:
        raise RuntimeError("CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


@cache_once
def _init_ca_comm() -> CustomAllReduceV2:
    """Tuned thresholds and block counts, with both planes sized for the sweep."""
    return _new_comm(max_pull_size=MAX_BYTES, max_push_size=MAX_BYTES)


_FUSED_COMMS: dict[int, CustomAllReduceV2] = {}


def _fused_comm(tokens: int) -> CustomAllReduceV2:
    # The fused kernel needs one push counter per token. Like the model, reuse
    # the shared plane when it fits and size a push-only plane otherwise.
    shared = _init_ca_comm()
    if shared.supports_push(nbytes=tokens * HIDDEN * DTYPE.itemsize, num_blocks=tokens):
        return shared
    if tokens not in _FUSED_COMMS:
        _FUSED_COMMS[tokens] = _new_comm(
            max_pull_size=0,
            max_pull_blocks=0,
            max_push_size=tokens * HIDDEN * DTYPE.itemsize,
            max_push_blocks=tokens,
        )
    return _FUSED_COMMS[tokens]


@cache_once
def _init_fi_workspace():
    from sglang.srt.layers.flashinfer_comm_fusion import FlashInferWorkspaceManager

    cpu_group = _init_cpu_group()
    manager = FlashInferWorkspaceManager()
    manager.initialize(
        world_size=dist.get_world_size(cpu_group),
        rank=dist.get_rank(cpu_group),
        max_token_num=max(TOKENS),
        hidden_dim=HIDDEN,
        backend="mnnvl",
        dtype=DTYPE,
        device_group=_init_gpu_group(),
        cpu_group=cpu_group,
    )
    return manager.workspace  # None when this node cannot allocate it


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _fi_all_reduce(x: torch.Tensor) -> torch.Tensor:
    # The model's own call, so the benchmark measures what serving runs.
    from sglang.srt.models.beam import _flashinfer_all_reduce

    return _flashinfer_all_reduce(x, _init_fi_workspace())


def _make_all_reduce(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "nccl":
        group = _init_gpu_group()

        def nccl(x: torch.Tensor) -> torch.Tensor:
            dist.all_reduce(x, group=group)
            return x

        return nccl
    if name.startswith("fi-"):
        return _fi_all_reduce
    comm = _init_ca_comm()
    algo = _FORCED_ALGOS.get(name.removesuffix("-ar"))
    if algo is None:
        nccl = _make_all_reduce("nccl")

        def tuned(x: torch.Tensor) -> torch.Tensor:
            # Serving dispatch: sizes the tuned thresholds reject go to NCCL.
            return comm.custom_all_reduce(x) if comm.should_custom_ar(x) else nccl(x)

        return tuned

    def forced(x: torch.Tensor) -> torch.Tensor:
        comm.override_algo = algo
        try:
            return comm.custom_all_reduce(x)
        finally:
            comm.override_algo = None

    return forced


def _make_boundary(provider: str, tokens: int, post: torch.Tensor, nxt: torch.Tensor):
    if provider == "fused":
        comm = _fused_comm(tokens)

        def fused(branch, residual):
            return beam_all_reduce_norm(
                comm.obj, branch, residual, post, nxt, EPS, SCALES
            )

        return fused
    all_reduce = _make_all_reduce(provider.removesuffix("-pdl"))
    if provider.endswith("-ar"):
        return lambda branch, residual: all_reduce(branch)
    enable_pdl = provider.endswith("-pdl")

    def unfused(branch, residual):
        return fused_sandwich_residual_rmsnorm(
            all_reduce(branch),
            residual,
            post,
            nxt,
            EPS,
            *SCALES,
            enable_pdl=enable_pdl,
        )

    return unfused


def _available(provider: str) -> bool:
    return not provider.startswith("fi-") or _init_fi_workspace() is not None


def _graph_context(provider: str):
    # Pull algorithms read registered graph inputs; push-only `fused` needs none.
    if provider.startswith("ca-"):
        return _init_ca_comm().capture()
    return contextlib.nullcontext()


def _inputs(tokens: int, seed: int):
    gen = torch.Generator(device=_device()).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=_device(), dtype=DTYPE)

    branch = randn(tokens, HIDDEN)
    residual = randn(tokens, HIDDEN)
    post = 1 + 0.1 * randn(HIDDEN)
    nxt = 1 + 0.1 * randn(HIDDEN)
    return branch, residual, post, nxt


# ---------------------------------------------------------------------------
# Numerics check, run once before timing
# ---------------------------------------------------------------------------


def _reference(branch, residual, post, nxt):
    peers = [torch.empty_like(branch) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, branch, group=_init_gpu_group())
    # FP32 rank sum in rank order, as the custom all-reduce kernels accumulate.
    reduced = torch.zeros_like(branch, dtype=torch.float32)
    for peer in peers:
        reduced += peer.float()
    return BeamSandwichResidualRMSNormOp().forward_native(
        reduced.to(DTYPE), residual, post, nxt, EPS, SCALES
    )


# Same bounds as test_beam_all_reduce.py: about one BF16 step of the residual
# operands absolute, since cancellation in the combine defeats a relative bound.
_TOLERANCES = {"normed": (0.032, 0.016), "combined": (0.016, 0.016)}


def check_numerics() -> None:
    """Every full provider must match the BF16-rounded reference."""
    rank = dist.get_rank(_init_cpu_group())
    max_errors: dict[str, float] = {}
    for tokens in (1, 64, 1024):
        branch, residual, post, nxt = _inputs(tokens, seed=1000 * rank + tokens)
        expected = _reference(branch, residual, post, nxt)
        for provider in PROVIDERS:
            if provider.endswith("-ar") or not _available(provider):
                continue
            boundary = _make_boundary(provider, tokens, post, nxt)
            actual = boundary(branch.clone(), residual)
            for name, a, e in zip(("normed", "combined"), actual, expected):
                atol, rtol = _TOLERANCES[name]
                torch.testing.assert_close(
                    a, e, atol=atol, rtol=rtol, msg=f"{provider} {name} {tokens=}"
                )
                error = (a.float() - e.float()).abs().max().item()
                max_errors[provider] = max(max_errors.get(provider, 0.0), error)
    # A token's fused output must not depend on how many tokens share its batch.
    branch, residual, post, nxt = _inputs(64, seed=rank)
    fused = _make_boundary("fused", 64, post, nxt)
    full = fused(branch, residual)
    half = fused(branch[:32].contiguous(), residual[:32].contiguous())
    for h, f in zip(half, full):
        assert torch.equal(h, f[:32]), "fused rows differ between 32 and 64 tokens"
    # Swapping CustomAR for FlashInfer cannot change model outputs if the sums match.
    if _available("fi-auto"):
        for tokens in (96, 256, 640):
            branch = _inputs(tokens, seed=2000 * rank + tokens)[0]
            assert _init_ca_comm().should_custom_ar(branch), f"NCCL at {tokens=}"
            custom = _make_all_reduce("ca-auto")(branch.clone())
            flashinfer = _make_all_reduce("fi-auto")(branch.clone())
            assert torch.equal(custom, flashinfer), f"FlashInfer != CustomAR {tokens=}"
    # Serving replays the boundaries from CUDA graphs; replay must match eager.
    graph_cases = (("fused", 32), ("ca-auto-pdl", 128), ("fi-auto-pdl", 256))
    for provider, tokens in graph_cases:
        if not _available(provider):
            continue
        branch, residual, post, nxt = _inputs(tokens, seed=3000 * rank + tokens)
        boundary = _make_boundary(provider, tokens, post, nxt)
        eager = boundary(branch.clone(), residual)
        static = branch.clone()
        boundary(static, residual)
        graph = torch.cuda.CUDAGraph()
        with _graph_context(provider), torch.cuda.graph(graph):
            replayed = boundary(static, residual)
        static.copy_(branch)
        graph.replay()
        for r, e in zip(replayed, eager):
            assert torch.equal(r, e), f"{provider} graph replay != eager {tokens=}"
    torch.cuda.synchronize()
    if rank == 0:
        errors = ", ".join(f"{p} {e:.4f}" for p, e in max_errors.items())
        print(f"numerics ok; max abs error vs reference: {errors}")
        print("batch invariance, FlashInfer == CustomAR, graph replay == eager: ok")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

_EAGER_CALLS = 20
_EAGER_REPEATS = 5


def _bench_eager(fn, branch, residual, *, sync, memory_footprint):
    """Median GPU time of back-to-back eager calls, as a prefill forward issues them.

    The host queues all calls before the GPU drains them, so launch overhead
    is hidden as in serving and the time is the kernels plus any staging
    copies. Inputs rotate past 2x L2 when they are small enough to fit in it.
    """
    l2_bytes = torch.cuda.get_device_properties(_device()).L2_cache_size
    copies = min(_EAGER_CALLS, max(1, -(-2 * l2_bytes // (2 * branch.nbytes))))
    inputs = [(branch.clone(), residual.clone()) for _ in range(copies)]
    for args in inputs:
        fn(*args)
    times = []
    for _ in range(_EAGER_REPEATS):
        torch.cuda.synchronize()
        sync()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(_EAGER_CALLS):
            fn(*inputs[i % copies])
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / _EAGER_CALLS / 1e3)
    median = sorted(times)[len(times) // 2]
    return marker.BenchResult((0.5,), [median], memory_footprint)


@marker.parametrize("mode,l2", MODES, ci_vals=MODES[:1])
@marker.parametrize("tokens", TOKENS)
@marker.benchmark("provider", PROVIDERS)
def benchmark(tokens: int, mode: str, l2: str, provider: str):
    cpu_group = _init_cpu_group()
    gpu_group = _init_gpu_group()
    world_size = dist.get_world_size(cpu_group)
    nbytes = tokens * HIDDEN * DTYPE.itemsize
    if provider.startswith("ca-auto") and (
        _init_ca_comm()._pick_config(nbytes, mode == "graph") is None
    ):
        marker.skip("tuned thresholds hand this size to NCCL, same as `nccl`")
    if provider.endswith("-pdl") and not is_arch_support_pdl():
        marker.skip("PDL needs SM90+")
    if not _available(provider):
        marker.skip("FlashInfer MNNVL workspace unavailable")
    branch, residual, post, nxt = _inputs(tokens, seed=dist.get_rank(cpu_group))
    boundary = _make_boundary(provider, tokens, post, nxt)

    def barrier():
        dist.barrier(gpu_group)

    # Ring-equivalent bytes per rank, so GB/s reads as all-reduce bus bandwidth.
    bus_bytes = int(nbytes * 2 * (world_size - 1) / world_size)
    # In-place collectives grow `branch` across iterations; timing is value-independent.
    if mode == "eager":
        result = _bench_eager(
            boundary, branch, residual, sync=barrier, memory_footprint=bus_bytes
        )
    else:
        result = marker.do_bench(
            boundary,
            input_args=(branch, residual),
            graph_clone_args="all" if l2 == "cold" else None,
            graph_context_fn=(lambda: _graph_context(provider)),
            sync_multigpu_fn=barrier,
            memory_args=None,
            memory_output=None,
            extra_memory_footprint=bus_bytes,
        )
    if dist.get_rank(cpu_group) == 0:
        # The table prints only at the end; keep finished cells if a later one fails.
        print(
            f"cell {provider} {tokens=} {mode} {l2}: {result.times[0] * 1e6:.2f} us",
            flush=True,
        )
    return result


def main() -> None:
    logging.disable(logging.INFO)
    check_numerics()
    benchmark.run()


if __name__ == "__main__":
    multigpu_bench_main(
        name=__name__,
        file=__file__,
        num_gpus=[4],
        main_fn=main,
    )
