"""Muse Spark push all-reduce fused with the attention norm-residual boundary, multi-GPU.

The fused kernel must match the push all-reduce it replaces followed by
``fused_norm_residual`` bit for bit, on every rank: both sum the same bf16
partials in fp32 in rank order, round once, and run the same norm. Every test
runs with unicast and, where supported, multicast pushes.

Usage::

    python test_muse_spark_collective_norm_residual_multigpu.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import itertools
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import muse_spark_all_reduce_norm_residual as arn
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.kernels.ops.layernorm.muse_spark_norm_residual_jit import (
    _cluster_launch_config,
    fused_norm_residual,
)
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
    get_all_reduce_config,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN = 8192
PUSH_BYTES = 16 << 20
POST_EPS, PRE_EPS = 1e-6, 1e-5
# Covers every cluster width, both exchanges on each side of the crossover at worlds 8 / 4 / 2
# (M = 16 / 32 / 64), the SM107 TP4 launch override (M = 48), and M far above the push block
# count (rows run persistent). With PUSH_BYTES every M here fits at world >= 2, so an
# unsupported M is a regression, not a skip.
NUM_TOKENS = [1, 3, 8, 16, 17, 32, 33, 48, 64, 65, 128, 383, 1024]
REPEATS = 3
# (fused tokens, standalone elements): the cluster width, the exchange, and the standalone
# algorithm change every step, so each launch reuses slot words another layout wrote.
INTERLEAVED = [
    (8, 64 * HIDDEN + 64),
    (65, 3 * HIDDEN),
    (16, PUSH_BYTES // 4),
    (383, 8 * HIDDEN),
    (32, 129 * HIDDEN),
]


@cache_once
def _init_communicator() -> CustomAllReduceV2:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    comm = CustomAllReduceV2(
        coord.cpu_group, torch.device(f"cuda:{local_rank}"), max_push_size=PUSH_BYTES
    )
    if comm.disabled:
        raise RuntimeError("CustomAllReduceV2 is disabled on this system")
    register_comm_cleanup(comm)
    return comm


# Serving enables multicast only at eight GPUs; forcing it here covers it on smaller CI runners.
@pytest.fixture(params=[False, True], ids=["unicast", "multicast"])
def comm(request, monkeypatch) -> CustomAllReduceV2:
    comm = _init_communicator()
    if request.param and not comm.has_multicast:
        pytest.skip("no multicast push workspace")
    monkeypatch.setattr(arn, "_uses_multicast", lambda _: request.param)
    return comm


def _boundary_kwargs(variant: str) -> dict:
    """The two boundaries Muse Spark runs, plus one without a gate or weights."""
    gen = torch.Generator(device="cuda").manual_seed(3)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, device="cuda", generator=gen)

    pre_weight = (0.9 + 0.2 * rand(HIDDEN)).to(torch.bfloat16)
    if variant == "gate":
        return dict(pre_weight=pre_weight, alpha=rand(1), beta=rand(1))
    if variant == "input_scale":
        return dict(
            pre_weight=pre_weight,
            post_norm_is_input_scale=True,
            post_weight=(0.9 + 0.2 * rand(HIDDEN)).to(torch.bfloat16),
            alpha=rand(HIDDEN),
            beta=rand(HIDDEN),
        )
    assert variant == "plain"
    return {}


def _make_inputs(num_tokens: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's bf16 partial and the fp32 residual stream shared by every rank."""
    gen = torch.Generator(device="cuda").manual_seed(seed * 7919 + dist.get_rank())
    partial = torch.randn(num_tokens, HIDDEN, device="cuda", generator=gen)
    partial = (0.5 * partial).to(torch.bfloat16)
    # Exact zeros take the producer's +0 -> -0 remap.
    partial[0, :64] = 0
    gen = torch.Generator(device="cuda").manual_seed(seed)
    stream = torch.randn(num_tokens, HIDDEN, device="cuda", generator=gen)
    return partial, stream


def _reference(
    comm: CustomAllReduceV2,
    partial: torch.Tensor,
    stream: torch.Tensor,
    kwargs: dict,
    want_router: bool,
) -> tuple:
    two_shot = arn._uses_two_shot(partial.shape[0], comm.world_size)
    comm.override_algo = (
        AllReduceAlgo.TWO_SHOT_PUSH if two_shot else AllReduceAlgo.ONE_SHOT_PUSH
    )
    try:
        reduced = comm.custom_all_reduce(partial)
    finally:
        comm.override_algo = None
    cluster_size, _, _ = arn._launch_config(
        partial.shape[1], partial.shape[0], comm.world_size
    )
    return fused_norm_residual(
        reduced,
        stream,
        POST_EPS,
        PRE_EPS,
        want_router=want_router,
        cluster_size=cluster_size,
        **kwargs,
    )


def _assert_all_ranks(failures: list[str], context: str) -> None:
    """Raise on every rank if any rank failed, so no rank is left waiting in a collective."""
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, failures)
    report = [f"rank {r}: {f}" for r, fs in enumerate(per_rank) for f in fs]
    assert not report, f"{context}: " + "; ".join(report)


def _assert_bitwise(got: tuple, want: tuple, context: str) -> None:
    torch.cuda.synchronize()
    failures = []
    for name, a, b in zip(("output", "new_stream", "router_out"), got, want):
        if (a is None) != (b is None):
            failures.append(f"{name} present in only one of got / want")
        elif a is not None:
            bits = torch.int16 if a.element_size() == 2 else torch.int32
            bad = int((a.view(bits) != b.view(bits)).sum())
            if bad:
                failures.append(f"{name}: {bad} of {a.numel()} elements differ")
    _assert_all_ranks(failures, context)


@pytest.mark.parametrize("want_router", [False, True], ids=["no_router", "router"])
@pytest.mark.parametrize("variant", ["gate", "input_scale", "plain"])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@torch.inference_mode()
def test_matches_push_all_reduce_then_norm(comm, num_tokens, variant, want_router):
    assert arn.all_reduce_norm_residual_supported(comm, num_tokens, HIDDEN)
    kwargs = _boundary_kwargs(variant)
    for rnd in range(REPEATS):
        partial, stream = _make_inputs(num_tokens, 100 * num_tokens + rnd)
        want = _reference(comm, partial, stream, kwargs, want_router)
        got = arn.all_reduce_norm_residual(
            comm, partial, stream, POST_EPS, PRE_EPS, want_router=want_router, **kwargs
        )
        _assert_bitwise(got, want, f"M={num_tokens} {variant} round {rnd}")


@torch.inference_mode()
def test_production_workspace_bounds():
    """At the production push workspace, every cluster-path M fuses, the exchange switches to
    2-shot above 128 / world_size tokens, and the first M whose 2-shot slots exceed the workspace
    falls back to the unfused path."""
    comm = _init_communicator()
    world = comm.world_size
    push_bytes = get_all_reduce_config(world).max_push_bytes(world)
    # The 2-shot slot holds both halves of [M, H / world] bf16.
    max_tokens = push_bytes * world // (4 * HIDDEN)
    one_shot_max = 128 // world
    saved = comm.max_push_size
    comm.max_push_size = push_bytes
    try:
        for m in [1, 8, 16, 32, 64, one_shot_max, one_shot_max + 1, max_tokens]:
            assert arn.all_reduce_norm_residual_supported(comm, m, HIDDEN), m
        assert not arn.all_reduce_norm_residual_supported(comm, max_tokens + 1, HIDDEN)
    finally:
        comm.max_push_size = saved
    assert not arn._uses_two_shot(one_shot_max, world)
    assert arn._uses_two_shot(one_shot_max + 1, world)


@torch.inference_mode()
def test_launch_overrides():
    """Each per-architecture launch override covers exactly its token range, only for its compute
    capability and world size, with a cluster shape the norm can launch; this device's dispatch
    reads the same table, so on SM107 at world 4 the M = 48 cases above run the override."""
    lookup = arn._launch_override_for
    assert [lookup((10, 7), m, 4) for m in (47, 48, 49)] == [None, (False, 4), None]
    assert [lookup((10, 3), m, 4) for m in (47, 48, 64, 65)] == [
        None,
        (True, 1),
        (True, 1),
        None,
    ]
    for capability, world in [((10, 0), 4), ((10, 7), 8), ((10, 3), 8)]:
        assert lookup(capability, 48, world) is None, (capability, world)
    for entries in arn._LAUNCH_OVERRIDES.values():
        for min_tokens, max_tokens, _, cluster_size in entries:
            for m in (min_tokens, max_tokens):
                _cluster_launch_config(HIDDEN, m, cluster_size)

    comm = _init_communicator()
    world = comm.world_size
    want = lookup(torch.cuda.get_device_capability(), 48, world)
    assert arn._launch_override(48, world) == want
    if want is not None:
        assert arn._uses_two_shot(48, world) == want[0]
        assert arn._launch_config(HIDDEN, 48, world)[0] == want[1]
        assert arn.all_reduce_norm_residual_supported(comm, 48, HIDDEN)


def _standalone_inputs(numel: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's input and the exact all-reduce: small integers sum exactly in bf16."""
    parts = [
        torch.randint(
            -4,
            5,
            (numel,),
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed * 7919 + rank),
        ).to(torch.bfloat16)
        for rank in range(dist.get_world_size())
    ]
    return parts[dist.get_rank()], torch.stack(parts).float().sum(0).to(torch.bfloat16)


@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_interleaves_with_standalone_push(comm, use_graph):
    """Fused and standalone push all-reduces share the push slots and epoch counters
    but launch different grids: alternating them at changing sizes must stay exact."""
    kwargs = _boundary_kwargs("gate")
    algos = [AllReduceAlgo.ONE_SHOT_PUSH, AllReduceAlgo.TWO_SHOT_PUSH]
    steps = [(m, n, algo) for (m, n), algo in itertools.product(INTERLEAVED, algos)]
    for num_tokens, *_ in steps:
        assert arn.all_reduce_norm_residual_supported(comm, num_tokens, HIDDEN)
    fused = [_make_inputs(m, 5000 + i) for i, (m, *_) in enumerate(steps)]
    standalone = [
        _standalone_inputs(n, 6000 + i)[0] for i, (_, n, _) in enumerate(steps)
    ]

    def run_steps() -> tuple[list, list]:
        got, reduced = [], []
        for i, ((_, _, algo), (partial, stream), x) in enumerate(
            zip(steps, fused, standalone)
        ):
            # Every launch flips the epoch parity: alternate the order so both kinds
            # of launch use both halves of the double buffer.
            if i % 2:
                got.append(
                    arn.all_reduce_norm_residual(
                        comm, partial, stream, POST_EPS, PRE_EPS, **kwargs
                    )
                )
            comm.override_algo = algo
            try:
                reduced.append(comm.custom_all_reduce(x))
            finally:
                comm.override_algo = None
            if not i % 2:
                got.append(
                    arn.all_reduce_norm_residual(
                        comm, partial, stream, POST_EPS, PRE_EPS, **kwargs
                    )
                )
        return got, reduced

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        got, reduced = run_steps()
        if use_graph:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with comm.capture():
                with torch.cuda.graph(graph, stream=side):
                    got, reduced = run_steps()
    torch.cuda.synchronize()
    for rnd in range(REPEATS):
        expected = []
        for i, (num_tokens, numel, _) in enumerate(steps):
            for dst, src in zip(
                fused[i], _make_inputs(num_tokens, 7000 + 100 * rnd + i)
            ):
                dst.copy_(src)
            x, want = _standalone_inputs(numel, 8000 + 100 * rnd + i)
            standalone[i].copy_(x)
            expected.append(want)
        if use_graph:
            graph.replay()
        else:
            got, reduced = run_steps()
        torch.cuda.synchronize()
        # Standalone push stores an exact-zero sum as -0.0, which compares equal.
        failures = [
            f"standalone step {i}: {int((a.float() != w.float()).sum())} elements differ"
            for i, (a, w) in enumerate(zip(reduced, expected))
            if not torch.equal(a.float(), w.float())
        ]
        _assert_all_ranks(failures, f"round {rnd}")
        for i, ((partial, stream), out) in enumerate(zip(fused, got)):
            want = _reference(comm, partial, stream, kwargs, False)
            _assert_bitwise(out, want, f"round {rnd} fused all-reduce step {i}")


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
