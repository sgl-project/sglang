"""Muse Spark attention all-reduce fused with the norm-residual boundary, multi-GPU.

FlashInfer's fused kernel on CustomAllReduceV2's push plane must match the push
all-reduce followed by ``fused_norm_residual`` within BF16 rounding, on every rank.
Its own exchanges (pushed, published 1-shot, 2-shot) must match each other bit for
bit. o_proj's peer-store GEMM pushes with unicast and, where supported, multicast stores.

Usage::

    python test_muse_spark_collective_norm_residual_multigpu.py --num-gpu 4
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import itertools
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
import sglang.srt.layers.quantization.unquant as unquant
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
from sglang.srt.runtime_context import get_context, get_parallel
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


def _fused(
    comm: CustomAllReduceV2,
    partial: torch.Tensor | None,
    stream: torch.Tensor,
    kwargs: dict,
    want_router: bool = False,
) -> tuple:
    """The fused boundary as the model calls it, with ``comm`` as the attention TP communicator."""
    jit_kwargs = dict(
        post_eps=POST_EPS,
        pre_eps=PRE_EPS,
        post_norm_is_input_scale=False,
        post_weight=None,
        pre_weight=None,
        alpha=None,
        beta=None,
    )
    boundary = SimpleNamespace(jit_kernel_kwargs=lambda _dtype: jit_kwargs | kwargs)
    fused = arn.FusedAllReducePostNormResidualPreNorm(boundary, HIDDEN, torch.bfloat16)
    parallel = SimpleNamespace(attn_tp_group=SimpleNamespace(ca_comm=comm))
    router_dtype = torch.float32 if want_router else None
    with patch.object(arn, "get_parallel", lambda: parallel):
        return fused(stream, partial, router_dtype=router_dtype)


def _reference(
    comm: CustomAllReduceV2,
    partial: torch.Tensor,
    stream: torch.Tensor,
    kwargs: dict,
    want_router: bool,
) -> tuple:
    plan = arn._launch_plan(partial.shape[0], partial.shape[1], comm.world_size)
    comm.override_algo = (
        AllReduceAlgo.ONE_SHOT_PUSH if plan.one_shot else AllReduceAlgo.TWO_SHOT_PUSH
    )
    try:
        reduced = comm.custom_all_reduce(partial)
    finally:
        comm.override_algo = None
    return fused_norm_residual(
        reduced,
        stream,
        POST_EPS,
        PRE_EPS,
        want_router=want_router,
        cluster_size=plan.cluster_size,
        **kwargs,
    )


def _assert_all_ranks(failures: list[str], context: str) -> None:
    """Raise on every rank if any rank failed, so no rank is left waiting in a collective."""
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, failures)
    report = [f"rank {r}: {f}" for r, fs in enumerate(per_rank) for f in fs]
    assert not report, f"{context}: " + "; ".join(report)


def _assert_close(got: tuple, want: tuple, context: str) -> None:
    """Against the push all-reduce + norm reference: the reduction orders differ, so
    allow BF16-rounding differences (FlashInfer's dual-norm test tolerances)."""
    torch.cuda.synchronize()
    failures = []
    for name, a, b in zip(("output", "new_stream", "router_out"), got, want):
        if (a is None) != (b is None):
            failures.append(f"{name} present in only one of got / want")
        elif a is not None and not torch.allclose(
            a.float(), b.float(), rtol=1 / 64, atol=1 / 32
        ):
            diff = (a.float() - b.float()).abs()
            failures.append(f"{name}: max abs diff {diff.max().item():.3g}")
    _assert_all_ranks(failures, context)


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
        got = _fused(comm, partial, stream, kwargs, want_router)
        _assert_close(got, want, f"M={num_tokens} {variant} round {rnd}")


@pytest.mark.parametrize("num_tokens", [1, 3, 16, 17, 32])
@torch.inference_mode()
def test_one_shot_matches_two_shot(comm, num_tokens):
    """In the 1-shot range the published 1-shot exchange is bitwise the 2-shot one (both round the
    same fp32 rank-order sum once; only the sign of an exact-zero sum may differ, and the stream has no -0)."""
    world = comm.world_size
    plan = arn._launch_plan(num_tokens, HIDDEN, world)
    if not plan.one_shot:
        pytest.skip(f"M={num_tokens} is 2-shot at world {world}")
    two_shot = plan._replace(
        one_shot=False, slot_bytes=4 * num_tokens * HIDDEN // world
    )
    kwargs = _boundary_kwargs("input_scale")
    for rnd in range(REPEATS):
        partial, stream = _make_inputs(num_tokens, 300 * num_tokens + rnd)
        got = _fused(comm, partial, stream, kwargs, want_router=True)
        with patch.object(arn, "_launch_plan", lambda *_: two_shot):
            want = _fused(comm, partial, stream, kwargs, want_router=True)
        _assert_bitwise(got, want, f"M={num_tokens} round {rnd}")


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
    saved = comm._max_push_size
    comm._max_push_size = push_bytes
    try:
        for m in [1, 8, 16, 32, 64, one_shot_max, one_shot_max + 1, max_tokens]:
            assert arn.all_reduce_norm_residual_supported(comm, m, HIDDEN), m
        assert not arn.all_reduce_norm_residual_supported(comm, max_tokens + 1, HIDDEN)
    finally:
        comm._max_push_size = saved
    assert arn._launch_plan(one_shot_max, HIDDEN, world).one_shot
    assert not arn._launch_plan(one_shot_max + 1, HIDDEN, world).one_shot


@torch.inference_mode()
def test_launch_overrides():
    """Each per-architecture launch override covers exactly its token range, only for its compute
    capability and world size, with a cluster shape the norm can launch; this device's dispatch
    reads the same table, so on SM107 at world 4 the M = 48 cases above run the override."""
    lookup = arn._launch_override_for
    assert [lookup((10, 7), m, 4) for m in (47, 48, 49)] == [None, (False, 4), None]
    for capability, world in [((10, 0), 4), ((10, 3), 4), ((10, 7), 8), ((10, 3), 8)]:
        assert lookup(capability, 48, world) is None, (capability, world)
    for entries in arn._LAUNCH_OVERRIDES.values():
        for min_tokens, max_tokens, _, cluster_size in entries:
            for m in (min_tokens, max_tokens):
                _cluster_launch_config(HIDDEN, m, cluster_size)

    comm = _init_communicator()
    world = comm.world_size
    want = lookup(torch.cuda.get_device_capability(), 48, world)
    if want is not None:
        plan = arn._launch_plan(48, HIDDEN, world)
        assert (not plan.one_shot, plan.cluster_size) == want
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
    """The fused all-reduce and the standalone push all-reduce share the push slots and epoch
    counter but launch different grids: alternating them must stay exact."""
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
                got.append(_fused(comm, partial, stream, kwargs))
            comm.override_algo = algo
            try:
                reduced.append(comm.custom_all_reduce(x))
            finally:
                comm.override_algo = None
            if not i % 2:
                got.append(_fused(comm, partial, stream, kwargs))
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
            _assert_close(out, want, f"round {rnd} fused all-reduce step {i}")


# o_proj rows on the 1-shot exchange at worlds 2 / 4 / 8, through each peer-store GEMM.
PUSH_ROWS = [1, 8, 12, 16, 32, 48, 64]
# Route name -> FlashInfer peer-store GEMM that bf16_gemm_peer_store may launch.
_PEER_STORE_RUNS = {
    "warp_splitk": (
        "flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk",
        "run_warp_splitk_dense_peer_store",
    ),
    "splitk": (
        "flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk",
        "run_splitk_dense_peer_store",
    ),
    "low_latency": (
        "flashinfer.gemm.kernels.tgv_gemm_cute_ext",
        "run_tgv_cute_ext_peer_store",
    ),
}


@contextlib.contextmanager
def _record_peer_stores(launched: list):
    """Append the route of every peer-store GEMM actually launched to ``launched``."""
    with contextlib.ExitStack() as stack:
        for route, (module, name) in _PEER_STORE_RUNS.items():
            mod = importlib.import_module(module)

            def spy(*args, _run=getattr(mod, name), _route=route, **kwargs):
                launched.append(_route)
                return _run(*args, **kwargs)

            stack.enter_context(patch.object(mod, name, spy))
        yield


@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
@torch.inference_mode()
def test_o_proj_push_matches_unpushed(comm, use_graph):
    """o_proj stored straight into the 1-shot slots, then the pre-pushed boundary, is bitwise the
    boundary of o_proj's output published 1-shot; unpushed launches and standalone push all-reduces
    sit in between."""
    world = comm.world_size
    k = HIDDEN // world  # o_proj in_features per rank
    kwargs = _boundary_kwargs("gate")
    rows = [
        m
        for m in PUSH_ROWS
        if arn.all_reduce_norm_residual_supported(comm, m, HIDDEN)
        and arn._launch_plan(m, HIDDEN, world).one_shot
    ]
    gen = torch.Generator(device="cuda").manual_seed(11 + dist.get_rank())
    weight = (torch.randn(HIDDEN, k, device="cuda", generator=gen) / k**0.5).bfloat16()
    xs = {m: torch.empty(m, k, dtype=torch.bfloat16, device="cuda") for m in rows}
    streams = {m: torch.empty(m, HIDDEN, device="cuda") for m in rows}
    standalone = _standalone_inputs(64 * HIDDEN, 1)[0]

    def refill(m: int, seed: int) -> None:
        g = torch.Generator(device="cuda").manual_seed(seed * 7919 + dist.get_rank())
        xs[m].copy_(torch.randn(xs[m].shape, device="cuda", generator=g))
        if dist.get_rank() == 0:
            xs[m][0] = 0  # an exact-zero partial row takes the +0 -> -0 remap
        g.manual_seed(seed)
        streams[m].copy_(torch.randn(streams[m].shape, device="cuda", generator=g))

    def step(m: int):
        x, stream = xs[m], streams[m]
        want = _fused(comm, unquant.bf16_gemm_dispatch(x, weight, None), stream, kwargs)
        comm.custom_all_reduce(standalone)
        pushed = arn.o_proj_push(comm, x, weight)
        partial = None if pushed else unquant.bf16_gemm_dispatch(x, weight, None)
        got = _fused(comm, partial, stream, kwargs)
        return got, want

    # The serving dispatcher (split-K tables, low-latency GEMM) without the low-M autotune,
    # which would take M <= 16 off the peer-store GEMMs; restored afterwards.
    with (
        patch.dict(unquant.__dict__),
        get_context().override_server_args(
            bf16_gemm_backend="cutedsl", disable_flashinfer_autotune=True
        ),
    ):
        unquant.initialize_bf16_gemm_config()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        graphs, outs, routes, launched = {}, {}, {}, []
        with torch.cuda.stream(side), _record_peer_stores(launched):
            for m in rows:
                refill(m, m)
                launched.clear()
                outs[m] = step(m)
                routes[m] = launched[0] if launched else "unpushed"
                if use_graph:
                    torch.cuda.synchronize()
                    graphs[m] = torch.cuda.CUDAGraph()
                    with comm.capture(), torch.cuda.graph(graphs[m], stream=side):
                        outs[m] = step(m)
        torch.cuda.synchronize()
        for rnd in range(REPEATS):
            for m in rows:
                refill(m, 1000 * rnd + m)
                side.wait_stream(torch.cuda.current_stream())
                if use_graph:
                    graphs[m].replay()
                else:
                    with torch.cuda.stream(side):
                        outs[m] = step(m)
                got, want = outs[m]
                _assert_bitwise(got, want, f"M={m} {routes[m]} round {rnd}")
    print(f"world {world} o_proj push routes: {routes}")
    # Every rank pushes the same rows; at least one row takes a peer-store GEMM.
    assert set(routes.values()) - {"unpushed"}, routes


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
