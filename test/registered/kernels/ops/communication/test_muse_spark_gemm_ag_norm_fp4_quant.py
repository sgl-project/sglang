"""Muse Spark pooler: local GEMM slice + FlashInfer's gather + RMSNorm + static NVFP4 quant.

gemm_ag_norm_fp4_quant must match the exactly gathered GEMM slices followed by
fused_norm_fp4_quant_static on every rank and agree across ranks: eagerly and replayed from
CUDA graphs, over rounds that alternate the row count, interleaved with the push-plane
gemm_ag as served (FlashInfer's op shares the push plane and its epoch), and with exact zeros
in the gathered rows (the push plane's "not arrived" marker is +0). Where the dispatcher's slice
GEMM has a peer-store twin (warp split-K, split-K, low-latency GEMM), the GEMM-push path must be
bitwise equal to the publish path.

Run with ``python <this file>``; it relaunches under torchrun on 4 and 8 GPUs.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import logging
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
import sglang.srt.layers.communication.muse_spark_gemm_ag as gemm_ag_mod
import sglang.srt.layers.quantization.unquant as unquant
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication.muse_spark_all_gather import gemm_ag_supported
from sglang.kernels.ops.layernorm.muse_spark_rmsnorm_nvfp4_quant import (
    fused_norm_fp4_quant_static,
)
from sglang.srt.distributed.device_communicators.symm_workspace import (
    WorkspaceRequest,
    request_symm_workspace,
)
from sglang.srt.layers.communication.muse_spark_gemm_ag import (
    gemm_ag,
    gemm_ag_norm_fp4_quant,
    gemm_local,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

K, N = 8192, 4096
EPS = 1e-5
# gemm_ag's push slot holds 1024 rows of N bf16.
PUSH_BYTES = 8 * 1024 * 1024
# Row counts in call order, alternating small and large; the workspace holds 1024 rows at TP4
# and 512 otherwise.
ROWS = [1, 512, 7, 1024, 64, 65, 200]
ROUNDS = 4
# E2M1 codes 0..7 are these magnitudes; bit 3 is the sign.
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_SIGN = 8


@cache_once
def _init():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord, tp_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    lease = request_symm_workspace(
        coord.cpu_group,
        torch.device(f"cuda:{local_rank}"),
        WorkspaceRequest(owner="test_muse_spark_fp4", push_slot_bytes=PUSH_BYTES),
    )
    assert lease is not None
    return coord, lease


def _assert_all_ranks(failures: list[str], context: str) -> None:
    """Raise on every rank if any rank failed, so no rank is left waiting in a collective."""
    per_rank = [None] * dist.get_world_size()
    dist.all_gather_object(per_rank, failures)
    report = [f"rank {r}: {f}" for r, fs in enumerate(per_rank) for f in fs]
    assert not report, f"{context}: " + "; ".join(report)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8)


def _gather(coord, local: torch.Tensor) -> torch.Tensor:
    """Every rank's column slice, gathered bit-exactly over NCCL."""
    parts = [torch.empty_like(local) for _ in range(coord.world_size)]
    dist.all_gather(parts, local.contiguous(), group=coord.device_group)
    return torch.cat(parts, dim=1)


def _nibbles(fp4: torch.Tensor) -> torch.Tensor:
    return torch.stack([fp4 & 0xF, fp4 >> 4], dim=-1).flatten(-2).int()


def _scale(sf: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """Each element's dequant scale: its block's E4M3 SF over the global scale."""
    sf = sf.view(torch.float8_e4m3fn).float() / global_scale
    return sf.repeat_interleave(16, dim=-1)


def _dequant(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    mag = _E2M1.to(codes.device)[codes & (_SIGN - 1)]
    return torch.where(codes & _SIGN != 0, -mag, mag) * scale


def _check(
    coord, fp4, sf, rows, norm_weight, global_scale, neg_zero_in: bool = False
) -> list[str]:
    """Match up to one-step rounding flips; agree across ranks.

    FlashInfer sums the row in another order, so its rstd can differ in the last bit and a normed
    value at a bf16 rounding tie rounds the other way. That moves the E2M1 element one grid step
    or, for a block maximum, the block's E4M3 SF one code (and the block's elements with it).
    Flips cluster: every element of the row with the tied bf16 mantissa flips together (seen: 9
    in one row of 64 x 4096). Allow SF one code off in 4 + 1/256 of the blocks, and elements off
    by at most the largest grid step (2 x SF) in 64 + 1/4096 of them. A sign-of-zero difference
    is not allowed, except with ``neg_zero_in``: FlashInfer reads a real -0 input as +0 (the push
    plane carries +0 as -0, so it cannot tell them apart), so where the gathered row holds -0
    the output may be +0.
    """
    ref_fp4, ref_sf = fused_norm_fp4_quant_static(rows, norm_weight, EPS, global_scale)
    sf_step = (_bits(sf).int() - _bits(ref_sf).int()).abs()
    sf_flips = int((sf_step != 0).sum())
    got, ref = _nibbles(fp4), _nibbles(ref_fp4)
    differ = got != ref
    scale, ref_scale = _scale(sf, global_scale), _scale(ref_sf, global_scale)
    err = (_dequant(got, scale) - _dequant(ref, ref_scale)).abs()
    if neg_zero_in:
        differ &= ~((rows.view(torch.int16) == -32768) & (got == 0))
    zero_sign = differ & (got & (_SIGN - 1) == 0) & (ref & (_SIGN - 1) == 0)
    flips = int(differ.sum())
    failures = []
    if int(sf_step.max()) > 1 or sf_flips > 4 + sf_step.numel() // 2**8:
        failures.append(f"sf: {sf_flips} differ, up to {int(sf_step.max())} codes")
    steps = float((err / torch.maximum(scale, ref_scale)).nan_to_num().max())
    bound = 64 + got.numel() // 2**12 + 16 * sf_flips
    if steps > 2.001 or bool(zero_sign.any()) or flips > bound:
        failures.append(
            f"fp4: {flips} differ (bound {bound}), up to {steps:.2f} x SF, "
            f"{int(zero_sign.sum())} in the sign of 0"
        )
    # Every rank quantizes the same gathered rows, so the outputs must agree across ranks.
    mine = torch.cat([fp4.flatten(), _bits(sf).flatten()])
    peers = [torch.empty_like(mine) for _ in range(coord.world_size)]
    dist.all_gather(peers, mine, group=coord.device_group)
    failures += [
        f"fp4/sf differ from rank {r}"
        for r, p in enumerate(peers)
        if not torch.equal(p, mine)
    ]
    return failures


def _run_step(comm, x, weight, norm_weight, global_scale):
    """The fused op between two push-plane gathers, as the serving step interleaves them."""
    before = gemm_ag(comm, x, weight)
    fp4, sf = gemm_ag_norm_fp4_quant(comm, x, weight, norm_weight, EPS, global_scale)
    return fp4, sf, before, gemm_ag(comm, x, weight)


@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_matches_gathered_gemm_then_norm_quant(use_graph: bool):
    coord, comm = _init()
    if not comm.has_multicast:
        pytest.skip("needs a multicast-capable push plane")
    shared = torch.Generator(device="cuda").manual_seed(0)
    weight = (torch.randn(N, K, device="cuda", generator=shared) / K**0.5).bfloat16()
    weight[::97] = 0  # exact +0 columns in every gathered row
    norm_weight = (1 + 0.1 * torch.randn(N, device="cuda", generator=shared)).bfloat16()
    global_scale = torch.tensor([448.0 * 6.0 / 3.5], device="cuda")
    xs = {}
    for m in ROWS:
        x = torch.empty(m, K, dtype=torch.bfloat16, device="cuda")
        if gemm_ag_supported(comm, x, weight):
            xs[m] = x

    def refill(x):
        x.copy_(torch.randn(x.shape, device="cuda", generator=shared))
        x[3::5] = 0  # whole gathered rows of exact +0

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graphs, outs = {}, {}
    with torch.cuda.stream(side):
        for m, x in xs.items():
            refill(x)
            outs[m] = _run_step(comm, x, weight, norm_weight, global_scale)
            if use_graph:
                torch.cuda.synchronize()
                graphs[m] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[m], stream=side):
                    outs[m] = _run_step(comm, x, weight, norm_weight, global_scale)
    torch.cuda.synchronize()

    for rnd in range(ROUNDS):
        for m, x in xs.items():
            refill(x)
            if use_graph:
                graphs[m].replay()
            else:
                outs[m] = _run_step(comm, x, weight, norm_weight, global_scale)
            torch.cuda.synchronize()
            fp4, sf, before, after = outs[m]
            rows = _gather(coord, gemm_local(comm, x, weight))
            failures = _check(coord, fp4, sf, rows, norm_weight, global_scale)
            # The push plane carries +0 as -0, so compare its rows by value.
            for name, got in (("gemm_ag before", before), ("gemm_ag after", after)):
                if not torch.equal(got.float(), rows.float()):
                    failures.append(f"{name} rows differ")
            _assert_all_ranks(failures, f"round {rnd} rows {m}")


# (N, K) of the replicated weight: the pooler, and a shape whose slices take warp split-K.
PUSH_SHAPES = [(4096, 8192), (8192, 4096)]
PUSH_ROWS = [1, 2, 8, 64, 256, 512]


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


def _push_step(comm, x, weight, norm_weight, global_scale):
    """GEMM push and publish of the fused op, between push-plane gathers."""
    before = gemm_ag(comm, x, weight)
    push = gemm_ag_norm_fp4_quant(comm, x, weight, norm_weight, EPS, global_scale)
    with patch.object(unquant, "bf16_gemm_peer_store", lambda *_, **__: False):
        publish = gemm_ag_norm_fp4_quant(
            comm, x, weight, norm_weight, EPS, global_scale
        )
    return push, publish, before, gemm_ag(comm, x, weight)


def _check_push(coord, comm, n, k, use_graph, gen, global_scale) -> set[str]:
    """Push vs publish over PUSH_ROWS for a replicated [n, k] weight; returns the routes."""
    weight = (torch.randn(n, k, device="cuda", generator=gen) / k**0.5).bfloat16()
    weight[::97] = 0
    norm_weight = (1 + 0.1 * torch.randn(n, device="cuda", generator=gen)).bfloat16()
    xs = {}
    for m in PUSH_ROWS:
        x = torch.empty(m, k, dtype=torch.bfloat16, device="cuda")
        if gemm_ag_supported(comm, x, weight):
            xs[m] = x

    def refill(x):
        x.copy_(torch.randn(x.shape, device="cuda", generator=gen))
        x[3::5] = 0

    def step(x):
        return _push_step(comm, x, weight, norm_weight, global_scale)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graphs, outs, routes, launched = {}, {}, {}, []
    with torch.cuda.stream(side), _record_peer_stores(launched):
        for m, x in xs.items():
            refill(x)
            launched.clear()
            outs[m] = step(x)
            routes[m] = launched[0] if launched else "publish"
            if use_graph:
                torch.cuda.synchronize()
                graphs[m] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[m], stream=side):
                    outs[m] = step(x)
    torch.cuda.synchronize()

    for rnd in range(2):
        for m, x in xs.items():
            refill(x)
            if use_graph:
                graphs[m].replay()
            else:
                outs[m] = step(x)
            torch.cuda.synchronize()
            (fp4, sf), (ref_fp4, ref_sf), before, after = outs[m]
            rows = _gather(coord, gemm_local(comm, x, weight))
            failures = _check(coord, ref_fp4, ref_sf, rows, norm_weight, global_scale)
            route = routes[m]
            same = torch.equal(fp4, ref_fp4) and torch.equal(_bits(sf), _bits(ref_sf))
            if not same:
                failures.append(f"{route}: push differs from publish")
            for name, got in (("gemm_ag before", before), ("gemm_ag after", after)):
                if not torch.equal(got.float(), rows.float()):
                    failures.append(f"{name} rows differ")
            _assert_all_ranks(failures, f"{n}x{k} round {rnd} rows {m} ({route})")
    return set(routes.values())


@pytest.mark.parametrize("use_graph", [False, True], ids=["eager", "graph"])
def test_gemm_push_matches_publish(use_graph: bool):
    coord, comm = _init()
    if not comm.has_multicast:
        pytest.skip("needs a multicast-capable push plane")
    gen = torch.Generator(device="cuda").manual_seed(1)
    global_scale = torch.tensor([448.0 * 6.0 / 3.5], device="cuda")
    routes = set()
    # The serving dispatcher (split-K tables, low-latency GEMM) without the low-M autotune,
    # which would take M <= 16 off the peer-store GEMMs; restored afterwards.
    with (
        patch.dict(unquant.__dict__),
        get_context().override_server_args(
            bf16_gemm_backend="cutedsl", disable_flashinfer_autotune=True
        ),
    ):
        unquant.initialize_bf16_gemm_config()
        for n, k in PUSH_SHAPES:
            routes |= _check_push(coord, comm, n, k, use_graph, gen, global_scale)
    # Every peer-store GEMM and the fallback actually ran.
    assert routes >= {
        "warp_splitk",
        "splitk",
        "low_latency",
        "publish",
    }, routes


def test_gemm_push_low_m_autotune_publishes():
    """With FlashInfer's low-M autotune, M <= 16 publishes (no peer-store GEMM)
    and matches the forced publish path."""
    coord, comm = _init()
    if not comm.has_multicast:
        pytest.skip("needs a multicast-capable push plane")
    gen = torch.Generator(device="cuda").manual_seed(2)
    global_scale = torch.tensor([448.0 * 6.0 / 3.5], device="cuda")
    n, k = PUSH_SHAPES[0]
    weight = (torch.randn(n, k, device="cuda", generator=gen) / k**0.5).bfloat16()
    norm_weight = (1 + 0.1 * torch.randn(n, device="cuda", generator=gen)).bfloat16()
    launched = []
    with (
        patch.dict(unquant.__dict__),
        get_context().override_server_args(bf16_gemm_backend="cutedsl"),
        _record_peer_stores(launched),
    ):
        unquant.initialize_bf16_gemm_config()
        if unquant._mm_bf16 is None:
            # SM107 never enables the autotune: sentinels, never called in eager mode.
            unquant._mm_bf16 = object()
            unquant._flashinfer_autotuner = SimpleNamespace(is_tuning_mode=False)
        for m in (1, 8, unquant._BF16_LOW_M_AUTOTUNE_MAX_M):
            x = torch.randn(m, k, device="cuda", generator=gen).bfloat16()
            if not gemm_ag_supported(comm, x, weight):
                continue
            push, publish, _, _ = _push_step(comm, x, weight, norm_weight, global_scale)
            torch.cuda.synchronize()
            same = torch.equal(push[0], publish[0]) and torch.equal(
                _bits(push[1]), _bits(publish[1])
            )
            _assert_all_ranks(
                [] if same else ["differs from publish"], f"autotune rows {m}"
            )
    assert not launched, launched


class _Lease:
    """A stand-in lease with the fields _norm_quant_op reads; unhashable, like the real one."""

    __hash__ = None

    def __init__(self, plane):
        self.rank, self.world_size = 0, 2
        self.capacity = SimpleNamespace(push_slot_bytes=1 << 20)
        self.data = SimpleNamespace(push_slots=[plane], push_mc=16)
        self.control = SimpleNamespace(epoch=plane)


def test_norm_quant_op_per_lease_and_width():
    """One op per (lease, n) without resetting the cache; each eps compiled on it."""

    class _Op:
        def __init__(self, world, rank, max_tokens, n, *, group, push_plane):
            self.n, self.plane, self.eps = n, push_plane.buffer, []

        def compile(self, *, eps):
            self.eps.append(eps)

    leases = [_Lease(object()), _Lease(object())]
    with (
        patch.dict(gemm_ag_mod._NORM_QUANT_OPS, clear=True),
        patch(
            "flashinfer.comm.mnnvl_cutedsl_ag.MNNVLCuteDSLAllGatherNormQuantWorkspace",
            _Op,
        ),
        patch("sglang.srt.runtime_context.get_parallel"),
    ):
        a = gemm_ag_mod._norm_quant_op(leases[0], 4096, 1e-5)
        b = gemm_ag_mod._norm_quant_op(leases[0], 8192, 1e-5)
        c = gemm_ag_mod._norm_quant_op(leases[1], 4096, 1e-6)
        assert gemm_ag_mod._norm_quant_op(leases[0], 4096, 1e-6) is a
    assert (a.n, b.n, c.n) == (4096, 8192, 4096)
    assert a.plane is b.plane is leases[0].data.push_slots[0]
    assert c.plane is leases[1].data.push_slots[0]
    assert a.eps == [1e-5, 1e-6] and c.eps == [1e-6]


@pytest.mark.parametrize("rows", [1, 64, 512])
def test_signed_zeros(rows: int):
    """Exact +0 and -0 in the shard, and whole rows of each, through FlashInfer's op."""
    from flashinfer.comm.mnnvl_cutedsl_ag import all_gather_rmsnorm_fp4_quantize

    coord, comm = _init()
    if not comm.has_multicast:
        pytest.skip("needs a multicast-capable push plane")
    workspace = gemm_ag_mod._norm_quant_op(comm, N, EPS)
    g = torch.Generator(device="cuda").manual_seed(rows * 64 + coord.rank)
    local = torch.randn(rows, N // coord.world_size, device="cuda", generator=g)
    local[:, ::7] = 0.0
    local[:, 3::7] = -0.0
    local[1::4] = 0.0
    local[2::4] = -0.0
    local = local.bfloat16()
    norm_weight = torch.ones(N, dtype=torch.bfloat16, device="cuda")
    global_scale = torch.tensor([448.0 * 6.0 / 3.5], device="cuda")

    for rnd in range(ROUNDS):
        fp4, sf = all_gather_rmsnorm_fp4_quantize(
            local, norm_weight, global_scale, workspace, eps=EPS
        )
        torch.cuda.synchronize()
        failures = _check(
            coord, fp4, sf, _gather(coord, local), norm_weight, global_scale, True
        )
        _assert_all_ranks(failures, f"round {rnd} rows {rows}")


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4, 8))
