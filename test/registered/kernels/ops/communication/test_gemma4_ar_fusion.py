"""Gemma-4 fused all-reduce + norm -> residual add -> norm -> per-128 FP8 quant,
against an fp32 reference of the unfused path; every rank must receive the same
rows. Decode-sized sites also replay inside CUDA graphs next to the stock
all-reduce.

Run with ``python <this file>``; it relaunches under torchrun on 2, 4 and 8 GPUs.
"""

from __future__ import annotations

import atexit
import logging
import os
import random
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import fused_all_reduce
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.configs.custom_all_reduce_v2 import (
    get_all_reduce_config,
)
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.communication import gemma4_ar_fusion as fusion_glue
from sglang.srt.layers.quantization.fp8_utils import (
    deepgemm_w8a8_block_fp8_linear_with_fallback,
    requant_weight_ue8m0,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

# SM90 only, like the model's gate; 8 GPUs cover TP2, TP4 and TP8
register_cuda_ci(est_time=120, stage="extra-b", runner_config="8-gpu-h200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="the Gemma-4 all-reduce fusion runs on SM90 only",
)

HIDDENS = [5376, 4096]  # Gemma-4 31B, and one more block width
EPS = 1e-6
GROUP = fusion_glue.GROUP
MAX_ROWS = 1024
# fewer rows than ranks, trailing ranks owning none (5 at TP4, 9 at TP8), ragged
# shards, and a full barrier-plane grid
ROWS = [1, 5, 9, 512, 1024]
# decode CUDA graph batch sizes: below the TP size, ragged, and up to the
# largest default decode graph
DECODE_BATCHES = [1, 2, 3, 4, 7, 8, 16, 32, 48, 64, 96, 128, 160]


class _Symm:
    def __init__(self, shape, dtype, group_name):
        self.tensor = _SymmetricMemory.empty_strided_p2p(
            shape, [shape[1], 1], dtype, torch.device("cuda"), group_name
        )
        handle = _SymmetricMemory.rendezvous(self.tensor)
        assert handle is not None and handle.multicast_ptr != 0, (
            "needs NVLink multicast"
        )
        self.mc = int(handle.multicast_ptr)
        self.tensor.zero_()


_stock_comm: CustomAllReduceV2 | None = None


@cache_once
def _init():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)), local_rank=local_rank, backend="nccl"
    )
    get_parallel().override_permanently(world_group=coord)
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    comm = CustomAllReduceV2(
        coord.cpu_group,
        torch.device(f"cuda:{local_rank}"),
        max_pull_size=1024 * 1024,
        max_push_size=0,
        max_push_blocks=0,
    )
    assert not comm.disabled and comm.has_multicast, "needs NVLink multicast"
    register_comm_cleanup(comm)
    global _stock_comm  # the stock all-reduce the decode-graph test interleaves
    _stock_comm = comm
    return coord


@cache_once
def _buffers(hidden: int):
    coord = _init()
    name = coord.cpu_group.group_name
    return (
        _Symm((MAX_ROWS, hidden), torch.bfloat16, name),
        _Symm((MAX_ROWS, hidden), torch.float8_e4m3fn, name),
        _Symm((MAX_ROWS, hidden // GROUP), torch.float32, name),
    )


def _gather(t: torch.Tensor) -> list[torch.Tensor]:
    coord = _init()
    t = t.contiguous()
    out = [torch.empty_like(t) for _ in range(coord.world_size)]
    dist.all_gather(out, t, group=coord.device_group)
    return out


def _rows(num_tokens: int, rank: int, world_size: int):
    per_rank = (num_tokens + world_size - 1) // world_size
    row0 = min(rank * per_rank, num_tokens)
    return row0, max(0, min(num_tokens, row0 + per_rank) - row0)


def _rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def _reference(partials, residual, w_a, w_b, scalar):
    """Hugging Face's layer in fp32 with its bf16 rounding points; no scalar on
    the attention site."""
    x = torch.stack([p.float() for p in partials]).sum(0)
    a = _rmsnorm(x, w_a).bfloat16()
    r = (a.float() + residual.float()).bfloat16()
    if scalar is not None:
        r = (r.float() * scalar.bfloat16().float()).bfloat16()
    y = _rmsnorm(r.float(), w_b).bfloat16().float()
    return r, y


def _site(bufs, num_tokens, scales, residual, w_a, w_b, scalar, row0, rows):
    """One fused site over ``bufs`` (partial, out, scales), at the model's grid;
    ``scales`` is the scale view at the scale buffer's base."""
    partial, out, scales_buf = bufs
    fused_all_reduce.pull.all_reduce_norm_residual_norm_quant(
        _stock_comm.obj,
        partial.tensor[:num_tokens],
        partial.mc,
        residual,
        w_a,
        w_b,
        scalar,
        EPS,
        out.tensor[:num_tokens],
        out.mc,
        scales,
        scales_buf.mc,
        row0=row0,
        rows=rows,
        num_blocks=get_all_reduce_config(_stock_comm.world_size).num_pull_blocks,
        rows_per_group=1,
    )


def _final(partial, num_tokens, residual, w_a, scalar, row0, rows):
    """The last layer's site, in place on the partial buffer."""
    staged = partial.tensor[:num_tokens]
    fused_all_reduce.pull.all_reduce_norm_residual(
        _stock_comm.obj,
        staged,
        partial.mc,
        residual,
        w_a,
        scalar,
        EPS,
        staged,
        partial.mc,
        row0=row0,
        rows=rows,
        num_blocks=get_all_reduce_config(_stock_comm.world_size).num_pull_blocks,
        rows_per_group=1,
    )


def _inputs(rows: int, hidden: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(rows, hidden, device="cuda", generator=g).bfloat16()
    w_a = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).bfloat16()
    w_b = (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).bfloat16()
    scalar = torch.full((1,), 0.75, device="cuda")
    return residual, w_a, w_b, scalar


def _scales_view(scales_buf, rows: int, layout: str) -> torch.Tensor:
    """[rows, groups] scales at the buffer's base: row-major, or the model's
    column-major view (DeepGEMM's layout)."""
    if layout == "row":
        return scales_buf.tensor[:rows]
    return fusion_glue.scales_view(scales_buf.tensor, scales_buf.tensor.shape[1], rows)


@pytest.mark.parametrize("hidden", HIDDENS)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("use_scalar", [False, True])  # attention / MLP site
@pytest.mark.parametrize("layout", ["row", "col"])
def test_site(hidden: int, rows: int, use_scalar: bool, layout: str):
    coord = _init()
    partial_buf, out_buf, scales_buf = _buffers(hidden)
    residual_all, w_a, w_b, scalar = _inputs(rows, hidden, 0)
    scalar = scalar if use_scalar else None
    partial = torch.randn(rows, hidden, device="cuda").bfloat16() * (coord.rank + 1) / 4
    row0, mine = _rows(rows, coord.rank, coord.world_size)
    residual = residual_all[row0 : row0 + mine].clone()
    partial_buf.tensor[:rows].copy_(partial)
    scales_view = _scales_view(scales_buf, rows, layout)
    if layout == "col":  # DeepGEMM's layout, at the base the multicast VA maps
        assert scales_view.stride() == (1, (rows + 3) // 4 * 4)
        assert scales_view.data_ptr() == scales_buf.tensor.data_ptr()
    _site(_buffers(hidden), rows, scales_view, residual, w_a, w_b, scalar, row0, mine)
    torch.cuda.synchronize()
    out, scales = out_buf.tensor[:rows].clone(), scales_view.contiguous()

    ref_res, ref_y = _reference(_gather(partial), residual_all, w_a, w_b, scalar)
    # gather before any assert: a rank that fails alone would hang its peers
    all_out, all_scales = _gather(out), _gather(scales)
    # this rank's residual rows, then every rank's view of the quantized rows
    # multimem.ld_reduce does not round to nearest, so x can sit one bf16 ulp off the
    # fp32 sum; that moves a = bf16(norm(x)) by up to two ulps (2/32 for |a| < 8), times 0.75
    torch.testing.assert_close(
        residual.float(), ref_res[row0 : row0 + mine].float(), rtol=2e-2, atol=5e-2
    )
    deq = out.float() * scales.repeat_interleave(GROUP, dim=1)
    torch.testing.assert_close(
        deq, ref_y, rtol=7e-2, atol=4e-2
    )  # e4m3 has 3 mantissa bits
    assert torch.all(scales > 0)
    for name, t, gathered in (("out", out, all_out), ("scales", scales, all_scales)):
        for i, other in enumerate(gathered):
            assert torch.equal(other, t), f"{name} differs on rank {i}"


@pytest.mark.parametrize("hidden", HIDDENS[:1])
@pytest.mark.parametrize("rows", ROWS)
def test_final(hidden: int, rows: int):
    coord = _init()
    partial_buf, _, _ = _buffers(hidden)
    residual_all, w_a, _, scalar = _inputs(rows, hidden, 1)
    partial = torch.randn(rows, hidden, device="cuda").bfloat16() * (coord.rank + 1) / 4
    row0, mine = _rows(rows, coord.rank, coord.world_size)
    residual = residual_all[row0 : row0 + mine].clone()
    staged = partial_buf.tensor[:rows]
    staged.copy_(partial)
    _final(partial_buf, rows, residual, w_a, scalar, row0, mine)
    torch.cuda.synchronize()
    out = staged.clone()
    ref_res, _ = _reference(_gather(partial), residual_all, w_a, w_a, scalar)
    all_out = _gather(out)
    torch.testing.assert_close(out.float(), ref_res.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(residual, out[row0 : row0 + mine])
    for i, other in enumerate(all_out):
        assert torch.equal(other, out), f"out differs on rank {i}"


@pytest.mark.parametrize("rows", [512, 1024])
def test_site_reads_a_gemm_written_partial(rows: int):
    """The model path: the block-FP8 GEMM that produces the partial writes the
    site's symmetric buffer itself (RowParallelLinear's ``output_tensor``), with
    no staging copy. Same result as the GEMM computed on its own."""
    if not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        pytest.skip("DeepGEMM unavailable")
    hidden, k = HIDDENS[0], 1024
    coord = _init()
    partial_buf, out_buf, scales_buf = _buffers(hidden)
    residual_all, w_a, w_b, scalar = _inputs(rows, hidden, 2)
    g = torch.Generator(device="cuda").manual_seed(3)  # the same weight on every rank
    weight = (40 * torch.randn(hidden, k, device="cuda", generator=g)).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.full((hidden // GROUP, k // GROUP), 1e-3, device="cuda")
    if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0:
        # Blackwell DeepGEMM reads packed UE8M0 scales, which the loader requantizes to
        weight, weight_scale = requant_weight_ue8m0(
            weight, weight_scale, [GROUP, GROUP]
        )
    x = torch.randn(rows, k, device="cuda").bfloat16() * (coord.rank + 1) / 4
    gemm = dict(input=x, weight=weight, block_size=[GROUP, GROUP])
    gemm["weight_scale"] = weight_scale
    partial = deepgemm_w8a8_block_fp8_linear_with_fallback(**gemm)
    row0, mine = _rows(rows, coord.rank, coord.world_size)
    residual = residual_all[row0 : row0 + mine].clone()

    staged = partial_buf.tensor[:rows]
    staged.fill_(float("nan"))
    with mock.patch.object(torch.Tensor, "copy_", side_effect=AssertionError("copied")):
        written = deepgemm_w8a8_block_fp8_linear_with_fallback(**gemm, output=staged)
    wrote_in_place = written.data_ptr() == staged.data_ptr()
    same_as_gemm = torch.equal(staged, partial)
    _site(
        _buffers(hidden),
        rows,
        scales_buf.tensor[:rows],
        residual,
        w_a,
        w_b,
        None,
        row0,
        mine,
    )
    torch.cuda.synchronize()
    out, scales = out_buf.tensor[:rows].clone(), scales_buf.tensor[:rows].clone()
    ref_res, ref_y = _reference(_gather(partial), residual_all, w_a, w_b, None)
    # gather before any assert: a rank that fails alone would hang its peers
    all_out = _gather(out)
    assert wrote_in_place and same_as_gemm
    torch.testing.assert_close(
        residual.float(), ref_res[row0 : row0 + mine].float(), rtol=2e-2, atol=5e-2
    )
    deq = out.float() * scales.repeat_interleave(GROUP, dim=1)
    torch.testing.assert_close(deq, ref_y, rtol=7e-2, atol=4e-2)
    for i, other in enumerate(all_out):
        assert torch.equal(other, out), f"out differs on rank {i}"


def test_stage_skips_the_copy_for_its_own_buffer():
    """Gemma4ArFusion stages a foreign partial by copy and takes one already in
    its buffer (``partial_out``) as is."""
    hidden, rows = HIDDENS[0], 8
    _init()
    partial_buf, _, _ = _buffers(hidden)
    fusion = object.__new__(fusion_glue.Gemma4ArFusion)
    fusion.partial = fusion_glue._Symm(partial_buf.tensor, partial_buf.mc)
    own = fusion.partial_out(rows)
    assert own.data_ptr() == partial_buf.tensor.data_ptr() and own.shape[0] == rows
    with mock.patch.object(torch.Tensor, "copy_", side_effect=AssertionError("copied")):
        assert fusion._stage(own).data_ptr() == own.data_ptr()
    foreign = torch.randn(rows, hidden, device="cuda").bfloat16()
    staged = fusion._stage(foreign)
    assert staged.data_ptr() == own.data_ptr() and torch.equal(staged, foreign)


def test_decode_graphs_replay_with_stock_all_reduce():
    """One captured CUDA graph per decode batch size, each running a stock
    CustomAllReduceV2 call (it shares the fused kernel's barrier plane at a
    different grid) and an attention site, an MLP site and the final site.
    Replayed in shuffled order with fresh inputs, with one rank delayed before
    the replay and another inside the graph; every replay matches the fp32
    reference and the stock all-reduce matches the exact sum."""
    hidden = HIDDENS[0]
    coord = _init()
    rank, world_size = coord.rank, coord.world_size
    comm = _stock_comm
    partial_buf, out_buf, scales_buf = _buffers(hidden)
    g = torch.Generator(device="cuda").manual_seed(4)  # same weights on every rank
    w = [
        (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).bfloat16()
        for _ in range(5)
    ]
    scalar = torch.full((1,), 0.75, device="cuda")
    delay = 200_000  # GPU cycles, ~0.1 ms

    graphs = {}

    def body(bs, st):
        """The decode step under test; ``st`` holds its static tensors."""
        row0, mine = _rows(bs, rank, world_size)
        st["side"] = comm.custom_all_reduce(st["small"])
        outs = []
        for i, (w_a, w_b, s) in enumerate(((w[0], w[1], None), (w[2], w[3], scalar))):
            if i == 1 and rank == (bs + 1) % world_size:
                torch.cuda._sleep(delay)  # a peer arrives one barrier late
            partial_buf.tensor[:bs].copy_(st["p"][i])  # o_proj / down_proj output
            _site(
                _buffers(hidden),
                bs,
                _scales_view(scales_buf, bs, "col"),  # the model's layout
                st["res"],
                w_a,
                w_b,
                s,
                row0,
                mine,
            )
            # the next GEMM consumes them before the next site overwrites the buffers
            outs.append(
                (
                    out_buf.tensor[:bs].clone(),
                    _scales_view(scales_buf, bs, "col").clone(),
                )
            )
        partial_buf.tensor[:bs].copy_(st["p"][2])
        _final(partial_buf, bs, st["res"], w[4], scalar, row0, mine)
        st["outs"], st["final"] = outs, partial_buf.tensor[:bs].clone()

    def static(bs):
        row0, mine = _rows(bs, rank, world_size)
        return {
            "small": torch.zeros(bs, 256, device="cuda", dtype=torch.bfloat16),
            "p": [
                torch.zeros(bs, hidden, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ],
            "res": torch.zeros(mine, hidden, device="cuda", dtype=torch.bfloat16),
        }

    for bs in DECODE_BATCHES:  # warm up (JIT builds) eagerly
        body(bs, static(bs))
    torch.cuda.synchronize()
    with comm.capture():
        for bs in DECODE_BATCHES:
            st = static(bs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                body(bs, st)
            graphs[bs] = graph, st
    dist.barrier(coord.device_group)

    order = DECODE_BATCHES * 3
    random.Random(0).shuffle(order)  # the same order on every rank
    failures = []
    for step, bs in enumerate(order):
        graph, st = graphs[bs]
        row0, mine = _rows(bs, rank, world_size)
        gen = torch.Generator(device="cuda").manual_seed(1000 * step + rank)
        partials = [
            torch.randn(bs, hidden, device="cuda", generator=gen).bfloat16()
            * (rank + 1)
            / 4
            for _ in range(3)
        ]
        residual_all = _inputs(bs, hidden, step)[0]
        small = torch.randn(bs, 256, device="cuda", generator=gen).bfloat16()
        for dst, src in zip(st["p"], partials):
            dst.copy_(src)
        st["res"].copy_(residual_all[row0 : row0 + mine])
        st["small"].copy_(small)
        dist.barrier(coord.device_group)
        if rank == step % world_size:
            torch.cuda._sleep(delay)  # this rank starts the graph late
        graph.replay()
        torch.cuda.synchronize()

        p = [_gather(x) for x in [small] + partials]  # every rank's inputs
        side_ref = torch.stack([t.float() for t in p[0]]).sum(0)
        # the layer in fp32: attention site, MLP site, final site
        res, a = _reference(p[1], residual_all, w[0], w[1], None)
        res, y = _reference(p[2], res, w[2], w[3], scalar)
        final, _ = _reference(p[3], res, w[4], w[4], scalar)
        fused = [o.clone() for pair in st["outs"] for o in pair] + [st["final"].clone()]
        # gather before any assert: a rank that fails alone would hang its peers
        gathered = [_gather(t) for t in fused]
        try:
            torch.testing.assert_close(
                st["side"].float(), side_ref, rtol=1e-2, atol=1e-3
            )
            for (out, scales), ref in zip(st["outs"], (a, y)):
                deq = out.float() * scales.repeat_interleave(GROUP, dim=1)
                torch.testing.assert_close(deq, ref, rtol=7e-2, atol=4e-2)
                assert torch.all(scales > 0)
            torch.testing.assert_close(
                st["final"].float(), final.float(), rtol=2e-2, atol=5e-2
            )
            # the final site leaves its rows in the residual as well
            torch.testing.assert_close(st["res"], st["final"][row0 : row0 + mine])
            for t, others in zip(fused, gathered):
                assert all(torch.equal(o, t) for o in others), "ranks disagree"
        except AssertionError as e:
            failures.append(f"step {step} bs {bs}: {str(e)[:300]}")
    assert not failures, "\n".join(failures)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(2, 4, 8))
