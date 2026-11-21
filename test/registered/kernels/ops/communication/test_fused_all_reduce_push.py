"""Fused push all-reduce (1-shot, and 2-shot for MiniMax M3) against the unfused path: the rank-order fp32 sum
of the bf16 partials rounded to bf16, then the production unfused ops (fused_add_rmsnorm / rmsnorm / Beam's
fused_sandwich_residual_rmsnorm) or the documented roundings (finalize FMA chain, gated shared, HC = 4 post mix, MXFP8
quantization with swizzled ue8m0 scales, Beam's top-k sum). Every rank must match bit for bit.

Run with ``python <this file>``; it relaunches under torchrun on 4 GPUs.
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once
from sglang.kernels.ops.communication import fused_all_reduce
from sglang.kernels.ops.communication.all_reduce import Communicator, PushPlane
from sglang.kernels.ops.layernorm.beam_fused import fused_sandwich_residual_rmsnorm
from sglang.kernels.ops.layernorm.norm import fused_add_rmsnorm, rmsnorm
from sglang.srt.distributed.comm_cleanup import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=120, stage="extra-b", runner_config="4-gpu-b200")

EPS = 1e-6
# CTAs per row's cluster, as the callers launch them (the ops take it explicitly)
CLUSTER = {6144: 6, 5120: 5, 3584: 7, 2560: 2}
# MiniMax M3's 2-shot threshold at world size 4 (the ops take `two_shot` explicitly)
TWO_SHOT_MIN_ROWS = 44


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
        max_pull_size=0,
        max_pull_blocks=0,
        max_push_size=4 * 1024 * 1024,
        max_push_blocks=512,
    )
    assert not comm.disabled
    register_comm_cleanup(comm)
    return coord, comm


def _gather(t: torch.Tensor) -> list[torch.Tensor]:
    coord, _ = _init()
    out = [torch.empty_like(t) for _ in range(coord.world_size)]
    dist.all_gather(out, t.contiguous(), group=coord.device_group)
    return out


def _all_reduce(partial: torch.Tensor) -> torch.Tensor:
    parts = _gather(partial)
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.bfloat16()


def _fma(a, b, acc):
    """fp32 fma(a, b, acc): the product of a bf16 and an fp32 value is exact in float64."""
    return (a.double() * b.double() + acc.double()).float()


def _finalize(gemm2, idx, weights, top_k):
    rows = idx.view(-1, top_k)
    acc = torch.zeros(rows.shape[0], gemm2.shape[1], device="cuda")
    for k in range(top_k):
        x = gemm2[rows[:, k].clamp(min=0).long()]
        acc = torch.where(
            (rows[:, k] >= 0)[:, None], _fma(x, weights[:, k : k + 1].float(), acc), acc
        )
    return acc.bfloat16()


def _rand(*shape, scale=1.0, same=False):
    coord, _ = _init()
    g = torch.Generator(device="cuda").manual_seed(
        sum(shape) * 7 + (0 if same else coord.rank)
    )
    x = (torch.randn(shape, device="cuda", generator=g) * scale).bfloat16()
    # zeros exercise the +0 marker remap
    x[torch.rand(shape, device="cuda", generator=g) < 0.2] = 0.0
    return x


def _routing(rows, top_k, dtype):
    idx = torch.randperm(rows * top_k, device="cuda").int()
    idx[::5] = -1  # dropped slots contribute nothing, whatever their weight holds
    weights = torch.rand(rows, top_k, device="cuda").to(dtype)
    weights.view(-1)[idx == -1] = float("nan")
    return idx, weights


def _gamma(hidden):
    g = torch.Generator(device="cuda").manual_seed(3)
    return (1 + 0.1 * torch.randn(hidden, device="cuda", generator=g)).bfloat16()


def _check(outs, refs):
    """Equal to the reference up to the order of the norm's sum of squares (1 ulp, rarely); identical on every rank."""
    assert len(outs) == len(refs), (len(outs), len(refs))
    for out, ref in zip(outs, refs):
        torch.testing.assert_close(out.float(), ref.float(), rtol=8e-3, atol=1e-5)
        mismatch = (out.float() != ref.float()).float().mean().item()
        assert mismatch < 1e-3, (
            f"{mismatch:.2e} of the elements differ from the unfused reference"
        )
        raw = out.view(torch.uint8)
        for i, other in enumerate(_gather(raw)):
            assert torch.equal(other, raw), f"rank {i} differs"


def _residual_norm_ref(x, residual, gamma):
    out, res = x.clone(), residual.clone()
    fused_add_rmsnorm(out, res, gamma, EPS)
    return out, res


# MiniMax M3 rows on both sides of the 2-shot threshold; 256 rows run the 2-shot as two waves, 341 fill the 4 MB
# push slot. The forced cases: a 2-shot with ranks that own no row, a 1-shot above the threshold.
RESIDUAL_NORM_ROWS = [1, 7, 43, 44, 47, 256, 341]


@pytest.mark.parametrize(
    "rows,two_shot",
    [(r, r >= TWO_SHOT_MIN_ROWS) for r in RESIDUAL_NORM_ROWS]
    + [(3, True), (64, False)],
)
def test_all_reduce_residual_norm(rows, two_shot):
    _, comm = _init()
    x, residual, gamma = _rand(rows, 6144), _rand(rows, 6144, same=True), _gamma(6144)
    out = fused_all_reduce.push.all_reduce_residual_norm(
        comm.obj,
        x,
        residual.clone(),
        gamma,
        EPS,
        cluster_size=CLUSTER[6144],
        two_shot=two_shot,
    )
    _check(out, _residual_norm_ref(_all_reduce(x), residual, gamma))


@pytest.mark.parametrize("rows", [1, 43, 44, 256, 341])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shared", [False, True])
def test_moe_finalize_all_reduce_residual_norm(rows, weight_dtype, shared):
    _, comm = _init()
    top_k = 4
    gemm2 = _rand(rows * top_k, 6144, scale=0.1)
    idx, weights = _routing(rows, top_k, weight_dtype)
    sh = _rand(rows, 6144) if shared else None
    residual, gamma = _rand(rows, 6144, same=True), _gamma(6144)
    out = fused_all_reduce.push.moe_finalize_all_reduce_residual_norm(
        comm.obj,
        gemm2,
        idx,
        weights,
        sh,
        residual.clone(),
        gamma,
        EPS,
        cluster_size=CLUSTER[6144],
        two_shot=rows >= TWO_SHOT_MIN_ROWS,
    )
    partial = _finalize(gemm2, idx, weights, top_k)
    if shared:
        partial = (partial.float() + sh.float()).bfloat16()
    _check(out, _residual_norm_ref(_all_reduce(partial), residual, gamma))


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_moe_finalize_shared_gate_all_reduce(weight_dtype):
    _, comm = _init()
    rows, top_k, hidden = 1, 10, 2560
    gemm2 = _rand(rows * top_k, hidden, scale=0.1)
    idx, weights = _routing(rows, top_k, weight_dtype)
    shared, gate = _rand(rows, hidden), torch.rand(rows, device="cuda")
    out = fused_all_reduce.push.moe_finalize_shared_gate_all_reduce(
        comm.obj, gemm2, idx, weights, shared, gate, cluster_size=CLUSTER[hidden]
    )
    gated = (gate[:, None] * shared.float()).bfloat16()
    partial = (gated.float() + _finalize(gemm2, idx, weights, top_k).float()).bfloat16()
    _check((out,), (_all_reduce(partial),))


def _mhc_ref(x, residual, post, comb, pre, gamma):
    r = residual.float()
    streams = []
    for s in range(4):  # mhc_post_split_h's order
        acc = _fma(post[:, s : s + 1], x.float(), comb[:, 0, s : s + 1] * r[:, 0])
        for q in range(1, 4):
            acc = _fma(comb[:, q, s : s + 1], r[:, q], acc)
        streams.append(acc.bfloat16())
    refs = [x, torch.stack(streams, 1)]
    if pre is not None:
        collapsed = torch.zeros_like(x, dtype=torch.float32)
        for s in range(4):
            collapsed = _fma(streams[s].float(), pre[:, s : s + 1], collapsed)
        normalized = torch.empty_like(x)
        rmsnorm(collapsed.bfloat16(), gamma, normalized, EPS)
        refs.append(normalized)
    return refs


def _mxfp8_ref(y):
    """``mhc_quant_vec`` / FlashInfer ``mxfp8_quantize`` (swizzled layout) of ``y`` [T <= 128, H]: fp8 e4m3 codes and
    one ue8m0 scale per 32 values, positive rounding, in one 128-row tile whose padding rows are zero."""
    rows, hidden = y.shape
    v = y.float().view(rows, hidden // 32, 32)
    normalized = v.abs().amax(-1) * (1.0 / 448.0)
    bits = normalized.view(torch.int32)
    exponent, mantissa = (bits >> 23) & 255, bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = torch.where(normalized <= 0, 0, torch.clamp(exponent + bump.int(), max=254))
    inv = torch.where(sf == 0, 0, (254 - sf) << 23).int().view(torch.float32)
    q = (
        torch.clamp(v * inv[..., None], -448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(rows, hidden)
    )
    t = torch.arange(rows, device=y.device)[:, None]
    g = torch.arange(hidden // 32, device=y.device)[None, :]
    off = (g // 4) * 512 + ((t % 32) * 4 + (t // 32) % 4) * 4 + g % 4
    scales = torch.zeros(hidden // 32 * 128, dtype=torch.uint8, device=y.device)
    scales[off.flatten()] = sf.flatten().to(torch.uint8)
    return q, scales


def _check_quant(normalized, quantized, scales, ref_normalized):
    """The codes and every scale byte (live and padding) equal the quantization of the kernel's own normalized row;
    against the reference row, only the groups its 1-ulp norm differences touch may differ."""
    q, s = _mxfp8_ref(normalized)
    assert torch.equal(quantized.view(torch.uint8), q.view(torch.uint8)), (
        "fp8 codes differ"
    )
    assert torch.equal(scales, s), (
        f"{int((scales != s).sum())} scale bytes differ (padding: {int(scales[s == 0].ne(0).sum())})"
    )
    rows, hidden = normalized.shape
    q_ref, _ = _mxfp8_ref(ref_normalized)
    group = lambda d: d.view(rows, hidden // 32, 32).any(-1)
    moved = group(normalized.view(torch.int16) != ref_normalized.view(torch.int16))
    differs = group(q_ref.view(torch.uint8) != quantized.view(torch.uint8))
    assert not (differs & ~moved).any(), (
        f"{int((differs & ~moved).sum())} groups of codes differ from the unfused reference with equal inputs"
    )
    for i, other in enumerate(_gather(scales)):
        assert torch.equal(other, scales), f"rank {i} scales differ"
    raw = quantized.view(torch.uint8)
    for i, other in enumerate(_gather(raw)):
        assert torch.equal(other, raw), f"rank {i} codes differ"


@pytest.mark.parametrize("rows", [1, 64, 300])
@pytest.mark.parametrize("kind", ["attn", "post", "quant"])
def test_mhc(rows, kind):
    _, comm = _init()
    if kind == "quant" and rows > 128:
        pytest.skip("one 128-row scale tile")
    hidden, top_k, cluster = 5120, 6, CLUSTER[5120]
    residual = _rand(rows, 4, hidden, same=True)
    # the mixes are replicated across ranks
    g = torch.Generator(device="cuda").manual_seed(rows)
    post, comb, pre = (
        torch.rand(rows, *s, device="cuda", generator=g) for s in ((4,), (4, 4), (4,))
    )
    gamma = _gamma(hidden)
    if kind == "attn":
        x = _rand(rows, hidden)
        outs = fused_all_reduce.push.all_reduce_mhc_post_combine_norm(
            comm.obj, x, residual, post, comb, pre, gamma, EPS, cluster_size=cluster
        )
        _check(outs, _mhc_ref(_all_reduce(x), residual, post, comb, pre, gamma))
        return
    gemm2 = _rand(rows * top_k, hidden, scale=0.1)
    idx, weights = _routing(rows, top_k, torch.float32)
    shared = _rand(rows, hidden)
    partial = (
        _finalize(gemm2, idx, weights, top_k).float() + shared.float()
    ).bfloat16()
    if kind == "post":
        outs = fused_all_reduce.push.moe_finalize_all_reduce_mhc_post(
            comm.obj,
            gemm2,
            idx,
            weights,
            shared,
            residual,
            post,
            comb,
            cluster_size=cluster,
        )
        _check(outs, _mhc_ref(_all_reduce(partial), residual, post, comb, None, gamma))
    else:
        # a poisoned scale tile: a padding byte the kernel does not zero fails the check
        scales = torch.full(
            (hidden // 32 * 128,), 0xFF, dtype=torch.uint8, device="cuda"
        )
        args = (
            comm.obj,
            gemm2,
            idx,
            weights,
            shared,
            residual,
            post,
            comb,
            pre,
            gamma,
            EPS,
        )
        outs = (
            fused_all_reduce.push.moe_finalize_all_reduce_mhc_post_combine_norm_quant(
                *args, cluster_size=cluster, scales=scales
            )
        )
        assert len(outs) == 5 and outs[4] is scales
        refs = _mhc_ref(_all_reduce(partial), residual, post, comb, pre, gamma)
        _check(outs[:3], refs)
        _check_quant(outs[2], outs[3], outs[4], refs[2])
        # a freshly allocated tile gives the same bytes
        fresh = (
            fused_all_reduce.push.moe_finalize_all_reduce_mhc_post_combine_norm_quant(
                *args, cluster_size=cluster
            )
        )
        for a, b in zip(fresh, outs):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_empty_batch():
    """No rows: empty outputs and no launch (an idle batch skips the call on every rank)."""
    _, comm = _init()
    c, h = comm.obj, 6144
    z = torch.empty(0, h, dtype=torch.bfloat16, device="cuda")
    gamma = _gamma(h)
    out, res = fused_all_reduce.push.all_reduce_residual_norm(
        c, z, z.clone(), gamma, EPS, cluster_size=CLUSTER[h], two_shot=False
    )
    assert out.shape == (0, h) and res.shape == (0, h)
    idx = torch.empty(0, dtype=torch.int32, device="cuda")
    w = torch.empty(0, 4, device="cuda")
    out, _ = fused_all_reduce.push.moe_finalize_all_reduce_residual_norm(
        c,
        z,
        idx,
        w,
        None,
        z.clone(),
        gamma,
        EPS,
        cluster_size=CLUSTER[h],
        two_shot=False,
    )
    assert out.shape == (0, h)
    x = torch.empty(0, 5120, dtype=torch.bfloat16, device="cuda")
    res4 = torch.empty(0, 4, 5120, dtype=torch.bfloat16, device="cuda")
    mix = torch.empty(0, 4, device="cuda")
    mix4 = torch.empty(0, 4, 4, device="cuda")
    outs = fused_all_reduce.push.all_reduce_mhc_post_combine_norm(
        c, x, res4, mix, mix4, mix, _gamma(5120), EPS, cluster_size=CLUSTER[5120]
    )
    assert [t.shape[0] for t in outs] == [0, 0, 0]
    w = torch.empty(0, 6, device="cuda")
    outs = fused_all_reduce.push.moe_finalize_all_reduce_mhc_post(
        c, x, idx, w, None, res4, mix, mix4, cluster_size=CLUSTER[5120]
    )
    assert [t.shape[0] for t in outs] == [0, 0]
    outs = fused_all_reduce.push.moe_finalize_all_reduce_mhc_post_combine_norm(
        c,
        x,
        idx,
        w,
        None,
        res4,
        mix,
        mix4,
        mix,
        _gamma(5120),
        EPS,
        cluster_size=CLUSTER[5120],
    )
    assert [t.shape[0] for t in outs] == [0, 0, 0]


@pytest.mark.parametrize("rows,norm_rows", [(1, 1), (73, 73), (24, 8)])
def test_all_reduce_norm_prefix(rows, norm_rows):
    _, comm = _init()
    x, gamma = _rand(rows, 3584), _gamma(3584)
    ref = _all_reduce(x)
    normed = ref.clone()
    rmsnorm(ref[:norm_rows].contiguous(), gamma, normed[:norm_rows], EPS)
    out = fused_all_reduce.push.all_reduce_norm_prefix(
        comm.obj,
        x.clone(),
        gamma,
        EPS,
        num_norm_rows=norm_rows,
        cluster_size=CLUSTER[3584],
    )
    _check((out,), (normed,))


@pytest.mark.parametrize("rows", [1, 73])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_moe_finalize_all_reduce_norm(rows, weight_dtype):
    _, comm = _init()
    top_k, gamma = 16, _gamma(3584)
    gemm2 = _rand(rows * top_k, 3584, scale=0.1)
    idx, weights = _routing(rows, top_k, weight_dtype)
    out = torch.empty(rows, 3584, dtype=torch.bfloat16, device="cuda")
    fused_all_reduce.push.moe_finalize_all_reduce_norm(
        comm.obj, out, gemm2, idx, weights, gamma, EPS, cluster_size=CLUSTER[3584]
    )
    normed = torch.empty_like(out)
    rmsnorm(_all_reduce(_finalize(gemm2, idx, weights, top_k)), gamma, normed, EPS)
    _check((out,), (normed,))


@cache_once
def _beam_planes() -> dict:
    """The communicator's plane (multicast where it has a multicast VA) and a unicast plane over the same workspaces
    and epoch counter."""
    _, comm = _init()
    lease = comm._lease
    unicast = Communicator(
        push=PushPlane(
            lease.rank,
            lease.world_size,
            workspaces=lease.data.push_slots,
            counter=lease.control.epoch,
            default_blocks=comm.config.num_push_blocks,
        ),
        pull=comm.obj.pull,
    )
    return {"unicast": unicast, "multicast": comm.obj if comm.has_multicast else None}


# Beam 1.0 (hidden 5120): both cluster sizes (5 is world size 8's up to 16 rows, forced here),
# up to the 64-row push slot
@pytest.mark.parametrize("rows", [1, 16, 64])
@pytest.mark.parametrize("top_k", [1, 8])
@pytest.mark.parametrize("next_norm", [False, True])
@pytest.mark.parametrize("cluster_size", [1, 5])
@pytest.mark.parametrize("transport", ["unicast", "multicast"])
def test_sandwich_norm(rows, top_k, next_norm, cluster_size, transport):
    plane = _beam_planes()[transport]
    if plane is None:
        pytest.skip("no multicast VA on this plane")
    hidden, scales, routed_scale = 5120, (0.13, 0.97), 2.5
    branch = _rand(rows, top_k, hidden) if top_k > 1 else _rand(rows, hidden)
    residual, post, next_weight = _rand(rows, hidden, same=True), _gamma(hidden), None
    if next_norm:
        next_weight = (2 - _gamma(hidden).float()).bfloat16()
    args = (plane, branch, residual, post, next_weight, EPS, scales)
    if top_k > 1:
        combined, normalized = fused_all_reduce.push.topk_sum_all_reduce_sandwich_norm(
            *args, routed_scale=routed_scale, cluster_size=cluster_size
        )
    else:
        combined, normalized = fused_all_reduce.push.all_reduce_sandwich_norm(
            *args, cluster_size=cluster_size
        )
    partial = branch
    if top_k > 1:  # fp32 sum in expert order, scaled, rounded once
        acc = torch.zeros(rows, hidden, device="cuda")
        for k in range(top_k):
            acc += branch[:, k].float()
        partial = (acc * routed_scale).bfloat16()
    ref_normalized, ref_combined = fused_sandwich_residual_rmsnorm(
        _all_reduce(partial), residual, post, next_weight, EPS, *scales
    )
    outs, refs = (combined,), (ref_combined,)
    if next_norm:
        outs, refs = (combined, normalized), (ref_combined, ref_normalized)
    else:
        assert normalized is None
    _check(outs, refs)


if __name__ == "__main__":
    multigpu_pytest_main(__name__, __file__, num_gpus=(4,))
