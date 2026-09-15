"""Fused deferred-MoE finalize + push all-reduce (``moe_finalize_all_reduce``)
against a torch reference, for bf16 and fp32 routing weights.

The routed-MoE runners hand this kernel FlashInfer's ``do_finalize=False``
triple. With unpacked ``(topk_ids, topk_weights)`` routing the weights arrive
in fp32, with packed routing in bf16; both must reproduce the unfused path's
numerics (fp32 accumulation, bf16 rounding at the routed combine, after the
``+ shared`` add and at the all-reduce output).
"""

from __future__ import annotations

import atexit
import logging
import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.jit.utils import cache_once, get_ci_test_range
from sglang.kernels.ops.communication import all_reduce_fusion
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN = 5120  # DeepSeek-V4 hidden size, the width the fused path is used at
TOP_K = 6
MB = 1024 * 1024
NUM_TOKENS = get_ci_test_range([1, 2, 8, 64], [1, 8, 64])
WEIGHT_DTYPES = [torch.bfloat16, torch.float32]


def _precompile(num_gpus):
    for ws in num_gpus:
        for dt in WEIGHT_DTYPES:
            all_reduce_fusion.compile_moe_finalize_all_reduce(
                ws, HIDDEN, TOP_K, weight_dtype=dt
            )


@cache_once
def _init_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coord = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    atexit.register(dist.destroy_process_group)
    logging.disable(logging.INFO)
    torch.cuda.set_stream(torch.cuda.Stream())
    return coord.cpu_group


@cache_once
def _init_nccl_group():
    _init_world()
    local_rank = int(os.environ["LOCAL_RANK"])
    group = dist.new_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    assert isinstance(group, dist.ProcessGroup)
    return group


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


@cache_once
def _init_comm() -> CustomAllReduceV2:
    cpu_group = _init_world()
    comm = CustomAllReduceV2(
        cpu_group, _device(), max_pull_size=1 * MB, max_push_size=2 * MB
    )
    if comm.disabled:
        raise RuntimeError("moe_finalize_all_reduce requires CustomAllReduceV2")
    all_reduce_fusion.register_comm(comm.obj)
    register_comm_cleanup(comm)
    return comm


def _make_inputs(num_tokens: int, weight_dtype: torch.dtype, exact: bool, seed: int):
    """Per-rank permuted GEMM2 rows, routing slots and shared-expert output.

    ``exact`` keeps every value a small dyadic number so the fp32 sums and the
    bf16 roundings are lossless and the kernel can be checked bit-exactly.
    """
    g = torch.Generator().manual_seed(seed * 7919 + dist.get_rank())
    num_slots = num_tokens * TOP_K
    num_rows = num_slots + 8  # a few padded rows no slot points at
    if exact:
        gemm2 = torch.randint(-8, 9, (num_rows, HIDDEN), generator=g).to(torch.bfloat16)
        weights = torch.randint(0, 8, (num_tokens, TOP_K), generator=g) / 8.0
        shared = torch.randint(-8, 9, (num_tokens, HIDDEN), generator=g).to(
            torch.bfloat16
        )
    else:
        gemm2 = torch.randn(num_rows, HIDDEN, generator=g).to(torch.bfloat16)
        weights = torch.rand(num_tokens, TOP_K, generator=g) * 1.5
        shared = torch.randn(num_tokens, HIDDEN, generator=g).to(torch.bfloat16)
    idx = torch.randperm(num_rows, generator=g)[:num_slots].to(torch.int32)
    # EP: slots routed to an expert another rank owns carry -1 and contribute nothing.
    idx[torch.rand(num_slots, generator=g) < 0.1] = -1
    dev = _device()
    return gemm2.to(dev), idx.to(dev), weights.to(weight_dtype).to(dev), shared.to(dev)


def _local_ref(gemm2, idx, weights, shared):
    """Unfused numerics: fp32 accumulate, bf16 at the combine and after + shared."""
    rows = idx.view(weights.shape).long()
    valid = (rows >= 0).float()
    gathered = gemm2[rows.clamp(min=0)].float()  # [T, top_k, H]
    w = weights.float() * valid
    routed = (gathered * w.unsqueeze(-1)).sum(dim=1).to(torch.bfloat16)
    if shared is None:
        return routed
    return (routed.float() + shared.float()).to(torch.bfloat16)


def _all_reduce_ref(local: torch.Tensor) -> torch.Tensor:
    """fp32-accumulating bf16 all-reduce in rank order."""
    group = _init_nccl_group()
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, local, group=group)
    acc = torch.zeros(local.shape, dtype=torch.float32, device=local.device)
    for x in gathered:
        acc += x.float()
    return acc.to(torch.bfloat16)


def _fused(comm, gemm2, idx, weights, shared):
    out = all_reduce_fusion.moe_finalize_all_reduce(
        gemm2,
        idx,
        weights,
        TOP_K,
        shared,
        world_size=comm.world_size,
        hidden_dim=HIDDEN,
    )
    torch.cuda.synchronize()
    return out


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("weight_dtype", WEIGHT_DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("use_shared", [False, True])
@torch.inference_mode()
def test_moe_finalize_all_reduce_exact(num_tokens, weight_dtype, use_shared):
    comm = _init_comm()
    gemm2, idx, weights, shared = _make_inputs(
        num_tokens, weight_dtype, exact=True, seed=num_tokens
    )
    shared = shared if use_shared else None
    ref = _all_reduce_ref(_local_ref(gemm2, idx, weights, shared))
    out = _fused(comm, gemm2, idx, weights, shared)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("use_shared", [False, True])
@torch.inference_mode()
def test_moe_finalize_all_reduce_fp32_weights(num_tokens, use_shared):
    """fp32 weights are consumed at fp32: the kernel tracks an fp32 reference
    within bf16 output tolerance, and is not the bf16-rounded-weight result."""
    comm = _init_comm()
    gemm2, idx, weights, shared = _make_inputs(
        num_tokens, torch.float32, exact=False, seed=100 + num_tokens
    )
    shared = shared if use_shared else None
    ref = _all_reduce_ref(_local_ref(gemm2, idx, weights, shared))
    out = _fused(comm, gemm2, idx, weights, shared)
    # The kernel's sequential fmaf and torch's sum round differently in fp32,
    # so a rank-local combine (magnitude up to ~16) can flip one bf16 ulp
    # before the cross-rank sum: allow that, nothing more.
    torch.testing.assert_close(out, ref, atol=0.125, rtol=0.01)
    out_bf16_weights = _fused(comm, gemm2, idx, weights.to(torch.bfloat16), shared)
    assert not torch.equal(out, out_bf16_weights)


@pytest.mark.parametrize("num_tokens", [1, 5, 6, 8])
@pytest.mark.parametrize("weight_dtype", WEIGHT_DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("epilogue", ["post", "norm", "quant"])
@pytest.mark.parametrize("use_shared", [False, True])
@pytest.mark.parametrize("seed", [0, 13])
@torch.inference_mode()
def test_mhc_epilogue_graph(num_tokens, weight_dtype, epilogue, use_shared, seed):
    from sglang.kernels.ops.communication import all_reduce_mhc
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    comm = _init_comm()
    gemm2, idx, weights, shared = _make_inputs(
        num_tokens, weight_dtype, exact=False, seed=31
    )
    shared = shared if use_shared else None
    top_k = TOP_K
    # The attention epilogue uses the same reduction with one contribution.
    if epilogue == "norm" and not use_shared:
        top_k = 1
        gemm2 = gemm2[:num_tokens].contiguous()
        idx = torch.arange(num_tokens, device=_device(), dtype=torch.int32)
        weights = torch.ones(num_tokens, 1, device=_device(), dtype=weight_dtype)
    residual = torch.randn(
        num_tokens, 4, HIDDEN, device=_device(), dtype=torch.bfloat16
    )
    post = torch.randn(num_tokens, 4, device=_device())
    comb = torch.randn(num_tokens, 4, 4, device=_device())
    pre = torch.rand(num_tokens, 4, device=_device())
    nw = torch.randn(HIDDEN, device=_device(), dtype=torch.bfloat16)
    torch.manual_seed(seed * 7919 + dist.get_rank())
    residual.normal_()
    post.normal_()
    comb.normal_()
    pre.uniform_()
    nw.normal_()
    kernel = {
        "post": all_reduce_mhc.moe_finalize_all_reduce_mhc,
        "norm": all_reduce_mhc.moe_finalize_all_reduce_mhc_norm,
        "quant": all_reduce_mhc.moe_finalize_all_reduce_mhc_quant,
    }[epilogue]

    def chain():
        old = all_reduce_fusion.moe_finalize_all_reduce(
            gemm2,
            idx,
            weights,
            top_k,
            shared,
            world_size=comm.world_size,
            hidden_dim=HIDDEN,
        )
        ref_post = mhc_post_split_h(old, residual, post, comb)
        ref_norm = hc_combine_norm(ref_post.flatten(1), pre, nw, 1e-6)
        args = [gemm2, idx, weights, top_k, shared, residual, post, comb]
        if epilogue != "post":
            args.extend((pre, nw, 1e-6))
        outputs = kernel(*args, world_size=comm.world_size)
        # Exercise the shared counters between generic push and row-cluster AR.
        comm.custom_all_reduce(outputs[0])
        return old, ref_post, ref_norm, outputs

    chain()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with comm.capture(), torch.cuda.graph(graph):
        old, ref_post, ref_norm, outputs = chain()

    def check_outputs():
        reduced, actual_post = outputs[:2]
        assert torch.equal(old.view(torch.int16), reduced.view(torch.int16))
        assert torch.equal(ref_post.view(torch.int16), actual_post.view(torch.int16))
        if epilogue == "post":
            return
        actual_norm = outputs[2]
        # The cluster and Triton RMS reductions have different addition orders.
        torch.testing.assert_close(actual_norm, ref_norm, rtol=0.008, atol=0.0001)
        # Different reduction trees can land on opposite sides of BF16 ties.
        # Bound each value, rather than a data-dependent bit-identical fraction.
        ulp = (
            actual_norm.view(torch.int16).int() - ref_norm.view(torch.int16).int()
        ).abs()
        ulp.masked_fill_(actual_norm == ref_norm, 0)  # Treat signed zeros equally.
        assert ulp.max() <= 1
        if epilogue == "quant":
            q, sf = outputs[3:]
            for backend in ("cuda", "cute-dsl"):
                ref_q, ref_sf = flashinfer_mxfp8_quantize(
                    actual_norm, True, 32, backend
                )
                assert torch.equal(
                    q.view(torch.uint8).flatten(), ref_q.view(torch.uint8).flatten()
                )
                assert torch.equal(sf, ref_sf.flatten())

    for replay, magnitude in enumerate((1e-3, 1.0, 1e3, 0.0, 1.0, 0.0, 1e-2)):
        residual.normal_().mul_(magnitude)
        gemm2.normal_().mul_(magnitude)
        if epilogue == "quant":
            outputs[4].fill_(0xAB)  # Also require rewriting the padded scale rows.
        if dist.get_rank() == replay % dist.get_world_size():
            torch.cuda._sleep(100000)
        graph.replay()
        torch.cuda.synchronize()
        error = None
        try:
            check_outputs()
        except AssertionError as exc:
            error = f"rank={dist.get_rank()}, replay={replay}, scale={magnitude}: {exc}"
        errors = [None] * dist.get_world_size()
        # A failed assertion must stop all ranks before the next collective.
        dist.all_gather_object(errors, error)
        assert not any(errors), "\n".join(e for e in errors if e)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(4,),
        pre_launch_fn=_precompile,
    )
