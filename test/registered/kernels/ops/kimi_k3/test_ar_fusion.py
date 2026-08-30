"""Correctness test for the K3 MNNVL fused all-reduce (ar_fusion) kernels.

Compares the 1shot multicast-push and the in-place low-SM NVLS 2shot pull
(with and without the fused residual) against
NCCL, bit-exact on small-int bf16 inputs; the fused-RMSNorm pull against a
torch reference; the pull tuning knobs (num_blocks, unroll) on sizes whose
shard split is uneven; plus a CUDA-graph capture/replay pass and a mixed
stress loop exercising the push phase double-buffering and the pull
semaphore window cycling.

Usage::

    python test/registered/jit/kimi_k3/test_ar_fusion.py   # relaunches under torchrun (8 GPUs)
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
from sglang.kernels.ops.communication.mp import register_comm_cleanup
from sglang.kernels.ops.kimi_k3 import all_reduce
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.utils import multigpu_pytest_main

register_cuda_ci(
    est_time=240,
    stage="extra-b",
    runner_config="8-gpu-h200",
)

H = 7168  # Kimi-K3 hidden size; the kernels are tuned/used at multiples of it
NORM_DIM = 3584  # latent width; the norm buffer is [N, NORM_DIM] + [N, 2*NORM_DIM]
MB = 1024 * 1024

PUSH_BS = [1, 2, 8, 32, 128]
PULL_BS = [1, 8, 64, 1024, 4096]
PUSH_BS = get_ci_test_range(PUSH_BS, [1, 32, 128])
PULL_BS = get_ci_test_range(PULL_BS, [1, 64, 4096])


def _precompile(num_gpus):
    for ws in num_gpus:
        all_reduce._jit_module(ws)


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


def _symm_alloc_mc(shape, dtype) -> tuple[torch.Tensor, int]:
    import torch.distributed._symmetric_memory as torch_symm_mem

    cpu_group = _init_world()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    pool = torch_symm_mem.get_mem_pool(device)
    with torch.cuda.use_mem_pool(pool):
        buf = torch.empty(shape, dtype=dtype, device=device)
    hdl = torch_symm_mem.rendezvous(buf, cpu_group.group_name)
    assert hdl.multicast_ptr != 0
    mc = hdl.multicast_ptr + (buf.data_ptr() - hdl.buffer_ptrs[rank])
    return buf, mc


@cache_once
def _init_comm() -> CustomAllReduceV2:
    cpu_group = _init_world()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    comm = CustomAllReduceV2(
        cpu_group, device, max_pull_size=1 * MB, max_push_size=2 * MB
    )
    if comm.disabled or not comm.has_multicast:
        raise RuntimeError("ar_fusion requires CustomAllReduceV2 with multicast")
    all_reduce.register_comm(comm.obj)
    register_comm_cleanup(comm)
    return comm


@cache_once
def _init_pool_buf() -> tuple[torch.Tensor, int]:
    # 1.5x headroom: the norm tests view the buffer as [N, 3584 + 7168]
    return _symm_alloc_mc((max(PULL_BS) * H * 3 // 2,), torch.bfloat16)


def _device() -> torch.device:
    return torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")


def _int_input(n: int, seed: int, per_rank: bool) -> torch.Tensor:
    # small ints are exact in bf16 even after an fp32-accumulated 8-way sum,
    # so the comparison against NCCL is bit-exact
    rank = dist.get_rank() if per_rank else 0
    g = torch.Generator().manual_seed(seed * 1009 + rank)
    return torch.randint(0, 16, (n,), dtype=torch.bfloat16, generator=g).to(_device())


def _nccl_ref(x: torch.Tensor, residual):
    ref = x.clone()
    dist.all_reduce(ref, group=_init_nccl_group())
    return ref if residual is None else ref + residual


def _norm_ref(x_reduced: torch.Tensor, num_norm_rows: int, weight, eps: float):
    """allreduce result -> RMSNorm over the first num_norm_rows rows of the
    [numel / NORM_DIM, NORM_DIM] row view, in fp32 like the kernels."""
    out = x_reduced.clone()
    normed_part = out[: num_norm_rows * NORM_DIM].view(num_norm_rows, NORM_DIM).float()
    factor = torch.rsqrt(normed_part.pow(2).mean(-1, keepdim=True) + eps)
    normed = (normed_part * factor * weight.float()).to(torch.bfloat16)
    out[: num_norm_rows * NORM_DIM] = normed.view(-1)
    return out


def _assert_norm_close(x: torch.Tensor, ref: torch.Tensor, num_norm_rows: int):
    # the non-normed tail is a plain allreduce: bit-exact; the normed prefix
    # gets the fp32 norm epilogue: bf16 tolerances
    torch.testing.assert_close(
        x[num_norm_rows * NORM_DIM :], ref[num_norm_rows * NORM_DIM :], atol=0, rtol=0
    )
    torch.testing.assert_close(
        x[: num_norm_rows * NORM_DIM], ref[: num_norm_rows * NORM_DIM]
    )


@pytest.mark.parametrize("bs", PUSH_BS)
@pytest.mark.parametrize("use_residual", [False, True])
@torch.inference_mode()
def test_ar_fusion_push(bs: int, use_residual: bool):
    comm = _init_comm()
    world = comm.world_size
    n = bs * H
    x = _int_input(n, bs, per_rank=True)
    residual = _int_input(n, bs + 7, per_rank=False) if use_residual else None
    ref = _nccl_ref(x, residual)
    all_reduce.all_reduce_push_res(world, x, residual)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, ref, atol=0, rtol=0)


@pytest.mark.parametrize("bs", PULL_BS)
@pytest.mark.parametrize("use_residual", [False, True])
@torch.inference_mode()
def test_ar_fusion_pull_2shot(bs: int, use_residual: bool):
    comm = _init_comm()
    world = comm.world_size
    buf, mc = _init_pool_buf()
    n = bs * H
    x = buf[:n]
    x.copy_(_int_input(n, bs + 13, per_rank=True))
    residual = _int_input(n, bs + 17, per_rank=False) if use_residual else None
    ref = _nccl_ref(x, residual)
    all_reduce.all_reduce_pull_res(world, x, residual, input_mc_ptr=mc)
    torch.cuda.synchronize()
    torch.testing.assert_close(x, ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_blocks", [1, 2, 4, 8])
@pytest.mark.parametrize("unroll", [4, 8])
@torch.inference_mode()
def test_ar_fusion_pull_tuning_grid(num_blocks: int, unroll: int):
    """Every (num_blocks, unroll) combination must agree with NCCL on a size
    whose 16B-vector count is not divisible by the world size (uneven shards)
    and whose per-thread range leaves an unrolled-loop tail."""
    _init_comm()
    world = dist.get_world_size()
    buf, mc = _init_pool_buf()
    n = (3 * H + 7) * 8  # 21511 vecs: % 8 ranks != 0, small vs blocks*512*unroll
    x = buf[:n]
    x.copy_(_int_input(n, num_blocks * 10 + unroll, per_rank=True))
    ref = _nccl_ref(x, None)
    all_reduce.all_reduce_pull_res(
        world, x, None, input_mc_ptr=mc, num_blocks=num_blocks, unroll=unroll
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(x, ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", PULL_BS)
@pytest.mark.parametrize("rows_per_token", [3, 1])  # [N|2N] MoE buf / latent-only
@torch.inference_mode()
def test_ar_fusion_pull_norm(num_tokens: int, rows_per_token: int):
    _init_comm()
    world = dist.get_world_size()
    buf, mc = _init_pool_buf()
    n = num_tokens * rows_per_token * NORM_DIM
    x = buf[:n]
    x.copy_(_int_input(n, num_tokens + 23 + rows_per_token, per_rank=True))
    weight = _int_input(NORM_DIM, 29, per_rank=False) + 1  # small positive ints
    ref = _norm_ref(_nccl_ref(x, None), num_tokens, weight, eps=1e-6)
    all_reduce.all_reduce_pull_norm(
        world, x, weight, 1e-6, num_norm_rows=num_tokens, input_mc_ptr=mc
    )
    torch.cuda.synchronize()
    _assert_norm_close(x, ref, num_tokens)


@pytest.mark.parametrize("num_tokens", [1, 8, 24])
@pytest.mark.parametrize("rows_per_token", [3, 1])
@torch.inference_mode()
def test_ar_fusion_push_norm(num_tokens: int, rows_per_token: int):
    """The push-side norm (small-message regime of the serving dispatch) with
    an explicit num_norm_rows, on both the MoE-buffer and latent-only row
    layouts."""
    comm = _init_comm()
    world = comm.world_size
    n = num_tokens * rows_per_token * NORM_DIM
    x = _int_input(n, num_tokens + 41 + rows_per_token, per_rank=True)
    weight = _int_input(NORM_DIM, 43, per_rank=False) + 1
    ref = _norm_ref(_nccl_ref(x, None), num_tokens, weight, eps=1e-6)
    all_reduce.all_reduce_push_norm(world, x, weight, 1e-6, num_norm_rows=num_tokens)
    torch.cuda.synchronize()
    _assert_norm_close(x, ref, num_tokens)


FIN_TOPK = 16


def _build_permuted_layout(num_tokens: int, seed: int):
    """trtllm-gen permuted gemm2 layout (rows grouped by expert, per-expert
    tile padding). Deterministic on CPU: idx/weights are identical on every
    rank (TP semantics — same routing), gemm2 values are per-rank."""
    num_experts, tile = 896, 8
    gen = torch.Generator(device="cpu").manual_seed(seed)
    topk_ids = torch.stack(
        [
            torch.randperm(num_experts, generator=gen)[:FIN_TOPK]
            for _ in range(num_tokens)
        ]
    )
    counts = torch.bincount(topk_ids.flatten(), minlength=num_experts)
    padded = (counts + tile - 1) // tile * tile
    bases = torch.cumsum(padded, 0) - padded
    fill = torch.zeros(num_experts, dtype=torch.long)
    idx = torch.empty(num_tokens * FIN_TOPK, dtype=torch.int32)
    for i, e in enumerate(topk_ids.flatten().tolist()):
        idx[i] = bases[e] + fill[e]
        fill[e] += 1
    weights = torch.rand(num_tokens, FIN_TOPK, generator=gen).to(torch.bfloat16)
    num_rows = int(padded.sum())
    g = torch.Generator(device="cpu").manual_seed(seed * 31 + dist.get_rank())
    gemm2 = (torch.randn(num_rows, NORM_DIM, generator=g) * 2).to(torch.bfloat16)
    dev = _device()
    return gemm2.to(dev), idx.to(dev), weights.to(dev)


def _finalize_norm_ref(gemm2, idx, weights, norm_w, eps: float) -> torch.Tensor:
    """Replicates the fused kernel numerics: fp32 ascending-k local finalize
    cast to bf16 (the staged push value), rank-ordered fp32 cross-rank sum,
    fp32 RMSNorm. Only the rsqrt may differ from the kernel by ulps."""
    num_tokens = weights.shape[0]
    idx2 = idx.view(num_tokens, FIN_TOPK).long()
    acc = torch.zeros(num_tokens, NORM_DIM, dtype=torch.float32, device=gemm2.device)
    for k in range(FIN_TOPK):
        acc += weights[:, k, None].float() * gemm2[idx2[:, k]].float()
    local = acc.to(torch.bfloat16)
    world = dist.get_world_size()
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local, group=_init_nccl_group())
    total = torch.zeros_like(acc)
    for r in range(world):
        total += gathered[r].float()
    factor = torch.rsqrt(total.square().mean(dim=-1, keepdim=True) + eps)
    return (total * factor * norm_w.float()).to(torch.bfloat16)


@pytest.mark.parametrize("bs", PUSH_BS)
@torch.inference_mode()
def test_ar_fusion_finalize_push_norm(bs: int):
    comm = _init_comm()
    world = comm.world_size
    eps = 1e-6
    gemm2, idx, weights = _build_permuted_layout(bs, seed=bs + 23)
    g = torch.Generator(device="cpu").manual_seed(77)
    norm_w = (torch.rand(NORM_DIM, generator=g) + 0.5).to(torch.bfloat16).to(_device())
    ref = _finalize_norm_ref(gemm2, idx, weights, norm_w, eps)
    out = torch.empty(bs, NORM_DIM, dtype=torch.bfloat16, device=_device())
    all_reduce.finalize_all_reduce_push_norm(
        world, out, gemm2, idx, weights, norm_w, eps
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_ar_fusion_finalize_push_norm_stress():
    """Back-to-back fused calls interleaved with plain pushes exercise the
    shared push-workspace phase double-buffering across kernel variants."""
    comm = _init_comm()
    world = comm.world_size
    eps = 1e-5
    g = torch.Generator(device="cpu").manual_seed(78)
    norm_w = (torch.rand(NORM_DIM, generator=g) + 0.5).to(torch.bfloat16).to(_device())
    for it in range(12):
        bs = (1, 8, 32)[it % 3]
        gemm2, idx, weights = _build_permuted_layout(bs, seed=9000 + it)
        ref = _finalize_norm_ref(gemm2, idx, weights, norm_w, eps)
        out = torch.empty(bs, NORM_DIM, dtype=torch.bfloat16, device=_device())
        all_reduce.finalize_all_reduce_push_norm(
            world, out, gemm2, idx, weights, norm_w, eps
        )
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        x = _int_input(bs * H, 8000 + it, per_rank=True)
        ref2 = _nccl_ref(x, None)
        all_reduce.all_reduce_push_res(world, x, None)
        torch.testing.assert_close(x, ref2, atol=0, rtol=0)


@pytest.mark.parametrize("num_blocks", [1, 4, 16])
@pytest.mark.parametrize("unroll", [4, 8])
@torch.inference_mode()
def test_ar_fusion_pull_norm_tuning_grid(num_blocks: int, unroll: int):
    """Every (num_blocks, unroll) combination must agree on a token count
    whose row count is not divisible by the world size (uneven row shards)
    and not by unroll (partial last row group per block)."""
    _init_comm()
    world = dist.get_world_size()
    buf, mc = _init_pool_buf()
    num_tokens = 13  # 39 rows: % 8 ranks != 0, per-rank rows < num_blocks*unroll
    n = num_tokens * 3 * NORM_DIM
    x = buf[:n]
    x.copy_(_int_input(n, 500 + num_blocks * 10 + unroll, per_rank=True))
    weight = _int_input(NORM_DIM, 31, per_rank=False) + 1
    ref = _norm_ref(_nccl_ref(x, None), num_tokens, weight, eps=1e-6)
    all_reduce.all_reduce_pull_norm(
        world,
        x,
        weight,
        1e-6,
        num_norm_rows=num_tokens,
        input_mc_ptr=mc,
        num_blocks=num_blocks,
        unroll=unroll,
    )
    torch.cuda.synchronize()
    _assert_norm_close(x, ref, num_tokens)


@torch.inference_mode()
def test_ar_fusion_stress_mixed():
    """Back-to-back mixed calls exercise the push phase double-buffering and
    the pull semaphore window cycling (with varying grids)."""
    comm = _init_comm()
    world = comm.world_size
    buf, mc = _init_pool_buf()
    for it in range(32):
        n = (1, 8, 64)[it % 3] * H
        num_blocks = (1, 2, 4, 8)[it % 4]
        x = _int_input(n, 3000 + it, per_rank=True)
        ref = _nccl_ref(x, None)
        all_reduce.all_reduce_push_res(world, x, None)
        torch.testing.assert_close(x, ref, atol=0, rtol=0)
        y = buf[:n]
        y.copy_(_int_input(n, 4000 + it, per_rank=True))
        ref2 = _nccl_ref(y, None)
        all_reduce.all_reduce_pull_res(
            world, y, None, input_mc_ptr=mc, num_blocks=num_blocks
        )
        torch.testing.assert_close(y, ref2, atol=0, rtol=0)


@torch.inference_mode()
def test_ar_fusion_graph_capture():
    comm = _init_comm()
    world = comm.world_size
    buf, mc = _init_pool_buf()
    cpu_group = _init_world()
    n = 64 * H
    gres = _int_input(n, 99, per_rank=False)
    gx = torch.zeros(n, dtype=torch.bfloat16, device=_device())
    # two disjoint regions of the symm buffer, one per captured pull kernel
    gy, mc_y = buf[:n], mc
    gz, mc_z = buf[n : 2 * n], mc + n * buf.element_size()

    def _run_all():
        all_reduce.all_reduce_push_res(world, gx, gres)
        all_reduce.all_reduce_pull_res(world, gy, gres, input_mc_ptr=mc_y)
        all_reduce.all_reduce_pull_res(world, gz, gres, input_mc_ptr=mc_z)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _run_all()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_all()

    for it in range(4):
        vx = _int_input(n, 5000 + it, per_rank=True)
        vy = _int_input(n, 6000 + it, per_rank=True)
        vz = _int_input(n, 7000 + it, per_rank=True)
        ref_x = _nccl_ref(vx, gres)
        ref_y = _nccl_ref(vy, gres)
        ref_z = _nccl_ref(vz, gres)
        gx.copy_(vx)
        gy.copy_(vy)
        gz.copy_(vz)
        dist.barrier(group=cpu_group)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(gx, ref_x, atol=0, rtol=0)
        torch.testing.assert_close(gy, ref_y, atol=0, rtol=0)
        torch.testing.assert_close(gz, ref_z, atol=0, rtol=0)


if __name__ == "__main__":
    multigpu_pytest_main(
        __name__,
        __file__,
        num_gpus=(8,),
        pre_launch_fn=_precompile,
    )
