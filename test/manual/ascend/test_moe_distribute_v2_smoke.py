"""Standalone smoke test for the torch_npu MC2 MoE distribute ops.

It validates the contract the ``ascend_ops`` dispatcher relies on:

* the 7 dispatch return values come back in the expected order/dtype,
* ``combine_v2`` restores the input when the expert output is passed through
  unchanged (i.e. the op applies the top-k weights itself),
* the top-k weights are applied once (unnormalised weights scale the result),
* ``ep_recv_counts`` from dispatch can be reused as ``ep_send_counts``,
* ``x_active_mask`` padding only routes the active tokens,
* per-token int8 quantisation (``quant_mode=2``) round-trips within int8 error.

Run on one A3 node with the target EP size, e.g.:

    torchrun --nproc_per_node=16 test/manual/ascend/test_moe_distribute_v2_smoke.py

To also exercise the padded-batch path:

    torchrun --nproc_per_node=16 test/manual/ascend/test_moe_distribute_v2_smoke.py \
        --bs 5 --pad-bs 8
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.distributed import distributed_c10d


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _hccl_comm_name(group, rank: int) -> str:
    backend = group._get_backend(torch.device("npu"))
    if hasattr(backend, "get_hccl_comm_name"):
        return backend.get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _build_batch(seed, bs, pad_bs, hidden, num_experts, top_k, weight_sum, device):
    torch.manual_seed(seed)
    x = torch.randn(bs, hidden)
    logits = torch.randn(bs, num_experts)
    topk_weights, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_weights = topk_weights.softmax(dim=-1)
    if weight_sum != 1.0:
        # Unnormalised weights: combine must scale the output by their sum.
        topk_weights = topk_weights * weight_sum
    topk_ids = topk_ids.to(torch.int32)

    mask = None
    if pad_bs > bs:
        padded_x = torch.zeros(pad_bs, hidden)
        padded_x[:bs] = x
        padded_ids = torch.zeros(pad_bs, top_k, dtype=torch.int32)
        padded_ids[:bs] = topk_ids
        padded_weights = torch.zeros(pad_bs, top_k)
        padded_weights[:bs] = topk_weights
        mask = torch.zeros(pad_bs, dtype=torch.bool)
        mask[:bs] = True
        x, topk_ids, topk_weights = padded_x, padded_ids, padded_weights

    x = x.to(device).to(torch.bfloat16)
    topk_ids = topk_ids.to(device)
    topk_weights = topk_weights.to(device)
    if mask is not None:
        mask = mask.to(device)
    return x, topk_ids, topk_weights, mask


def _run_case(args, rank, world, comm, device, quant_mode, weight_sum, pad_bs):
    x, topk_ids, topk_weights, mask = _build_batch(
        1234 + rank,
        args.bs,
        pad_bs,
        args.hidden,
        args.num_experts,
        args.top_k,
        weight_sum,
        device,
    )

    (
        expand_x,
        dynamic_scales,
        assist_info_for_combine,
        expert_token_nums,
        ep_recv_counts,
        tp_recv_counts,
        expand_scales,
    ) = torch.ops.npu.npu_moe_distribute_dispatch_v2(
        x,
        topk_ids,
        comm,
        world,
        rank,
        args.num_experts,
        x_active_mask=mask,
        quant_mode=quant_mode,
        global_bs=0,
        expert_token_nums_type=1,
        comm_alg=args.dispatch_comm_alg,
    )

    local_experts = args.num_experts // world
    assert expert_token_nums.dtype == torch.int64, expert_token_nums.dtype
    assert tuple(expert_token_nums.shape) == (local_experts,), expert_token_nums.shape
    assert ep_recv_counts.dtype == torch.int32, ep_recv_counts.dtype
    _log(
        rank,
        f"[quant_mode={quant_mode} pad={pad_bs}] expand_x={tuple(expand_x.shape)} "
        f"{expand_x.dtype} scales={tuple(dynamic_scales.shape)} "
        f"{dynamic_scales.dtype} assist={tuple(assist_info_for_combine.shape)} "
        f"expert_token_nums={tuple(expert_token_nums.shape)} "
        f"ep_recv_counts={tuple(ep_recv_counts.shape)} "
        f"expand_scales={tuple(expand_scales.shape)} "
        f"recv_pairs={int(expert_token_nums.sum())}",
    )

    # Natural total routed pairs: every active token is routed to top_k experts.
    valid_bs = args.bs
    total_pairs = torch.tensor(
        [int(expert_token_nums.sum())], dtype=torch.int64, device=device
    )
    dist.all_reduce(total_pairs, op=dist.ReduceOp.SUM)
    expected_pairs = valid_bs * args.top_k * world
    if int(total_pairs) != expected_pairs:
        raise AssertionError(
            f"routed pairs {int(total_pairs)} != {expected_pairs} "
            f"(x_active_mask not honoured?)"
        )

    # Pass the "expert output" through unchanged: combine should weight-and-sum
    # the top-k routes, giving `weight_sum * x` for the active rows.
    expert_out = expand_x
    if quant_mode == 2:
        expert_out = (expand_x.float() * dynamic_scales[:, None]).to(torch.bfloat16)

    combined = torch.ops.npu.npu_moe_distribute_combine_v2(
        expert_out,
        topk_ids,
        assist_info_for_combine,
        ep_recv_counts,
        topk_weights,
        comm,
        world,
        rank,
        args.num_experts,
        x_active_mask=mask,
        # Dispatch is never given top-k weights, so the returned expand_scales is
        # filler: empty on some builds, uninitialised ``(a,)`` memory on A3.
        expand_scales=None,
        global_bs=0,
        comm_quant_mode=0,
        comm_alg=args.combine_comm_alg,
    )
    assert tuple(combined.shape) == (x.shape[0], x.shape[1]), combined.shape

    out = combined[: args.bs].float()
    ref = x[: args.bs].float() * weight_sum
    err = (out - ref).abs()
    scale = ref.abs().max().item()
    _log(
        rank,
        f"[quant_mode={quant_mode} pad={pad_bs} wsum={weight_sum}] "
        f"max_abs_err={err.max().item():.4f} "
        f"max_err/scale={(err.max().item() / max(scale, 1e-6)):.4f}",
    )
    # bf16 rounding and int8 quantisation both stay well inside atol+rtol*|ref|;
    # a wrong number of weight applications is off by a factor of top_k.
    bad = err > args.atol + args.rtol * ref.abs()
    if bad.any():
        raise AssertionError(
            f"round-trip mismatch: {int(bad.sum())} elements out of tolerance "
            f"(quant_mode={quant_mode}, weight_sum={weight_sum}, pad={pad_bs}, "
            f"max_abs_err={err.max().item():.4f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument(
        "--pad-bs",
        type=int,
        default=0,
        help="Pad the batch to this size and pass x_active_mask (0 disables).",
    )
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--dispatch-comm-alg", default="fullmesh_v1")
    parser.add_argument("--combine-comm-alg", default="")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device(f"npu:{local_rank}")
    comm = _hccl_comm_name(distributed_c10d._get_default_group(), rank)
    _log(rank, f"world={world} ep={world} comm={comm} device={device}")

    if args.num_experts % world != 0:
        raise SystemExit(
            f"--num-experts {args.num_experts} must be divisible by world size {world}"
        )

    pad_bs = args.pad_bs or args.bs
    # bf16: weights applied exactly once.
    _run_case(
        args, rank, world, comm, device, quant_mode=0, weight_sum=1.0, pad_bs=pad_bs
    )
    # bf16: unnormalised weights must scale the result by top_k.
    _run_case(
        args,
        rank,
        world,
        comm,
        device,
        quant_mode=0,
        weight_sum=float(args.top_k),
        pad_bs=pad_bs,
    )
    # int8 per-token dynamic quantisation.
    _run_case(
        args, rank, world, comm, device, quant_mode=2, weight_sum=1.0, pad_bs=pad_bs
    )
    if args.pad_bs:
        # Masked padding must not route the inactive rows; covered by the
        # routed-pairs assertion above.
        _log(rank, "padded + x_active_mask path OK")

    dist.barrier()
    _log(rank, "ALL OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
