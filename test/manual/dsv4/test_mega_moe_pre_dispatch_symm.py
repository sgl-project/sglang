"""B200 smoke test for pre-dispatch writes into a real MegaMoE symm buffer.

Example:
    python test/manual/dsv4/test_mega_moe_pre_dispatch_symm.py --num-processes 2
"""

from __future__ import annotations

import argparse

import deep_gemm
import torch
import torch.distributed as dist
from deep_gemm.utils import per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist

from sglang.kernels.ops.attention.dsv4 import mega_moe_pre_dispatch
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate


def _run_case(
    buffer,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
    rank: int,
) -> None:
    torch.manual_seed(rank * 1009 + num_tokens)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")

    # Compile the SGLang JIT kernel before launching the primary TopK kernel.
    warmup_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")
    warmup_weights = torch.zeros((num_tokens, topk), device="cuda")
    mega_moe_pre_dispatch(
        x,
        warmup_ids,
        warmup_weights,
        buffer.x,
        buffer.x_sf,
        buffer.topk_idx,
        buffer.topk_weights,
        quant_group_size=32,
    )
    torch.cuda.synchronize()
    buffer.x.zero_()
    buffer.x_sf.zero_()
    buffer.topk_idx.fill_(-777)
    buffer.topk_weights.fill_(float("nan"))

    if num_tokens:
        scores = torch.randn((num_tokens, num_experts), device="cuda")
        bias = torch.randn((num_experts,), device="cuda")
        topk_weights, topk_ids = moe_fused_gate(
            scores,
            bias,
            topk,
            scoring_func="sqrtsoftplus",
        )
    else:
        topk_ids = torch.empty((0, topk), dtype=torch.int32, device="cuda")
        topk_weights = torch.empty((0, topk), device="cuda")

    mega_moe_pre_dispatch(
        x,
        topk_ids,
        topk_weights,
        buffer.x,
        buffer.x_sf,
        buffer.topk_idx,
        buffer.topk_weights,
        quant_group_size=32,
    )
    torch.cuda.synchronize()

    if num_tokens:
        x_ref, x_sf_ref = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        assert torch.equal(
            buffer.x[:num_tokens].view(torch.int8), x_ref.view(torch.int8)
        )
        assert torch.equal(buffer.x_sf[:num_tokens], x_sf_ref)
        assert torch.equal(buffer.topk_idx[:num_tokens], topk_ids.to(torch.int64))
        assert torch.equal(buffer.topk_weights[:num_tokens], topk_weights)
    assert torch.all(buffer.topk_idx[num_tokens:] == -1)
    assert torch.all(buffer.topk_weights[num_tokens:] == 0)


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = None
    try:
        capability = torch.cuda.get_device_capability()
        if capability[0] != 10:
            raise RuntimeError(
                f"MegaMoE symmetric-buffer smoke requires SM100, got {capability}"
            )
        if args.num_experts % num_ranks:
            raise ValueError("num_experts must be divisible by num_processes")

        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            args.num_experts,
            args.num_max_tokens_per_rank,
            args.num_topk,
            args.hidden,
            args.intermediate_hidden,
            use_fp8_dispatch=True,
            mma_type="fp8xfp4",
            activation="swiglu",
        )
        for num_tokens in (0, 1, 7, args.num_max_tokens_per_rank):
            _run_case(
                buffer,
                num_tokens,
                args.hidden,
                args.num_experts,
                args.num_topk,
                rank,
            )
            dist_print(f"pre-dispatch symm-buffer M={num_tokens}: PASS")
        dist.barrier()
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate-hidden", type=int, default=3072)
    args = parser.parse_args()
    torch.multiprocessing.spawn(
        _worker, args=(args.num_processes, args), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()
