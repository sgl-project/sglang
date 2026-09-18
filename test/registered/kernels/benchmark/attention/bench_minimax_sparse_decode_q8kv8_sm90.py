import argparse
from functools import partial

import torch
import triton

from sglang.kernels.ops.attention.minimax_sparse.decode.sgl_native_q8kv8 import (
    sgl_native_q8kv8_sparse_decode,
)
from sglang.kernels.ops.attention.minimax_sparse.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def make_case(batch: int, seq_len: int, topk: int):
    torch.manual_seed(1)
    device = "cuda"
    num_q_heads, num_kv_heads, head_dim, block_size = 8, 1, 128, 128
    num_blocks = (seq_len + block_size - 1) // block_size
    max_slots = num_blocks * block_size
    q = (torch.randn(batch, num_q_heads, head_dim, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    k = (torch.randn(max_slots, num_kv_heads, head_dim, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    v = (torch.randn_like(k.float()) * 0.2).to(torch.float8_e4m3fn)
    req_to_token = (
        torch.arange(max_slots, dtype=torch.int32, device=device)
        .expand(batch, -1)
        .contiguous()
    )
    slot_ids = torch.arange(batch, dtype=torch.int64, device=device)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    selected = torch.randperm(num_blocks, device=device)[:topk].sort().values.int()
    topk_idx = selected.view(1, 1, topk).expand(num_kv_heads, batch, topk).contiguous()
    return q, k, v, req_to_token, slot_ids, seq_lens, topk_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,8,32")
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=32)
    args = parser.parse_args()

    print("batch,native_us,triton_us,speedup,max_abs")
    for batch in map(int, args.batch_sizes.split(",")):
        q, k, v, req_to_token, slot_ids, seq_lens, topk_idx = make_case(
            batch, args.seq_len, args.topk
        )

        native = partial(
            sgl_native_q8kv8_sparse_decode,
            q,
            k,
            v,
            req_to_token,
            slot_ids,
            seq_lens,
            topk_idx,
            128,
            128,
        )
        triton_step3 = partial(
            flash_decode_with_gqa_share_sparse,
            q=q,
            sink=None,
            k_cache=k,
            v_cache=v,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            slot_ids=slot_ids,
            block_size=128,
            topk_idx=topk_idx,
        )

        native_out = native()
        triton_out = triton_step3()
        torch.cuda.synchronize()
        native_ms = triton.testing.do_bench(native, warmup=100, rep=500)
        triton_ms = triton.testing.do_bench(triton_step3, warmup=100, rep=500)
        max_abs = (native_out.float() - triton_out.float()).abs().max().item()
        print(
            f"{batch},{native_ms * 1000:.2f},{triton_ms * 1000:.2f},"
            f"{triton_ms / native_ms:.3f},{max_abs:.6f}"
        )


if __name__ == "__main__":
    main()
