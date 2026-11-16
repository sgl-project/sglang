"""The fp4 DSA prefill indexer (candidate slab) against today's path, DeepGEMM bf16 logits
plus DeepSelect's top-k, on single-request last chunks."""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsa.litetopk_prefill.fp4_indexer import (
    fp4_indexer_candidates,
    is_fp4_indexer_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="nightly", runner_config="4-gpu-b200")


def make_inputs(heads, rows, ctx, ratio):
    from deep_gemm.utils.math import per_token_cast_to_fp4

    torch.manual_seed(0)
    num_kv = ctx // ratio
    pos = torch.arange(ctx - rows, ctx, device="cuda")
    ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ke = ((pos + 1) // ratio).to(torch.int32)
    q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(num_kv, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
    q_fp4, q_sf = per_token_cast_to_fp4(
        q.view(-1, 128), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    kv_fp4, kv_sf = per_token_cast_to_fp4(
        kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    return (
        q_fp4.view(rows, heads, 64).contiguous(),
        q_sf.view(rows, heads).contiguous(),
        weights,
        kv_fp4.contiguous(),
        kv_sf.view(-1).contiguous(),
        ks,
        ke,
    )


def ours(q_fp4, q_sf, weights, kv_fp4, kv_sf, ks, ke, topk):
    return fp4_indexer_candidates(
        q_fp4, q_sf, weights, kv_fp4, kv_sf, ks, ke, topk
    ).scores


def deep_gemm_deep_select(q_fp4, q_sf, weights, kv_fp4, kv_sf, ks, ke, topk):
    from deep_gemm import fp8_fp4_mqa_logits

    from sglang.kernels.ops.attention.deep_select import topk as ds_topk

    rows = q_fp4.shape[0]
    width = (int((ke - ks).max()) + 63) // 64 * 64
    logits = fp8_fp4_mqa_logits(
        (q_fp4, q_sf),
        (kv_fp4, kv_sf),
        weights,
        ks,
        ke,
        False,
        width,
        logits_dtype=torch.bfloat16,
    )
    out = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    ds_topk(
        logits,
        topk,
        end=(ke - ks),
        output_idx=out,
        output_idx_offset=torch.zeros_like(ks),
        idx_oob_fill_value=-1,
    )
    return out


@marker.parametrize("heads,topk,ratio", [(64, 1024, 4), (32, 512, 2)], [(64, 1024, 4)])
@marker.parametrize("rows", [2048, 8192], [2048])
@marker.parametrize("ctx", [262144, 1048576], [262144])
@marker.benchmark("impl", ["ours", "deep_gemm_bf16+deep_select"])
def benchmark(heads, topk, ratio, rows, ctx, impl):
    if not is_fp4_indexer_supported():
        marker.skip("the fp4 prefill indexer needs SM100")
    args = make_inputs(heads, rows, ctx, ratio)
    fn = ours if impl == "ours" else deep_gemm_deep_select
    # the KV is re-read per row block, so a clone per iteration only moves it; time warm
    return marker.do_bench(
        fn,
        input_args=(*args, topk),
        graph_clone_args=(),
        use_cuda_graph=False,
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
