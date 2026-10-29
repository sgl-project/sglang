"""Benchmark: MiniMax-M3 EAGLE verify index step, per-row scoring vs one pass per request.

Score + top-k for `requests` requests of 4 linear-verify rows each, one index head per rank (TP4).
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.test.ci.ci_register import register_cuda_ci

# On B200: SM90 does not pack verify rows, so both arms would time its own kernels.
register_cuda_ci(est_time=30, stage="nightly", runner_config="4-gpu-b200")

HD, BLK, TOPK, PAGE, NDT = 128, 128, 16, 128, 4
PACK = {"per_row": 1, "packed": NDT}
# index K, q
DTYPES = {
    "bf16": (torch.bfloat16, torch.bfloat16),
    "fp8": (torch.float8_e4m3fn, torch.float8_e4m3fn),
}


def _index_step(q, k_cache, req_to_token, seq_lens, slot_ids, **static):
    return flash_decode_with_topk_idx(
        q=q,
        sink=None,
        k_cache=k_cache,
        v_cache=None,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        slot_ids=slot_ids,
        **static,
    )


@marker.parametrize(
    "requests,prefix,dtype",
    [
        (16, 128 * 1024, "bf16"),
        (32, 512 * 1024, "bf16"),
        (16, 128 * 1024, "fp8"),
        (32, 512 * 1024, "fp8"),
    ],
    [(16, 128 * 1024, "fp8")],
)
@marker.benchmark("impl", list(PACK))
def benchmark(requests: int, prefix: int, dtype: str, impl: str):
    k_dtype, q_dtype = DTYPES[dtype]
    width = prefix + NDT + 8
    pages = (width + PAGE - 1) // PAGE
    slots = torch.randperm(requests * pages, device="cuda").view(
        requests, pages, 1
    ) * PAGE + torch.arange(PAGE, device="cuda")
    rows = requests * NDT
    args = (
        torch.randn(rows, 1, HD, device="cuda").to(q_dtype),
        torch.randn(requests * pages * PAGE, 1, HD, device="cuda").to(k_dtype),
        slots.reshape(requests, -1)[:, :width].to(torch.int32).contiguous(),
        (
            torch.full((requests, 1), prefix, dtype=torch.int64, device="cuda")
            + torch.arange(1, NDT + 1, device="cuda")
        ).reshape(-1),
        torch.arange(requests, dtype=torch.int64, device="cuda").repeat_interleave(NDT),
    )
    static = dict(
        max_seqlen=prefix + NDT,
        block_size=BLK,
        topk=TOPK,
        init_blocks=0,
        local_blocks=1,
        score_type="max",
        disable_index_value=True,
        use_dense_main_attn=False,
        page_size=PAGE,
        k_scale=1.0,
        packed_queries=PACK[impl],
    )
    return marker.do_bench(
        _index_step,
        input_args=args,
        input_kwargs=static,
        graph_clone_kwargs=None,
        memory_args=(args[1],),
    )


if __name__ == "__main__":
    benchmark.run()
