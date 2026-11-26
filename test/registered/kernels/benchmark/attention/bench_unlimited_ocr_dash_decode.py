"""Benchmark: Unlimited-OCR v2 DASH decode hot path per layer -- tail page
summary refresh + windowed entmax routing, and the split-K sparse attend.

Each stage is captured into its own CUDA graph and every timed replay starts
from a flushed L2 (``marker.do_bench`` without its graph loop), so the random
KV pages the router picks are read cold, as in serving.
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.unlimited_ocr_dash.attend import (
    dash_sparse_attn_decode,
)
from sglang.kernels.ops.attention.unlimited_ocr_dash.route import (
    SCAN_TILE,
    kbar_refresh_tail_triton,
    kbar_tail_plan,
    route_decode_triton,
    win_bounds_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

BLOCK_SIZE, INDEX_DIM, LOCAL_BLOCKS = 128, 128, 1
# Arbitrary per-rank MHA geometry; the checkpoint defines the real one.
NUM_HEADS = NUM_KV_HEADS = 16
HEAD_DIM = 128
# Router widths of a typical checkpoint: top-M 16, 8 output slots.
TOPM, OUT_SLOTS, BUDGET = 16, 8, 8
# One image page spans about 16 blocks; the allowed span fits SCAN_TILE.
ALLOWED_SPAN = 32
NUM_PAGES = 4096


def _decode_state(batch: int, ctx: int):
    width = ctx // BLOCK_SIZE
    gen = torch.Generator(device="cuda").manual_seed(batch * ctx)
    bt = torch.stack(
        [
            torch.randperm(NUM_PAGES, device="cuda", generator=gen)[:width]
            for _ in range(batch)
        ]
    ).to(torch.int32)
    seq = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    allow = torch.zeros(batch, width, dtype=torch.bool, device="cuda")
    starts = torch.randint(
        0,
        width - ALLOWED_SPAN - 2,
        (batch,),
        generator=torch.Generator().manual_seed(ctx),
    )
    for r, s in enumerate(starts.tolist()):
        allow[r, s : s + ALLOWED_SPAN] = True
    lo, hi = (torch.zeros(batch, dtype=torch.int32, device="cuda") for _ in range(2))
    fit = torch.zeros(batch, dtype=torch.int8, device="cuda")
    win_bounds_decode(
        allow=allow,
        seq_lens=seq,
        lo=lo,
        hi=hi,
        fit=fit,
        max_block=width,
        block_size=BLOCK_SIZE,
        local_blocks=LOCAL_BLOCKS,
        decode_query_len=1,
        tile=SCAN_TILE,
    )
    kv_shape = (NUM_PAGES, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    return dict(
        width=width,
        block_table=bt,
        seq_lens=seq,
        allow=allow.to(torch.int8),
        window=(lo, hi, fit),
        plan=kbar_tail_plan(
            block_table=bt,
            seq_lens=seq,
            block_size=BLOCK_SIZE,
            tail_pages=1,
            device="cuda",
        ),
        index_kv=torch.randn(NUM_PAGES, BLOCK_SIZE, INDEX_DIM, device="cuda").to(
            torch.bfloat16
        ),
        k_bar=torch.randn(NUM_PAGES, INDEX_DIM, device="cuda"),
        head_cls=torch.randn(INDEX_DIM, device="cuda").to(torch.bfloat16),
        idx_q=torch.randn(batch, INDEX_DIM, device="cuda").to(torch.bfloat16),
        k_cache=torch.randn(kv_shape, device="cuda").to(torch.bfloat16),
        v_cache=torch.randn(kv_shape, device="cuda").to(torch.bfloat16),
        q=torch.randn(batch, NUM_HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16),
        out=torch.empty(batch, NUM_HEADS, HEAD_DIM, device="cuda").to(torch.bfloat16),
        out_idx=torch.empty(batch, TOPM, dtype=torch.int32, device="cuda"),
        out_bias=torch.empty(batch, TOPM, device="cuda"),
        scores=torch.empty(batch, width, device="cuda"),
        scores_win=torch.empty(batch, SCAN_TILE, device="cuda"),
    )


def _route(s):
    lo, hi, fit = s["window"]
    kbar_refresh_tail_triton(
        kv_cache=s["index_kv"],
        k_bar=s["k_bar"],
        block_table=s["block_table"],
        seq_lens=s["seq_lens"],
        head_cls=s["head_cls"],
        block_size=BLOCK_SIZE,
        plan=s["plan"],
    )
    route_decode_triton(
        idx_q=s["idx_q"],
        k_bar=s["k_bar"],
        block_table=s["block_table"],
        seq_lens=s["seq_lens"],
        image_allow=s["allow"],
        out_idx=s["out_idx"],
        out_bias=s["out_bias"],
        max_block=s["width"],
        block_size=BLOCK_SIZE,
        local_blocks=LOCAL_BLOCKS,
        decode_query_len=1,
        idx_scale=0.37,
        prior_floor=1e-2,
        sigma=1.0,
        topm=TOPM,
        score_buf=s["scores"],
        budget=BUDGET,
        out_slots=OUT_SLOTS,
        entmax_base=lo,
        entmax_fit=fit,
        entmax_width=SCAN_TILE,
        score_buf_win=s["scores_win"],
        bounds=(lo, hi),
    )


def _attend(s):
    idx = s["out_idx"][:, :OUT_SLOTS].unsqueeze(0).expand(NUM_KV_HEADS, -1, -1)
    bias = s["out_bias"][:, :OUT_SLOTS].unsqueeze(0).expand(NUM_KV_HEADS, -1, -1)
    dash_sparse_attn_decode(
        s["q"],
        s["k_cache"],
        s["v_cache"],
        idx,
        bias,
        s["block_table"],
        s["seq_lens"],
        NUM_KV_HEADS,
        HEAD_DIM**-0.5,
        s["out"],
        1,
        local_blocks=LOCAL_BLOCKS,
        sparse_block_size=BLOCK_SIZE,
        chunk_width=TOPM,
    )


STAGES = {
    "route": (_route,),
    "attend": (_attend,),
    "route+attend": (_route, _attend),
}


def _graph(state, stage: str):
    def step():
        for fn in STAGES[stage]:
            fn(state)

    # The router's output feeds the attend stage; route once so it is valid.
    _route(state)
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


@marker.parametrize("ctx", [16384, 131072], [16384])
@marker.parametrize("batch", [1, 8, 32], [1, 32])
@marker.benchmark("stage", list(STAGES))
def benchmark(ctx: int, batch: int, stage: str):
    graph = _graph(_decode_state(batch, ctx), stage)
    # Without the internal graph loop, do_bench flushes L2 before every replay.
    return marker.do_bench(
        graph.replay, use_cuda_graph=False, disable_log_bandwidth=True
    )


if __name__ == "__main__":
    benchmark.run()
