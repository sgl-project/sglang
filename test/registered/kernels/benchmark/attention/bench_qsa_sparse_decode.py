"""Benchmark QSA (Qwen4-Exp) decode / target-verify attention over an fp8_e4m3
KV pool, new-token KV store included.

Lines:
  - ``trtllm``: the default backend path, ``fused_kv_prepare`` (store, block
    expansion and pack of the selected KV into a page-strided scratch) followed
    by FlashInfer's ``trtllm_batch_decode_with_kv_cache``.
  - ``cuda``: ``store_cache`` + ``expand_qsa_block_indices`` + the split-K
    ``qsa_sparse_decode`` kernel (``SGLANG_ENABLE_QSA_SPARSE_DECODE_CUDA=1``).

Shape axes: ``requests`` x ``width`` query rows (width 1 = decode, 4 = verify),
``context`` tokens per request (uniform in [context / 2, context)). Constants
follow the Qwen3.8-Flash-Next QSA layers at TP1: 24 q heads, 2 KV heads,
head_dim 256, 512 selected 4-token blocks plus a 3-token tail per row.
"""

import sys

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.utils import get_trtllm_gen_multi_ctas_kv_counter_bytes

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.qsa_sparse_decode import qsa_sparse_decode
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.srt.layers.attention.qsa.fused_kv import fused_kv_prepare
from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices
from sglang.srt.utils import is_sm100_supported, is_sm120
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=5, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

Q_HEADS, KV_HEADS, HEAD_DIM = 24, 2, 256
RATIO, BUDGET, PAGE = 4, 2048, 64
TOPK = BUDGET + RATIO - 1
STRIDE = -(-TOPK // PAGE) * PAGE


def _make_inputs(requests: int, width: int, context: int):
    gen = torch.Generator(device="cuda").manual_seed(requests * 31 + width)
    lens = torch.randint(
        context // 2, context, (requests,), generator=gen, device="cuda"
    )
    pages_per_req = (lens + width + PAGE - 1) // PAGE
    pages = int(pages_per_req.sum()) + 2
    order = (torch.randperm(pages - 1, generator=gen, device="cuda") + 1).int()
    req_to_token = torch.zeros(
        requests + 1, context + 2 * PAGE, dtype=torch.int32, device="cuda"
    )
    start = 0
    for r in range(requests):
        n = int(pages_per_req[r])
        req_to_token[r + 1, : n * PAGE] = (
            order[start : start + n, None] * PAGE + torch.arange(PAGE, device="cuda")
        ).flatten()
        start += n
    rows = requests * width
    row_req = torch.arange(
        1, requests + 1, dtype=torch.int32, device="cuda"
    ).repeat_interleave(width)
    positions = lens.repeat_interleave(width) + torch.arange(
        width, device="cuda"
    ).repeat(requests)
    num_blocks = (positions + 1) // RATIO
    scores = torch.rand(rows, int(num_blocks.max()), generator=gen, device="cuda")
    scores[torch.arange(scores.shape[1], device="cuda") >= num_blocks[:, None]] = -1
    top = scores.topk(BUDGET // RATIO, dim=1)
    fp8 = torch.float8_e4m3fn
    return dict(
        k_cache=torch.randn(pages * PAGE, KV_HEADS, HEAD_DIM, device="cuda").to(fp8),
        v_cache=torch.randn(pages * PAGE, KV_HEADS, HEAD_DIM, device="cuda").to(fp8),
        req_to_token=req_to_token,
        q=torch.randn(rows, Q_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16),
        new_k=torch.randn(rows, KV_HEADS, HEAD_DIM, device="cuda").to(fp8),
        new_v=torch.randn(rows, KV_HEADS, HEAD_DIM, device="cuda").to(fp8),
        loc=req_to_token[row_req.long(), positions].long(),
        blocks=torch.where(top.values >= 0, top.indices, -1).int(),
        seq_lens=(positions + 1).int(),
        query_positions=positions.int(),
        row_req=row_req,
    )


def _trtllm_fn(x, width):
    rows = x["q"].shape[0]
    packed_dtype = torch.float8_e4m3fn if is_sm100_supported() else torch.bfloat16
    packed_k = torch.zeros(
        rows * STRIDE, KV_HEADS, HEAD_DIM, dtype=packed_dtype, device="cuda"
    )
    packed_v = torch.zeros_like(packed_k)
    counts = torch.empty(rows, dtype=torch.int32, device="cuda")
    block_tables = torch.arange(
        rows * (STRIDE // PAGE), dtype=torch.int32, device="cuda"
    ).view(rows, -1)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    counters = torch.zeros(
        get_trtllm_gen_multi_ctas_kv_counter_bytes(rows, Q_HEADS, sms),
        dtype=torch.uint8,
        device="cuda",
    )
    kv_view = [
        t.view(-1, PAGE, KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        for t in (packed_k, packed_v)
    ]

    def fn(q, new_k, new_v, loc, blocks, seq_lens, query_positions, row_req):
        fused_kv_prepare(
            x["k_cache"],
            x["v_cache"],
            new_k,
            new_v,
            loc,
            x["req_to_token"],
            row_req,
            blocks,
            seq_lens,
            counts,
            packed_k,
            packed_v,
            width,
            compress_ratio=RATIO,
            query_positions=query_positions,
            chain_positions=True,
            visible_page_size=PAGE,
        )
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=tuple(kv_view),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=counts,
            max_seq_len=STRIDE,
            bmm1_scale=HEAD_DIM**-0.5,
            bmm2_scale=1.0,
            multi_ctas_kv_counter_buffer=counters,
        )

    return fn


def _cuda_fn(x):
    k_cache, v_cache = x["k_cache"], x["v_cache"]
    rows = x["q"].shape[0]

    def fn(q, new_k, new_v, loc, blocks, seq_lens, query_positions, row_req):
        store_cache(
            new_k.view(rows, -1),
            new_v.view(rows, -1),
            k_cache.view(-1, KV_HEADS * HEAD_DIM),
            v_cache.view(-1, KV_HEADS * HEAD_DIM),
            loc,
        )
        indices = expand_qsa_block_indices(
            blocks, query_positions, seq_lens, RATIO, BUDGET
        )
        return qsa_sparse_decode(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            req_to_token=x["req_to_token"],
            row_req_pool_indices=row_req,
            indices=indices,
            seq_lens=seq_lens,
            sm_scale=HEAD_DIM**-0.5,
        )

    return fn


@marker.parametrize("context", [8192, 65536], [8192])
@marker.parametrize(
    "requests,width",
    [(16, 1), (32, 1), (64, 1), (96, 1), (4, 4), (16, 4), (32, 4), (64, 4), (96, 4)],
    [(16, 1), (16, 4)],
)
@marker.benchmark("impl", ["trtllm", "cuda"])
def benchmark(requests: int, width: int, context: int, impl: str):
    x = _make_inputs(requests, width, context)
    fn = _trtllm_fn(x, width) if impl == "trtllm" else _cuda_fn(x)
    args = tuple(
        x[name]
        for name in (
            "q",
            "new_k",
            "new_v",
            "loc",
            "blocks",
            "seq_lens",
            "query_positions",
            "row_req",
        )
    )
    # Selected fp8 K and V bytes: the KV a sparse decode has to read.
    footprint = x["q"].shape[0] * TOPK * KV_HEADS * HEAD_DIM * 2
    return marker.do_bench(
        fn,
        input_args=args,
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=footprint,
    )


if __name__ == "__main__":
    if not (is_sm100_supported() or is_sm120()):
        print("[skip] QSA sparse decode benchmark requires SM100 or SM120 CUDA.")
        sys.exit(0)
    benchmark.run()
