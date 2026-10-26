"""MSA-v1 kernels read and write page-row cache views exactly like dense tensors.

The pool packs each page's components into one row, so a component view differs
from a dedicated tensor only in its page stride. Every kernel touching the cache
must produce identical bytes for both.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.msa_v1.compact_dequant import (
    dequant_nvfp4_page_table,
    gather_dequant_nvfp4_topk_pages_pair,
)
from sglang.kernels.ops.kvcache.nvfp4_store_kv_index import nvfp4_store_kv_index
from sglang.srt.layers.attention.minimax_sparse_ops.msa_plan import (
    MSA_V1_DECODE_INDEXER_AVAILABLE,
    MSAV1DecodeAttentionPlan,
    MSAV1DecodeIndexerPlan,
    MSAV1PrefillPlan,
)
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import (
    MiniMaxM3Nvfp4IndexKPool,
    MiniMaxM3Nvfp4MainKVPool,
)
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# The fused store and dequant kernels are sglang's own; the plans call into MSA.
requires_msa = pytest.mark.skipif(
    not MSA_V1_DECODE_INDEXER_AVAILABLE,
    reason="needs the MiniMax MSA package (minimax-msa)",
)

PAGE = 128
KV_HEADS = 2
Q_HEADS = 16 * KV_HEADS
BS = 2
MAX_PAGES = 20
LENGTHS = [MAX_PAGES * PAGE - 37, 17 * PAGE + 5]


def _dense(view):
    clone = view.clone()
    assert clone.is_contiguous() and clone.stride(0) != view.stride(0)
    return clone


@pytest.fixture(scope="module")
def caches():
    if not get_platform().is_sm100:
        pytest.skip("MSA-v1 NVFP4 kernels target SM100/SM103")
    torch.manual_seed(0)
    size = (BS * MAX_PAGES + 1) * PAGE
    main = MiniMaxM3Nvfp4MainKVPool(
        size, PAGE, torch.float8_e4m3fn, KV_HEADS, 128, 1, "cuda", False
    )
    index = MiniMaxM3Nvfp4IndexKPool(
        size, PAGE, torch.float8_e4m3fn, 1, 128, 1, "cuda", False
    )
    slots = torch.arange(PAGE, size, device="cuda")
    k = torch.randn(slots.numel(), KV_HEADS, 128, device="cuda").bfloat16() * 0.3
    main.set_kv_buffer(SimpleNamespace(layer_id=0), slots, k, torch.randn_like(k))
    index.set_k_buffer(0, slots, k[:, :1].contiguous())
    k, v = main.get_kv_buffer(0)
    views = SimpleNamespace(
        k=k,
        v=v,
        k_scale=main.get_k_scale_buffer(0),
        v_scale=main.get_v_scale_buffer(0),
        index_k=index.get_key_buffer(0),
        index_scale=index.get_scale_buffer(0),
    )
    dense = SimpleNamespace(**{name: _dense(t) for name, t in vars(views).items()})
    page_table = torch.arange(
        1, BS * MAX_PAGES + 1, device="cuda", dtype=torch.int32
    ).view(BS, MAX_PAGES)
    lengths = torch.tensor(LENGTHS, device="cuda", dtype=torch.int32)
    return SimpleNamespace(
        main=main,
        index=index,
        views=views,
        dense=dense,
        page_table=page_table,
        lengths=lengths,
    )


def _fp8(shape, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=gen) * 0.3).to(
        torch.float8_e4m3fn
    )


def test_fused_store_writes_identical_bytes(caches):
    # Private pools: the shared fixture's views must keep matching their snapshots.
    size = 8 * PAGE
    main = MiniMaxM3Nvfp4MainKVPool(
        size, PAGE, torch.float8_e4m3fn, KV_HEADS, 128, 1, "cuda", False
    )
    index = MiniMaxM3Nvfp4IndexKPool(
        size, PAGE, torch.float8_e4m3fn, 1, 128, 1, "cuda", False
    )
    n = 300
    k = torch.randn(n, KV_HEADS, 128, device="cuda").bfloat16()
    v, idx_k = torch.randn_like(k), torch.randn(n, 128, device="cuda").bfloat16()
    slots = torch.randperm(main.total_slots, device="cuda")[:n]
    strided = [*main.get_raw_kv_buffers(0), *index.get_raw_k_buffers(0)]
    dense = [_dense(t) for t in strided]
    for k_p, k_s, v_p, v_s, i_p, i_s in (
        (strided[0], strided[2], strided[1], strided[3], strided[4], strided[5]),
        (dense[0], dense[2], dense[1], dense[3], dense[4], dense[5]),
    ):
        nvfp4_store_kv_index(
            k,
            v,
            idx_k,
            k_p,
            k_s,
            v_p,
            v_s,
            i_p,
            i_s,
            slots,
            main.global_scale,
            page_size=PAGE,
        )
    for a, b in zip(strided, dense):
        assert torch.equal(a, b)


def test_dequant_kernels_match(caches):
    s, d, gs = caches.views, caches.dense, caches.main.global_scale
    query_rows = BS
    topk = torch.randint(
        0, MAX_PAGES, (query_rows, KV_HEADS, 16), device="cuda", dtype=torch.int32
    )
    outputs = []
    for c in (s, d):
        out_k = torch.empty(
            query_rows,
            KV_HEADS,
            16,
            128,
            128,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        out_v = torch.empty_like(out_k)
        gather_dequant_nvfp4_topk_pages_pair(
            c.k,
            c.v,
            c.k_scale,
            c.v_scale,
            gs,
            caches.page_table,
            topk,
            caches.lengths,
            out_k,
            out_v,
            q_len_per_req=1,
        )
        mirror = torch.zeros(
            c.index_k.shape[0], 128, 128, dtype=torch.float8_e4m3fn, device="cuda"
        )
        dequant_nvfp4_page_table(
            c.index_k,
            c.index_scale.view(torch.float8_e4m3fn),
            gs,
            caches.page_table,
            caches.lengths,
            mirror,
        )
        outputs.append((out_k, out_v, mirror))
    for a, b in zip(*outputs):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@requires_msa
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("query_length", [1, 4])
def test_decode_indexer_and_attention_match(caches, fallback, query_length):
    index, gs = caches.index, caches.main.global_scale
    rows = BS * query_length
    iq = _fp8((rows, KV_HEADS, 128), seed=1)
    q = _fp8((rows, Q_HEADS, 128), seed=2)
    results = []
    for c in (caches.views, caches.dense):
        indexer = MSAV1DecodeIndexerPlan(
            batch_size=BS,
            query_length=query_length,
            max_pages=MAX_PAGES,
            device=torch.device("cuda"),
            use_cuda_graph=False,
            num_index_heads=KV_HEADS,
            prefer_qmul4=not fallback,
            fp8_fallback_scratch=(
                torch.empty(
                    index.num_pages, 128, 128, dtype=torch.float8_e4m3fn, device="cuda"
                )
                if fallback
                else None
            ),
        )
        attention = MSAV1DecodeAttentionPlan(
            batch_size=BS,
            query_length=query_length,
            max_pages=MAX_PAGES,
            num_q_heads=Q_HEADS,
            num_kv_heads=KV_HEADS,
            device=torch.device("cuda"),
            sm_scale=128**-0.5,
            prefer_qmul4=not fallback,
        )
        indexer.plan(caches.page_table, caches.lengths)
        attention.plan(caches.page_table, caches.lengths)
        topk = indexer.run(
            iq,
            c.index_k,
            c.index_scale,
            gs,
            out=attention.topk_qht.permute(1, 0, 2),
        )
        out = attention.run(q, topk, c.k, c.v, c.k_scale, c.v_scale, gs)
        results.append((topk.clone(), out.clone()))
    for a, b in zip(*results):
        assert torch.equal(a, b)


@requires_msa
@pytest.mark.parametrize("prefer_qmul4", [True, False])
def test_prefill_indexer_and_attention_match(caches, prefer_qmul4):
    gs = caches.main.global_scale
    q_lens = [37, 130]
    total_q = sum(q_lens)
    cu = torch.tensor([0, q_lens[0], total_q], dtype=torch.int32, device="cuda")
    idx_q = _fp8((total_q, KV_HEADS, 128), seed=3)
    q = _fp8((total_q, Q_HEADS, 128), seed=4)
    results = []
    for c in (caches.views, caches.dense):
        plan = MSAV1PrefillPlan(
            cu_seqlens_q=cu,
            seq_lens=caches.lengths,
            page_table=caches.page_table,
            total_q=total_q,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(LENGTHS),
            prefer_qmul4=prefer_qmul4,
            num_index_heads=KV_HEADS,
        )
        topk = plan.run_indexer(idx_q, c.index_k, c.index_scale, gs)
        out = plan.run_attention(
            q,
            c.k,
            c.v,
            topk,
            sm_scale=128**-0.5,
            k_scale=c.k_scale,
            v_scale=c.v_scale,
            global_scale=gs,
        )
        results.append((topk.clone(), out.clone()))
    for a, b in zip(*results):
        assert torch.equal(a, b)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
