"""MSA-v1 NVFP4 plans: decode top-k, multi-token verify, and prefill indexing."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.msa_v1.compact_dequant import dequant_nvfp4_page_table
from sglang.srt.layers.attention.minimax_sparse_ops.msa_plan import (
    MSA_V1_DECODE_INDEXER_AVAILABLE,
    MSAV1DecodeAttentionPlan,
    MSAV1DecodeIndexerPlan,
    MSAV1PrefillPlan,
    build_msa_v1_page_table,
)
from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import (
    MiniMaxM3Nvfp4IndexKPool,
    MiniMaxM3Nvfp4MainKVPool,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not MSA_V1_DECODE_INDEXER_AVAILABLE,
    reason="needs the MiniMax MSA package (minimax-msa)",
)

BS = 3
PAGE = 128
KV_HEADS = 2
# Well over the 16 pages top-k keeps, so the selection itself is exercised.
MAX_PAGES = 40
LENGTH_SETS = (
    [MAX_PAGES * PAGE, MAX_PAGES * PAGE - 77, 33 * PAGE + 5],
    [37 * PAGE + 1, 32 * PAGE, MAX_PAGES * PAGE - 1],
    [35 * PAGE, 39 * PAGE + 64, 34 * PAGE + 127],
)


@pytest.fixture(scope="module")
def caches():
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
    page_table = torch.arange(
        1, BS * MAX_PAGES + 1, device="cuda", dtype=torch.int32
    ).view(BS, MAX_PAGES)
    return main, index, page_table


def _indexer(
    query_length,
    fallback,
    index,
    batch_size=BS,
    max_pages=MAX_PAGES,
    heads=1,
    graph=True,
):
    scratch = (
        torch.empty(index.num_pages, 128, 128, dtype=torch.float8_e4m3fn, device="cuda")
        if fallback
        else None
    )
    return MSAV1DecodeIndexerPlan(
        batch_size=batch_size,
        query_length=query_length,
        max_pages=max_pages,
        device=torch.device("cuda"),
        use_cuda_graph=graph,
        num_index_heads=heads,
        prefer_qmul4=not fallback,
        fp8_fallback_scratch=scratch,
    )


def _attention(
    query_length, fallback, num_q_heads, num_kv_heads, batch_size, max_pages, **shared
):
    return MSAV1DecodeAttentionPlan(
        batch_size=batch_size,
        query_length=query_length,
        max_pages=max_pages,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        device=torch.device("cuda"),
        sm_scale=128**-0.5,
        prefer_qmul4=not fallback,
        **shared,
    )


def _plan(indexer, page_table, lengths):
    indexer.plan(
        page_table, torch.tensor(lengths, device=page_table.device, dtype=torch.int32)
    )


def _index_q(query_length, seed, heads=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(BS * query_length, heads, 128, device="cuda", generator=gen) * 0.3
    return q.to(torch.float8_e4m3fn)


def _index_cache(index):
    return index.get_key_buffer(0), index.get_scale_buffer(0), index.global_scale


def _check_topk(q, index, table, query_length, kv_lengths, output):
    """Selected pages carry the top-16 page scores, with the local page forced last."""
    keys = NVFP4KVQuantizeUtil.dequantize(
        index.get_key_buffer(0),
        index.get_scale_buffer(0).view(torch.float8_e4m3fn),
        index.global_scale,
    )
    keys = keys.to(torch.float8_e4m3fn).float()
    heads = q.shape[1]
    for batch, kv_length in enumerate(kv_lengths):
        rows = slice(batch * query_length, (batch + 1) * query_length)
        page_keys = keys[table[batch].long()].flatten(0, 1)
        scores = torch.einsum("qhd,kd->hqk", q[rows].float(), page_keys)
        scores = scores.unflatten(-1, (-1, PAGE)).amax(-1)
        local = (kv_length - query_length + torch.arange(query_length).cuda()) // PAGE
        pages = torch.arange(table.shape[1], device="cuda")
        scores.masked_fill_(pages[None, None] >= local[None, :, None], -torch.inf)
        scores.scatter_(-1, local[None, :, None].expand(heads, -1, -1), torch.inf)
        selected = output[:, rows].long()
        assert torch.equal(selected[..., -1], local[None].expand(heads, -1))
        torch.testing.assert_close(
            scores.gather(-1, selected).sort(descending=True).values,
            scores.topk(16).values,
            rtol=2e-4,
            atol=2e-4,
        )


# Decode, verify and a verify longer than one 16-query launch, at TP1/TP2/TP4 heads.
@pytest.mark.parametrize(
    "query_length, fallback, heads",
    [(1, False, 1), (4, False, 2), (8, False, 1), (8, True, 1), (20, False, 4)],
)
def test_topk_matches_reference_and_graph_replay(caches, query_length, fallback, heads):
    _, index, page_table = caches
    indexer = _indexer(query_length, fallback, index, heads=heads)
    # The backend hands the attention plan's [Q, H, 16] buffer transposed.
    out_qht = torch.empty(BS * query_length, heads, 16, dtype=torch.int32).cuda()
    out = out_qht.permute(1, 0, 2)
    iq = _index_q(query_length, seed=1, heads=heads)
    _plan(indexer, page_table, LENGTH_SETS[0])
    topk = indexer.run(iq, *_index_cache(index), out=out)
    assert topk.data_ptr() == out_qht.data_ptr()
    _check_topk(iq, index, page_table, query_length, LENGTH_SETS[0], topk)
    assert (topk[..., :15] >= 16).any(), "top-k never had to pick among pages"

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = indexer.run(iq, *_index_cache(index), out=out)
    for seed, lengths in enumerate(LENGTH_SETS[1:], start=2):
        iq.copy_(_index_q(query_length, seed, heads))
        _plan(indexer, page_table, lengths)
        graph.replay()
        _check_topk(iq, index, page_table, query_length, lengths, captured)


def test_decode_indexer_rejects_unsupported_head_count(caches):
    with pytest.raises(ValueError, match="index heads"):
        _indexer(1, False, caches[1], heads=3)


@pytest.mark.parametrize(
    "query_length, fallback", [(1, False), (8, False), (11, False), (11, True)]
)
def test_nvfp4_verify_causal_attention_and_graph_replay(caches, query_length, fallback):
    main, index, _ = caches
    torch.manual_seed(42)
    bs, n, hq = 2, 193, 32
    q = (torch.randn(bs * query_length, hq, 128, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    iq = (torch.randn(bs * query_length, KV_HEADS, 128, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    pt = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.int32)
    lens = torch.tensor([n, n - 11], device="cuda", dtype=torch.int32)
    dp = _indexer(
        query_length, fallback, index, batch_size=bs, max_pages=2, heads=KV_HEADS
    )
    ap = _attention(query_length, fallback, hq, KV_HEADS, bs, 2)
    assert ap.uses_qmul4 == (not fallback)

    def run():
        topk = dp.run(iq, *_index_cache(index), out=ap.topk_qht.permute(1, 0, 2))
        return ap.run(
            q,
            topk,
            *main.get_kv_buffer(0),
            main.get_k_scale_buffer(0),
            main.get_v_scale_buffer(0),
            main.global_scale,
        )

    def dequant(payload, scales):
        x = NVFP4KVQuantizeUtil.dequantize(payload, scales, main.global_scale)
        return x.to(torch.float8_e4m3fn).float().repeat_interleave(hq // KV_HEADS, 1)

    def check(output, lengths, table):
        results = []
        for i, (length, row) in enumerate(zip(lengths, table)):
            slots = torch.cat(
                [torch.arange(p * PAGE, (p + 1) * PAGE, device="cuda") for p in row]
            )[:length]
            pk, pv, ks, vs = main.get_quantized_rows(0, slots)
            query = q[i * query_length : (i + 1) * query_length].float()
            logits = torch.einsum("qhd,khd->hqk", query, dequant(pk, ks)) * 128**-0.5
            last_visible = length - query_length + torch.arange(query_length).cuda()
            mask = torch.arange(length, device="cuda")[None, :] > last_visible[:, None]
            logits.masked_fill_(mask[None], -float("inf"))
            results.append(
                torch.einsum("hqk,khd->qhd", logits.softmax(-1), dequant(pv, vs))
            )
        ref = torch.cat(results)
        torch.cuda.synchronize()
        assert torch.isfinite(output).all()
        assert ((output.float() - ref).norm() / ref.norm()).item() < 0.04

    for values in ([n, n - 11], [n - 7, n - 3]):
        lens.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        dp.plan(pt, lens)
        ap.plan(pt, lens)
        check(run(), values, [[1, 2], [3, 4]])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    pt.copy_(torch.tensor([[3, 4], [1, 2]], device="cuda", dtype=torch.int32))
    lens.copy_(torch.tensor([n - 1, n - 5], device="cuda", dtype=torch.int32))
    dp.plan(pt, lens)
    ap.plan(pt, lens)
    graph.replay()
    check(captured, [n - 1, n - 5], [[3, 4], [1, 2]])


# Rows end 4 slots past the last page, the pool's minimum context_len headroom;
# dead page p starts at -129 - 128p: floor gives page -2 - p, truncation -1 - p.
def _req_to_token(page_table, pages):
    live_pages = page_table.shape[1]
    req_to_token = torch.full(
        (page_table.shape[0], pages * PAGE + 4), -129, dtype=torch.int32, device="cuda"
    )
    req_to_token[:, : live_pages * PAGE : PAGE] = page_table * PAGE
    dead = torch.arange(live_pages, pages, dtype=torch.int32, device="cuda")
    req_to_token[:, live_pages * PAGE : pages * PAGE : PAGE] = -129 - PAGE * dead
    return req_to_token


def _run(caches, indexer, attention, q, iq):
    main, index, _ = caches
    topk = indexer.run(
        iq, *_index_cache(index), out=attention.topk_qht.permute(1, 0, 2)
    )
    return attention.run(
        q,
        topk,
        *main.get_kv_buffer(0),
        main.get_k_scale_buffer(0),
        main.get_v_scale_buffer(0),
        main.global_scale,
    )


# The eager op sequence that the fused refresh replaces.
def _legacy_refresh(
    indexer, attention, req_to_token, req_pool_indices, seq_lens, verify
):
    page_table = build_msa_v1_page_table(
        req_to_token, req_pool_indices, max_seqlen_k=indexer.max_pages * PAGE
    )
    final_seq_lens = seq_lens.to(torch.int32)
    if verify:
        final_seq_lens = final_seq_lens + indexer.query_length
    indexer.plan(page_table, final_seq_lens)
    attention.plan(page_table, final_seq_lens)


# What the indexer and attention kernels read, through their own pointers.
def _metadata(indexer, attention):
    metadata = {
        "page_table": attention.page_table,
        # The Q8K8 fallback dequantizes the index rows up to these lengths.
        "final_seq_lens": indexer.final_seq_lens,
        "query_seq_lens": attention.query_seq_lens,
    }
    if attention.fp8_attention is not None:
        metadata["flashinfer_seq_lens"] = attention.fp8_attention.seq_lens
    for kind, wrappers in (("qmul4", indexer.wrappers), ("q8k8", indexer.fp8_wrappers)):
        for chunk, wrapper in enumerate(wrappers):
            if wrapper is None:
                continue
            proxy = wrapper._proxy_score
            schedule = proxy._shared_plan
            if indexer.shared_plans is not None:
                # Graph plans read the schedule that plan() and refresh() update.
                assert schedule is indexer.shared_plans[chunk], (kind, chunk)
            metadata[kind, chunk, "page_table"] = proxy._page_table
            metadata[kind, chunk, "chunk_seq_lens"] = schedule._source_lengths
            metadata[kind, chunk, "seq_lens"] = proxy._seq_lens
            metadata[kind, chunk, "scheduler"] = schedule._scheduler
            metadata[kind, chunk, "num_valid_pages"] = wrapper._num_valid_pages
    return metadata


@pytest.mark.parametrize(
    "query_length, num_reqs, pages, runs_kernels, qmul4_indexer",
    [
        (1, BS + 1, MAX_PAGES + 2, True, False),
        (8, BS + 1, MAX_PAGES + 2, True, False),
        (20, BS + 1, MAX_PAGES + 2, True, False),
        # 257 requests (2056 query rows) cross the 256-request and 1024-row tiles.
        (8, 257, MAX_PAGES + 2, False, False),
        # 8192 pages (1M tokens) is eight page blocks; 1025 adds a one-page tail block.
        (1, BS + 1, 8192, False, False),
        (20, BS + 1, 1025, False, False),
        (1, BS + 1, MAX_PAGES + 2, True, True),
        (20, BS + 1, MAX_PAGES + 2, True, True),
    ],
    ids=[
        "decode",
        "verify",
        "verify-two-chunks",
        "verify-257-requests",
        "decode-8192-pages",
        "verify-two-page-blocks",
        "decode-qmul4-indexer",
        "verify-two-chunks-qmul4-indexer",
    ],
)
def test_fused_update_matches_legacy_refresh_and_graph_replay(
    caches, query_length, num_reqs, pages, runs_kernels, qmul4_indexer
):
    _, index, page_table = caches
    verify = query_length > 1
    # Metadata-only cases skip CuTe compiles for new shapes; one KV head halves K/V.
    q_heads, kv_heads = (32, KV_HEADS) if runs_kernels else (16, 1)
    offset = query_length if verify else 0
    # Pool row 0 is the padding row; columns past MAX_PAGES are garbage.
    req_to_token = _req_to_token(
        torch.cat((torch.zeros_like(page_table[:1]), page_table)), pages
    )

    def plans(fused):
        # The legacy indexer takes the eager path, where each wrapper plans its own
        # MSA schedule, as every decode plan did before the fused refresh.
        indexer = _indexer(
            query_length,
            fallback=not qmul4_indexer,
            index=index,
            batch_size=num_reqs,
            max_pages=pages,
            heads=kv_heads,
            graph=fused,
        )
        # The backend hands the indexer's page table and lengths to attention.
        shared = dict(
            page_table_buffer=indexer.page_table,
            final_seq_lens_buffer=indexer.final_seq_lens,
        )
        attention = _attention(
            query_length,
            fallback=True,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            batch_size=num_reqs,
            max_pages=pages,
            **(shared if fused else {}),
        )
        return indexer, attention

    fused_indexer, fused_attention = plans(fused=True)
    legacy_indexer, legacy_attention = plans(fused=False)
    for step, dtype in enumerate((torch.int64, torch.int32)):
        # Request i reads pool row i % BS + 1, rotated by step;
        # the last request pads a graph bucket: pool row 0, seq_len 1.
        requests = range(num_reqs - 1)
        req_pool_indices = torch.tensor(
            [(i + 2 * step) % BS + 1 for i in requests] + [0],
            dtype=dtype,
            device="cuda",
        )
        seq_lens = torch.tensor(
            [LENGTH_SETS[step][i % BS] - offset for i in requests] + [1],
            dtype=dtype,
            device="cuda",
        )
        fused_indexer.refresh(
            attention_plan=fused_attention,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            is_target_verify=verify,
        )
        _legacy_refresh(
            indexer=legacy_indexer,
            attention=legacy_attention,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            verify=verify,
        )
        fused_metadata = _metadata(indexer=fused_indexer, attention=fused_attention)
        legacy_metadata = _metadata(indexer=legacy_indexer, attention=legacy_attention)
        assert fused_metadata.keys() == legacy_metadata.keys()
        for name, expected in legacy_metadata.items():
            torch.testing.assert_close(
                fused_metadata[name],
                expected,
                rtol=0,
                atol=0,
                msg=lambda report, step=step, name=name: (
                    f"step {step}: {name}\n{report}"
                ),
            )
        if runs_kernels and step == 0:
            torch.manual_seed(7)
            q = (
                torch.randn(num_reqs * query_length, q_heads, 128, device="cuda") * 0.3
            ).to(torch.float8_e4m3fn)
            iq = (
                torch.randn(num_reqs * query_length, kv_heads, 128, device="cuda") * 0.3
            ).to(torch.float8_e4m3fn)
            _run(caches, indexer=fused_indexer, attention=fused_attention, q=q, iq=iq)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = _run(
                    caches, indexer=fused_indexer, attention=fused_attention, q=q, iq=iq
                )
    if runs_kernels:
        legacy_output = _run(
            caches, indexer=legacy_indexer, attention=legacy_attention, q=q, iq=iq
        )
        # Replayed kernels read plan-owned buffers only, as PRE_REPLAY requires.
        req_to_token.fill_(-129)
        graph.replay()
        torch.testing.assert_close(
            fused_attention.topk_qht, legacy_attention.topk_qht, rtol=0, atol=0
        )
        torch.testing.assert_close(captured, legacy_output, rtol=0, atol=0)


def test_nvfp4_prefill_indexer_preserves_each_head():
    """Each head must retain its own selected pages when the indexer reuses outputs."""
    heads, pages, queries = 4, 32, 3
    packed_k = torch.zeros(pages, 128, 64, dtype=torch.uint8, device="cuda")
    # Page p has only coordinate p set to 6, making page scores proportional to Q[p].
    for page in range(pages):
        packed_k[page, :, page // 2] = 7 << (4 * (page % 2))
    scales = torch.ones(pages, 128, 8, device="cuda").to(torch.float8_e4m3fn)
    global_scale = torch.ones(1, device="cuda")
    columns = torch.arange(pages, device="cuda")
    offsets = (
        7 * torch.arange(heads, device="cuda")[None, :, None]
        + 3 * torch.arange(queries, device="cuda")[:, None, None]
    )
    q = torch.zeros(queries, heads, 128, device="cuda")
    q[:, :, :pages] = (columns - offsets).remainder(pages).float() / 2 - 8
    q = q.to(torch.float8_e4m3fn)
    plan = MSAV1PrefillPlan(
        cu_seqlens_q=torch.tensor([0, queries], dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor([pages * 128], dtype=torch.int32, device="cuda"),
        page_table=torch.arange(pages, dtype=torch.int32, device="cuda")[None, :],
        total_q=queries,
        max_seqlen_q=queries,
        max_seqlen_k=pages * 128,
        num_index_heads=heads,
    )

    def run():
        return plan.run_indexer(q, packed_k, scales, global_scale)

    def check_output(output):
        history = q.float()[:, :, : pages - 1].topk(15, dim=-1).indices
        expected = (
            torch.cat((history, torch.full_like(history[:, :, :1], pages - 1)), dim=-1)
            .permute(1, 0, 2)
            .int()
        )
        torch.testing.assert_close(
            output.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
        )
        assert torch.all(output[:, :, -1] == pages - 1)

    output = run()
    check_output(output)
    saved_output = output.clone()
    for _ in range(2):
        updated_q = q.float()
        updated_q[:, :, :pages] = updated_q[:, :, :pages].roll(5, dims=-1)
        q.copy_(updated_q.to(q.dtype))
        check_output(run())
        torch.testing.assert_close(output, saved_output, rtol=0, atol=0)


def _dequant_rows(packed, scale, global_scale):
    """E2M1 values times their E4M3 group scale and the global scale, saturated to FP8."""
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    table = torch.cat((magnitudes, -magnitudes))
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    group_scale = scale.view(torch.float8_e4m3fn).float() * global_scale
    values = table[codes].view(-1, 8, 16) * group_scale[..., None]
    return values.clamp(-448, 448).flatten(-2).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("global_scale", [1.0, 0.37])
def test_full_page_dequant_writes_only_live_tokens(global_scale):
    torch.manual_seed(0)
    pages, max_pages = 64, 8192
    packed = torch.randint(0, 256, (pages, PAGE, 64), dtype=torch.uint8, device="cuda")
    scale = torch.randint(0, 0x7F, (pages, PAGE, 8), dtype=torch.uint8, device="cuda")
    global_scale = torch.tensor([global_scale], device="cuda")
    page_table = torch.zeros(6, max_pages, dtype=torch.int32, device="cuda")
    columns = torch.arange(max_pages, device="cuda")
    lengths = torch.zeros(6, dtype=torch.int32, device="cuda")
    out = torch.empty(pages, PAGE, 128, dtype=torch.uint8, device="cuda")

    def run():
        dequant_nvfp4_page_table(
            packed,
            scale.view(torch.float8_e4m3fn),
            global_scale,
            page_table,
            lengths,
            out.view(torch.float8_e4m3fn),
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for values in (
        [0, 1, PAGE + 1, 3 * PAGE, -5, max_pages * PAGE + 7],
        [700_000, 2, 0, 0, 33, PAGE],
    ):
        lengths.copy_(torch.tensor(values))
        live = lengths.clamp(0, max_pages * PAGE)
        # Only the columns past each request's live tokens point at page 0.
        page_table.random_(1, pages)
        page_table[columns[None, :] >= (live[:, None] + PAGE - 1) // PAGE] = 0
        out.fill_(0x5A)
        graph.replay()
        rows = []
        for b, n in enumerate(live.tolist()):
            tokens = torch.arange(n, device="cuda")
            rows.append(page_table[b, tokens // PAGE].long() * PAGE + tokens % PAGE)
        rows = torch.cat(rows).unique()
        expected = torch.full_like(out, 0x5A).view(-1, 128)
        expected[rows] = _dequant_rows(
            packed.view(-1, 64)[rows], scale.view(-1, 8)[rows], global_scale
        ).view(torch.uint8)
        torch.testing.assert_close(out.view(-1, 128), expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
