import pytest
import torch

from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.srt.layers.attention.qsa.fused_kv import fused_kv_prepare
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("requests", [1, 3, 8, 128])
@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("padding", [False, True])
def test_fused_kv_prepare(requests, width, dtype, padding):
    torch.manual_seed(23)
    rows, heads, dim, length, topk, stride = requests * width, 2, 256, 4096, 2051, 2112
    mapping = (torch.randperm(requests * length, device="cuda") + 1).view(
        requests, length
    )
    reqids = torch.randperm(requests, device="cuda").repeat_interleave(width).int()
    lens = torch.empty(rows, device="cuda", dtype=torch.int32)
    loc = torch.empty(rows, device="cuda", dtype=torch.int64)
    indices = torch.full((rows, topk), -1, device="cuda", dtype=torch.int32)
    for row in range(rows):
        base = [23, 2030, 3070][(row // width) % 3]
        pos = base + row % width
        lens[row] = pos + 1
        loc[row] = mapping[reqids[row], pos]
        old = torch.randperm(base, device="cuda")[: topk - width]
        chosen = torch.cat((old, torch.arange(base, pos + 1, device="cuda")))
        indices[row, : chosen.numel()] = chosen.int()
    if padding:
        loc[-width:] = 0
        lens[-width:] = 1
        indices[-width:] = -1
        indices[-width:, 0] = 0
    raw = torch.randn(rows, heads * dim * 3, device="cuda", dtype=torch.bfloat16)
    nk = raw[:, : heads * dim].view(rows, heads, dim).to(dtype)
    nv = raw[:, heads * dim : heads * dim * 2].view(rows, heads, dim).to(dtype)
    kc = torch.randn(requests * length + 1, heads, dim, device="cuda").to(dtype)
    vc = torch.randn_like(kc.float()).to(dtype)
    counts = torch.empty(rows, device="cuda", dtype=torch.int32)
    ok = torch.empty(rows * stride, heads, dim, device="cuda", dtype=torch.bfloat16)
    ov = torch.empty_like(ok)
    cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32) * stride
    store_cache(
        nk.view(rows, -1),
        nv.view(rows, -1),
        kc.view(-1, heads * dim),
        vc.view(-1, heads * dim),
        loc,
    )
    qwen_sparse_valid_counts_triton(lens, indices, counts, rows, topk)
    qwen_sparse_kv_extraction_compact_triton(
        kc,
        vc,
        mapping,
        reqids,
        indices,
        lens,
        cu,
        ok,
        ov,
        rows,
        topk,
        zero_fill_cols=stride,
    )
    expected = [x.clone() for x in [kc, vc, counts, ok, ov]]
    validloc = loc[loc != 0]
    kc[validloc] = torch.full_like(kc[validloc], float("nan"))
    vc[validloc] = torch.full_like(vc[validloc], float("nan"))

    def candidate():
        fused_kv_prepare(
            kc, vc, nk, nv, loc, mapping, reqids, indices, lens, counts, ok, ov, width
        )

    candidate()

    def verify():
        for actual, ref in zip([kc, vc, counts], expected[:3]):
            torch.testing.assert_close(actual.float(), ref.float(), rtol=0, atol=0)
        validrows = rows - width if padding else rows
        for actual, ref in zip([ok, ov], expected[3:]):
            torch.testing.assert_close(
                actual[: validrows * stride], ref[: validrows * stride], rtol=0, atol=0
            )
            if padding:
                assert torch.count_nonzero(actual[validrows * stride :]).item() == 0

    verify()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        candidate()
    for _ in range(10):
        g.replay()
    verify()


@pytest.mark.parametrize("width", [1, 4])
def test_qsa_pool_routes_to_fused_prepare(monkeypatch, width):
    from types import SimpleNamespace

    from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        UnquantizedKVCacheMethod,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    storage = MHATokenToKVPool.__new__(MHATokenToKVPool)
    storage.quant_method = UnquantizedKVCacheMethod()
    pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
    pool.full_kv_pool = storage
    cache = torch.zeros(64, 2, 256, device="cuda", dtype=torch.bfloat16)
    pool.get_key_buffer = lambda layer_id: cache
    pool.get_value_buffer = lambda layer_id: cache.clone()
    backend = backend_module.QwenSparseAttnBackend.__new__(
        backend_module.QwenSparseAttnBackend
    )
    backend.token_to_kv_pool = pool
    metadata = SimpleNamespace(sequence_lengths=torch.ones(width, device="cuda"))
    backend._resolve_metadata = lambda batch: metadata
    marker = object()
    captured = {}

    def attention(*args, **kwargs):
        captured.update(kwargs)
        return marker

    backend._forward_trtllm_sparse = attention
    monkeypatch.setattr(
        backend_module, "_resolve_trtllm_sparse_decode", lambda: attention
    )
    q = torch.zeros(width, 8, 256, device="cuda", dtype=torch.bfloat16)
    kv = q[:, :2]
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE if width == 1 else ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(draft_token_num=width),
        out_cache_loc=torch.arange(1, width + 1, device="cuda"),
        req_pool_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
    )
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=8, head_dim=256)
    indices = torch.zeros(width, 8, device="cuda", dtype=torch.int32)
    assert backend._try_fused_kv_attention(q, kv, kv, layer, batch, indices) is marker
    assert captured["new_kv"][3] == width


@pytest.mark.parametrize("requests", [1, 9, 128])
@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("chain_positions", [False, True])
def test_fused_block_expansion(requests, width, dtype, padding, chain_positions):
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

    torch.manual_seed(42)
    rows, length, stride = requests * width, 9216, 2112
    mapping = (torch.randperm(requests * length, device="cuda") + 1).view(
        requests, length
    )
    req = torch.randperm(requests, device="cuda").repeat_interleave(width).int()
    bases = [1, 2, 3, 4, 5, 2047, 2048, 2049, 8192]
    lens = torch.tensor(
        [bases[r // width % len(bases)] + r % width for r in range(rows)],
        device="cuda",
        dtype=torch.int32,
    )
    positions = lens - 1
    loc = mapping[req.long(), positions.long()].contiguous()
    blocks = torch.full((rows, 512), -1, device="cuda", dtype=torch.int32)
    for row in range(rows):
        n = int(lens[row]) // 4
        chosen = torch.randperm(n, device="cuda")[:512]
        blocks[row, : chosen.numel()] = chosen.int()
    if padding:
        loc[-width:] = 0
        lens[-width:] = 1
        positions[-width:] = 0
        blocks[-width:] = -1
    indices = expand_qsa_block_indices(blocks, positions, lens, 4, 2048)
    k = torch.randn(
        requests * length + 1, 1, 256, device="cuda", dtype=torch.bfloat16
    ).to(dtype)
    v = torch.randn_like(k.float()).to(dtype)
    nk = torch.randn(rows, 1, 256, device="cuda", dtype=torch.bfloat16).to(dtype)
    nv = torch.randn_like(nk.float()).to(dtype)
    counts = torch.empty(rows, device="cuda", dtype=torch.int32)
    ok = torch.empty(rows * stride, 1, 256, device="cuda", dtype=torch.bfloat16)
    ov = torch.empty_like(ok)
    fused_kv_prepare(
        k, v, nk, nv, loc, mapping, req, indices, lens, counts, ok, ov, width
    )
    expected = [t.clone() for t in [k, v, counts, ok, ov]]
    active = loc[loc != 0]
    k[active] = torch.full_like(k[active], float("nan"))
    v[active] = torch.full_like(v[active], float("nan"))

    def candidate():
        fused_kv_prepare(
            k,
            v,
            nk,
            nv,
            loc,
            mapping,
            req,
            blocks,
            lens,
            counts,
            ok,
            ov,
            width,
            compress_ratio=4,
            query_positions=positions,
            chain_positions=chain_positions,
        )

    candidate()
    for actual, ref in zip([k, v, counts, ok, ov], expected):
        torch.testing.assert_close(actual.float(), ref.float(), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for _ in range(10):
        graph.replay()
    for actual, ref in zip([k, v, counts, ok, ov], expected):
        torch.testing.assert_close(actual.float(), ref.float(), rtol=0, atol=0)


def test_block_indices_expand_on_attention_fallback(monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.layers.attention import qwen_sparse_attn_backend as module
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

    backend = module.QwenSparseAttnBackend.__new__(module.QwenSparseAttnBackend)
    backend.qsa_profile = SimpleNamespace(block_topk=512, budget=2048)
    backend.compress_ratio = 4
    cache = torch.zeros(16, 1, 256, device="cuda", dtype=torch.bfloat16)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _: cache, get_value_buffer=lambda _: cache
    )
    lengths = torch.tensor([11], device="cuda", dtype=torch.int32)
    positions = lengths - 1
    metadata = SimpleNamespace(
        indexer_metadata=SimpleNamespace(
            decode_logical_positions=positions, get_seqlens_int32=lambda: lengths
        )
    )
    backend._resolve_metadata = lambda _: metadata
    blocks = torch.full((1, 512), -1, device="cuda", dtype=torch.int32)
    blocks[0, :2] = torch.tensor([1, 0], device="cuda")
    expected = expand_qsa_block_indices(blocks, positions, lengths, 4, 2048)
    marker = object()

    def attention(q, k, v, layer, batch, metadata, indices, trtllm):
        torch.testing.assert_close(indices, expected, rtol=0, atol=0)
        return marker

    backend._forward_trtllm_sparse = attention
    monkeypatch.setattr(module, "_resolve_trtllm_sparse_decode", lambda: attention)
    q = torch.zeros(1, 8, 256, device="cuda", dtype=torch.bfloat16)
    assert (
        backend._forward_paged_attention(q, SimpleNamespace(layer_id=0), None, blocks)
        is marker
    )


def test_nonconsecutive_writes_keep_slot_matching():
    from sglang.srt.layers.attention.qsa.kernel import expand_qsa_block_indices

    torch.manual_seed(42)
    rows, stride = 4, 2112
    lens = torch.tensor([6, 11, 15, 21], device="cuda", dtype=torch.int32)
    positions = lens - 1
    mapping = (torch.randperm(64, device="cuda") + 1).view(1, 64)
    req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    loc = mapping[0, positions.long()].contiguous()
    blocks = torch.full((rows, 512), -1, device="cuda", dtype=torch.int32)
    for r in range(rows):
        n = int(lens[r]) // 4
        blocks[r, :n] = torch.randperm(n, device="cuda").int()
    indices = expand_qsa_block_indices(blocks, positions, lens, 4, 2048)
    k = torch.randn(65, 1, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    nk = torch.randn(rows, 1, 256, device="cuda", dtype=torch.bfloat16)
    nv = torch.randn_like(nk)
    counts = torch.empty(rows, device="cuda", dtype=torch.int32)
    ok = torch.empty(rows * stride, 1, 256, device="cuda", dtype=torch.bfloat16)
    ov = torch.empty_like(ok)
    fused_kv_prepare(
        k, v, nk, nv, loc, mapping, req, indices, lens, counts, ok, ov, rows
    )
    expected = [t.clone() for t in [counts, ok, ov]]
    k[loc] = float("nan")
    v[loc] = float("nan")
    fused_kv_prepare(
        k,
        v,
        nk,
        nv,
        loc,
        mapping,
        req,
        blocks,
        lens,
        counts,
        ok,
        ov,
        rows,
        compress_ratio=4,
        query_positions=positions,
    )
    for actual, ref in zip([counts, ok, ov], expected):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
