"""gfx950 prefill paths must select and attend exactly as the reference kernels do."""

import sys

import pytest
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import Nvfp4Cache
from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4_gfx950 import (
    nvfp4_buffer_loadable,
    nvfp4_even_odd,
    nvfp4_even_odd_w,
)
from sglang.kernels.ops.attention.minimax_sparse.common.utils import get_cu_seqblocks
from sglang.kernels.ops.attention.minimax_sparse.prefill import (
    flash_with_topk_idx as indexer,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill import (
    gluon_prefill,
    gluon_prefill_pair_gfx950,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill import (
    index_topk_gfx950 as fused_indexer,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import msa_v1_page_row_bytes
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.quant_ref_utils import nvfp4_kv_quantize

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn
E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
BLOCK = 128
TOPK = 16
NUM_Q_HEADS = 16  # 64 heads at tp4, one KV head per rank

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_gfx95_supported()),
    reason="gfx950-only prefill paths",
)


def _extend(prefix: int, q_len: int):
    seq = prefix + q_len
    torch.manual_seed(0)
    batch = dict(
        r2t=torch.arange(1, seq + 1, device=DEVICE, dtype=torch.int32)[None],
        slot_ids=torch.zeros(1, device=DEVICE, dtype=torch.int64),
        cu=torch.tensor([0, q_len], device=DEVICE, dtype=torch.int32),
        seq_lens=torch.tensor([seq], device=DEVICE, dtype=torch.int32),
        prefix=torch.tensor([prefix], device=DEVICE, dtype=torch.int32),
    )
    cu_b, max_b, all_b, *_ = get_cu_seqblocks(batch["cu"], q_len, 1, BLOCK, [q_len])
    batch.update(cu_b=cu_b, max_b=max_b, all_b=all_b)
    return seq, batch


def _nvfp4_dequant(payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[rows, D/2] e2m1 pairs and [rows, D/16] E4M3 scales -> exact [rows, D] fp32."""
    lut = torch.tensor(E2M1_MAG + [-m for m in E2M1_MAG], device=DEVICE)
    codes = torch.stack([payload & 0xF, payload >> 4], dim=-1).flatten(1)
    return lut[codes.long()] * scales.view(FP8).float().repeat_interleave(16, dim=1)


def _topk(q_len: int, seq: int, b: dict, init: int = 0, local: int = 1) -> torch.Tensor:
    idx_k = (torch.randn(seq + 256, 1, 128, device=DEVICE) * 0.5).to(FP8)
    idx_q = torch.randn(q_len, 1, 128, device=DEVICE, dtype=torch.bfloat16)
    _, topk = indexer.flash_prefill_with_topk_index(
        idx_q, idx_k, None, None, b["r2t"], b["slot_ids"], b["cu"], b["seq_lens"],
        b["prefix"], q_len, seq, 1, BLOCK, TOPK, init, local, score_type="max",
        disable_index_value=True, cu_seqblocks_q=b["cu_b"], max_seqblock_q=b["max_b"],
        all_seqblock_q=b["all_b"], q_scale=1.0, k_scale=1.0, page_size=BLOCK,
    )  # fmt: skip
    return topk


@pytest.mark.parametrize(
    "prefix,q_len", [(20000, 300), (9000, 1500), (0, 4096), (70000, 64)]
)
def test_split_context_indexer_selects_the_same_blocks(monkeypatch, prefix, q_len):
    """Splitting the context across programs in the fused index top-k must not change
    any query's top-k."""
    seq, b = _extend(prefix, q_len)
    split = _topk(q_len, seq, b)
    monkeypatch.setattr(fused_indexer, "_TARGET_PROGRAMS", 1)
    torch.manual_seed(0)
    unsplit = _topk(q_len, seq, b)
    assert torch.equal(split.sort(-1).values, unsplit.sort(-1).values)


@pytest.mark.parametrize(
    "prefix,q_len,init,local",
    [
        (20000, 300, 0, 1),
        (128 * 40 + 7, 1500, 0, 1),
        (0, 4096, 2, 1),
        (70000, 64, 1, 3),
        (3000, 1000, 0, 0),
        (12000, 4100, 1, 2),  # 256-query tiles
    ],
)
def test_fused_indexer_matches_two_pass_selection(
    monkeypatch, prefix, q_len, init, local
):
    """The fused score+top-k path must emit exactly the two-pass kernels' ids and layout."""
    seq, b = _extend(prefix, q_len)
    fused = _topk(q_len, seq, b, init, local)
    monkeypatch.setattr(indexer, "_use_fused_index_topk", lambda *a, **kw: False)
    torch.manual_seed(0)
    two_pass = _topk(q_len, seq, b, init, local)
    assert torch.equal(fused, two_pass)


@gluon.jit
def _token_rows(layout: gl.constexpr, R: gl.constexpr, ROW: gl.constexpr):
    """Offsets of a [R, 128] tile (row elements x tokens) in token rows of ROW."""
    r = gl.arange(0, R, layout=gl.SliceLayout(1, layout))
    t = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    return t[None, :] * ROW + r[:, None]


@gluon.jit
def _token_rows_w(layout: gl.constexpr, R: gl.constexpr, ROW: gl.constexpr):
    """`_token_rows` as 4 warps x 32 tokens, [4, R, 32]."""
    w = gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    r = gl.arange(0, R, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    t = gl.arange(0, 32, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    return (w[:, None, None] * 32 + t[None, None, :]) * ROW + r[None, :, None]


@gluon.jit
def _nvfp4_tiles_kernel(
    payload_ptr, scale_ptr, even_ptr, odd_ptr, PER_WARP: gl.constexpr
):
    """Dequantize 128 tokens into token-major even and odd dims, [128, 64] each, with
    the prefill tile helper or, as 4 warps x 32 tokens, the per-warp one."""
    if PER_WARP:
        lay: gl.constexpr = gl.BlockedLayout(
            [1, 4, 1], [1, 4, 16], [4, 1, 1], [2, 1, 0]
        )
        grp: gl.constexpr = gl.BlockedLayout(
            [1, 1, 4, 1], [1, 4, 1, 16], [4, 1, 1, 1], [3, 2, 1, 0]
        )
        payload = gl.load(payload_ptr + _token_rows_w(lay, 32, 32))
        scales = gl.load(scale_ptr + _token_rows_w(gl.SliceLayout(2, grp), 8, 8))
        even, odd = nvfp4_even_odd_w(payload, scales, grp)
        out = _token_rows_w(lay, 64, 64)
    else:
        lay: gl.constexpr = gl.BlockedLayout([8, 1], [4, 16], [1, 4], [0, 1])
        grp: gl.constexpr = gl.BlockedLayout(
            [2, 4, 1], [4, 1, 16], [1, 1, 4], [1, 0, 2]
        )
        payload = gl.load(payload_ptr + _token_rows(lay, 32, 32))
        scales = gl.load(scale_ptr + _token_rows(gl.SliceLayout(1, grp), 8, 8))
        even, odd = nvfp4_even_odd(payload, scales, grp)
        out = _token_rows(lay, 64, 64)
    gl.store(even_ptr + out, gl.convert_layout(even, lay))
    gl.store(odd_ptr + out, gl.convert_layout(odd, lay))


@pytest.mark.parametrize("per_warp", [False, True])
def test_nvfp4_tiles_equal_reference_dequant(per_warp):
    """The in-kernel NVFP4 dequant must give the exact e2m1 x E4M3 values bit for bit,
    every e2m1 code and every finite positive E4M3 scale included."""
    torch.manual_seed(0)
    payload = torch.randint(0, 256, (128, 64), dtype=torch.uint8, device=DEVICE)
    scales = torch.randint(0, 0x7F, (128, 8), dtype=torch.uint8, device=DEVICE)
    even = torch.empty(128, 64, dtype=torch.float16, device=DEVICE)
    odd = torch.empty_like(even)
    _nvfp4_tiles_kernel[(1,)](
        payload.view(torch.int16), scales, even, odd, PER_WARP=per_warp, num_warps=4
    )
    ref = _nvfp4_dequant(payload, scales).half()
    assert torch.equal(even.view(torch.int16), ref[:, 0::2].view(torch.int16))
    assert torch.equal(odd.view(torch.int16), ref[:, 1::2].view(torch.int16))


def test_page_row_views_past_32_bit_offsets_are_not_buffer_loadable():
    """A pool page-row view spans more bytes than it holds; the bound is on the span."""
    row_bytes, _ = msa_v1_page_row_bytes(
        kv_heads=1, head_dim=128, index_head_dim=128, page_size=BLOCK
    )

    def main_k(num_pages: int) -> Nvfp4Cache:
        rows = torch.empty((num_pages, row_bytes), dtype=torch.uint8, device="meta")
        payload = rows[:, : BLOCK * 64].view(num_pages, 1, BLOCK, 64)
        scales = rows[:, BLOCK * 64 : BLOCK * 72].view(num_pages, 1, BLOCK, 8)
        return Nvfp4Cache(payload=payload, scales=scales)

    assert nvfp4_buffer_loadable(main_k(100_000))
    past = main_k(130_000)
    assert past.payload.numel() < 2**31
    assert not nvfp4_buffer_loadable(past)


def _nvfp4_extend(prefix: int, q_len: int):
    """An extend batch over an NVFP4 index-K pool with shuffled pages."""
    seq = prefix + q_len
    torch.manual_seed(0)
    num_pages = -(-seq // BLOCK)
    pages = torch.randperm(num_pages, device=DEVICE) + 1
    r2t = (pages[:, None] * BLOCK + torch.arange(BLOCK, device=DEVICE)).view(1, -1)
    batch = dict(
        r2t=r2t[:, :seq].to(torch.int32).contiguous(),
        slot_ids=torch.zeros(1, device=DEVICE, dtype=torch.int64),
        cu=torch.tensor([0, q_len], device=DEVICE, dtype=torch.int32),
        seq_lens=torch.tensor([seq], device=DEVICE, dtype=torch.int32),
        prefix=torch.tensor([prefix], device=DEVICE, dtype=torch.int32),
    )
    cu_b, max_b, all_b, *_ = get_cu_seqblocks(batch["cu"], q_len, 1, BLOCK, [q_len])
    batch.update(cu_b=cu_b, max_b=max_b, all_b=all_b)
    k = torch.randn((num_pages + 1) * BLOCK, 128, device=DEVICE, dtype=torch.bfloat16)
    payload, scales = nvfp4_kv_quantize(k, torch.ones(1, device=DEVICE))
    pool = Nvfp4Cache(
        payload=payload.view(num_pages + 1, BLOCK, 64),
        scales=scales.view(num_pages + 1, BLOCK, 8),
    )
    return seq, batch, pool


@pytest.mark.parametrize(
    "prefix,q_len,init,local",
    [
        (20000, 300, 0, 1),
        (128 * 40, 1500, 0, 1),
        (0, 4096, 2, 1),
        (70000, 64, 1, 3),
        (12000, 4100, 1, 2),  # 256-query tiles
    ],
)
def test_fused_indexer_on_nvfp4_matches_two_pass_selection(
    monkeypatch, prefix, q_len, init, local
):
    """On an NVFP4 index-K pool the fused path must select what the score-only kernel
    + top-k select on the same values in bf16; rows may differ only among blocks whose
    scores tie up to the summation order."""
    seq, b, pool = _nvfp4_extend(prefix, q_len)
    idx_q = torch.randn(q_len, 1, 128, device=DEVICE, dtype=torch.bfloat16)
    # e2m1 x E4M3 carries at most 6 significant bits, so bf16 holds it exactly
    dense = _nvfp4_dequant(pool.payload.flatten(0, 1), pool.scales.flatten(0, 1))
    dense = dense.bfloat16()

    def topk_ids(k_cache):
        _, ids = indexer.flash_prefill_with_topk_index(
            idx_q, k_cache, None, None, b["r2t"], b["slot_ids"], b["cu"],
            b["seq_lens"], b["prefix"], q_len, seq, 1, BLOCK, TOPK, init, local,
            score_type="max", disable_index_value=True, cu_seqblocks_q=b["cu_b"],
            max_seqblock_q=b["max_b"], all_seqblock_q=b["all_b"], page_size=BLOCK,
        )  # fmt: skip
        return ids[0]

    assert fused_indexer.can_fuse_index_topk(idx_q, pool, TOPK, local, BLOCK, BLOCK)
    fused = topk_ids(pool)
    monkeypatch.setattr(indexer, "_use_fused_index_topk", lambda *a, **kw: False)
    two_pass = topk_ids(dense[:, None, :])
    # block-max scores of each query over the keys it sees, in token order
    keys = dense[b["r2t"][0, :seq].long()].float()
    logits = idx_q[:, 0].float() @ keys.T
    visible = torch.arange(seq, device=DEVICE)[None, :] <= (
        prefix + torch.arange(q_len, device=DEVICE)[:, None]
    )
    logits = logits.masked_fill(~visible, float("-inf"))
    num_blocks = -(-seq // BLOCK)
    logits = torch.nn.functional.pad(
        logits, (0, num_blocks * BLOCK - seq), value=float("-inf")
    )
    scores = logits.view(q_len, num_blocks, BLOCK).amax(-1)
    for row in (fused != two_pass).any(-1).nonzero().flatten().tolist():
        picked = [ids[row][ids[row] >= 0].long() for ids in (fused, two_pass)]
        tied = [scores[row, p].sort().values for p in picked]
        torch.testing.assert_close(*tied, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    gluon_prefill.pa_decode_gluon is None, reason="aiter Gluon paged attention"
)
@pytest.mark.parametrize("prefix,q_len", [(20000, 300), (0, 2048)])
@pytest.mark.parametrize("kv_scale", [1.0, 0.5])
@pytest.mark.parametrize("pair_min_tokens", [1 << 30, 1], ids=["decode", "pair"])
def test_gluon_prefill_on_fp8_kv_matches_triton(
    monkeypatch, prefix, q_len, kv_scale, pair_min_tokens
):
    """Gluon on an fp8 pool must attend like the Triton kernel, scales included."""
    monkeypatch.setattr(
        gluon_prefill_pair_gfx950, "PAIR_MIN_TOKENS_FP8", pair_min_tokens
    )
    seq, b = _extend(prefix, q_len)
    topk = _topk(q_len, seq, b)
    q = torch.randn(q_len, NUM_Q_HEADS, 128, device=DEVICE, dtype=torch.bfloat16)
    k = (torch.randn(seq + 256, 1, 128, device=DEVICE) * 2).to(FP8)
    v = (torch.randn(seq + 256, 1, 128, device=DEVICE) * 2).to(FP8)
    seq_lens_cpu = torch.tensor([seq])

    assert gluon_prefill.can_use_gluon_prefill(
        q, k, v, None, BLOCK, seq_lens_cpu, None, kv_scale, kv_scale
    )
    out = gluon_prefill.gluon_sparse_prefill(
        q, k, v, topk, b["r2t"], b["slot_ids"], b["cu"], b["seq_lens"], b["prefix"],
        seq_lens_cpu, BLOCK, k_scale=kv_scale, v_scale=kv_scale,
    ).float()  # fmt: skip
    ref = flash_prefill_with_gqa_share_sparse(
        q, k, v, None, b["r2t"], b["slot_ids"], topk, 1, BLOCK, b["cu"], b["seq_lens"],
        b["prefix"], q_len, cu_seqblocks_q=b["cu_b"], max_seqblock_q=b["max_b"],
        k_scale=kv_scale, v_scale=kv_scale,
    ).float()  # fmt: skip
    # both widen the same fp8 values to bf16; outputs differ by bf16 rounding
    torch.testing.assert_close(out, ref, atol=4e-2 * kv_scale, rtol=2e-2)
    assert (out - ref).abs().mean() < 1e-3


@pytest.mark.skipif(
    gluon_prefill.pa_decode_gluon is None, reason="aiter Gluon paged attention"
)
@pytest.mark.parametrize("prefix,q_len", [(20000, 300), (0, 2048)])
@pytest.mark.parametrize("pair_min_tokens", [1 << 30, 1], ids=["decode", "pair"])
def test_gluon_prefill_on_nvfp4_kv_matches_triton(
    monkeypatch, prefix, q_len, pair_min_tokens
):
    """Gluon on an NVFP4 pool must attend like the Triton kernel on its bf16 values."""
    monkeypatch.setattr(
        gluon_prefill_pair_gfx950, "PAIR_MIN_TOKENS_BF16", pair_min_tokens
    )
    seq, b, idx_pool = _nvfp4_extend(prefix, q_len)
    rows = idx_pool.payload.shape[0] * BLOCK
    idx_q = torch.randn(q_len, 1, 128, device=DEVICE, dtype=torch.bfloat16)
    _, topk = indexer.flash_prefill_with_topk_index(
        idx_q, idx_pool, None, None, b["r2t"], b["slot_ids"], b["cu"], b["seq_lens"],
        b["prefix"], q_len, seq, 1, BLOCK, TOPK, 0, 1, score_type="max",
        disable_index_value=True, cu_seqblocks_q=b["cu_b"], max_seqblock_q=b["max_b"],
        all_seqblock_q=b["all_b"], page_size=BLOCK,
    )  # fmt: skip
    one = torch.ones(1, device=DEVICE)
    caches, dense = [], []
    for _ in range(2):
        x = torch.randn(rows, 128, device=DEVICE, dtype=torch.bfloat16) * 2
        payload, scales = nvfp4_kv_quantize(x, one)
        caches.append(
            Nvfp4Cache(
                payload=payload.view(-1, 1, BLOCK, 64),
                scales=scales.view(-1, 1, BLOCK, 8),
            )
        )
        dense.append(_nvfp4_dequant(payload, scales).bfloat16()[:, None, :])
    q = torch.randn(q_len, NUM_Q_HEADS, 128, device=DEVICE, dtype=torch.bfloat16)
    seq_lens_cpu = torch.tensor([seq])

    assert gluon_prefill.can_use_gluon_prefill(
        q, caches[0], caches[1], None, BLOCK, seq_lens_cpu
    )
    out = gluon_prefill.gluon_sparse_prefill(
        q, caches[0], caches[1], topk, b["r2t"], b["slot_ids"], b["cu"], b["seq_lens"],
        b["prefix"], seq_lens_cpu, BLOCK,
    ).float()  # fmt: skip
    ref = flash_prefill_with_gqa_share_sparse(
        q, dense[0], dense[1], None, b["r2t"], b["slot_ids"], topk, 1, BLOCK, b["cu"],
        b["seq_lens"], b["prefix"], q_len, cu_seqblocks_q=b["cu_b"],
        max_seqblock_q=b["max_b"],
    ).float()  # fmt: skip
    # the bf16 values are exact; outputs differ by bf16 rounding
    torch.testing.assert_close(out, ref, atol=4e-2, rtol=2e-2)
    assert (out - ref).abs().mean() < 1e-3


@pytest.mark.skipif(
    gluon_prefill.pa_decode_gluon is None, reason="aiter Gluon paged attention"
)
def test_pair_prefill_across_requests_matches_triton(monkeypatch):
    """Query pairs that straddle requests must read each query's own request pages."""
    monkeypatch.setattr(gluon_prefill_pair_gfx950, "PAIR_MIN_TOKENS_FP8", 1)
    torch.manual_seed(0)
    reqs = [(0, 301), (5000, 155), (777, 512)]
    seqs = [p + n for p, n in reqs]
    max_kv = max(seqs) + 256
    r2t = torch.zeros(len(reqs), max_kv, device=DEVICE, dtype=torch.int32)
    for i, s in enumerate(seqs):
        r2t[i, :s] = torch.arange(1, s + 1, device=DEVICE) + i * max_kv
    lens = [n for _, n in reqs]
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], device=DEVICE, dtype=torch.int32
    )
    prefix = torch.tensor([p for p, _ in reqs], device=DEVICE, dtype=torch.int32)
    seq_lens = torch.tensor(seqs, device=DEVICE, dtype=torch.int32)
    slot_ids = torch.arange(len(reqs), device=DEVICE)
    total, slots = sum(lens), len(reqs) * max_kv + 1
    cu_b, max_b, all_b, *_ = get_cu_seqblocks(cu, max(lens), 1, BLOCK, lens)
    idx_k = (torch.randn(slots, 1, 128, device=DEVICE) * 0.5).to(FP8)
    idx_q = torch.randn(total, 1, 128, device=DEVICE, dtype=torch.bfloat16)
    _, topk = indexer.flash_prefill_with_topk_index(
        idx_q, idx_k, None, None, r2t, slot_ids, cu, seq_lens, prefix, max(lens),
        max(seqs), 1, BLOCK, TOPK, 0, 1, score_type="max", disable_index_value=True,
        cu_seqblocks_q=cu_b, max_seqblock_q=max_b, all_seqblock_q=all_b,
        page_size=BLOCK,
    )  # fmt: skip
    q = torch.randn(total, NUM_Q_HEADS, 128, device=DEVICE, dtype=torch.bfloat16)
    k = (torch.randn(slots, 1, 128, device=DEVICE) * 2).to(FP8)
    v = (torch.randn(slots, 1, 128, device=DEVICE) * 2).to(FP8)
    out = gluon_prefill.gluon_sparse_prefill(
        q, k, v, topk, r2t, slot_ids, cu, seq_lens, prefix, seq_lens.cpu(), BLOCK,
    ).float()  # fmt: skip
    ref = flash_prefill_with_gqa_share_sparse(
        q, k, v, None, r2t, slot_ids, topk, 1, BLOCK, cu, seq_lens, prefix, max(lens),
        cu_seqblocks_q=cu_b, max_seqblock_q=max_b,
    ).float()  # fmt: skip
    torch.testing.assert_close(out, ref, atol=4e-2, rtol=2e-2)
    assert (out - ref).abs().mean() < 1e-3


@pytest.mark.parametrize("kv_scale", [1.0, 0.5])
def test_pair_prefill_fp8_out_equals_o_proj_quant(monkeypatch, kv_scale):
    """The pair kernel's FP8 O must be the bytes and column-major scales the o_proj's
    separate per 1 x 128 quant writes for its bf16 O, which the block GEMM reads."""
    from aiter import dtypes

    from sglang.srt.layers.quantization.fp8_utils import (
        aiter_per1x128_quant,
        view_aiter_fused_rms_transposed_fp8_scale,
    )

    monkeypatch.setattr(gluon_prefill_pair_gfx950, "PAIR_MIN_TOKENS_FP8", 1)
    prefix, q_len = 9000, 1501
    seq, b = _extend(prefix, q_len)
    topk = _topk(q_len, seq, b)
    q = torch.randn(q_len, NUM_Q_HEADS, 128, device=DEVICE, dtype=torch.bfloat16)
    k = (torch.randn(seq + 256, 1, 128, device=DEVICE) * 2).to(FP8)
    v = (torch.randn(seq + 256, 1, 128, device=DEVICE) * 2).to(FP8)

    def run(out_quant):
        return gluon_prefill.gluon_sparse_prefill(
            q, k, v, topk, b["r2t"], b["slot_ids"], b["cu"], b["seq_lens"],
            b["prefix"], torch.tensor([seq]), BLOCK, k_scale=kv_scale,
            v_scale=kv_scale, out_quant=out_quant,
        )  # fmt: skip

    q8, scale = run("fp8")
    ref_q8, ref_scale = aiter_per1x128_quant(
        run("").view(q_len, -1), quant_dtype=dtypes.fp8, transpose_scale=True
    )
    ref_scale = view_aiter_fused_rms_transposed_fp8_scale(ref_scale)
    assert torch.equal(q8.view(torch.uint8), ref_q8.view(torch.uint8))
    assert torch.equal(scale, ref_scale) and scale.stride() == ref_scale.stride()


def test_gluon_prefill_meta_does_not_wait_for_queued_work():
    """The per-forward Gluon page layout is staged from host lists; on ROCm a
    pageable host-to-device copy would block until the stream drains."""
    cu = torch.tensor([0, 3, 8], device=DEVICE, dtype=torch.int32)
    prefix = torch.tensor([5, 300], device=DEVICE, dtype=torch.int32)
    seq_lens = prefix + torch.tensor([3, 5], device=DEVICE, dtype=torch.int32)
    seq_lens_cpu = seq_lens.cpu()
    # Same shapes once, so first-use kernel loading is not inside the timed call.
    gluon_prefill._build_gluon_prefill_meta(
        cu.clone(), prefix.clone(), seq_lens.clone(), seq_lens_cpu, 8
    )
    torch.cuda.synchronize()

    torch.cuda._sleep(200_000_000)  # ~0.1 s of work queued ahead
    queued_done = torch.cuda.Event()
    queued_done.record()
    meta = gluon_prefill._build_gluon_prefill_meta(
        cu, prefix, seq_lens, seq_lens_cpu, 8
    )
    assert not queued_done.query()

    req_id, abs_pos, page_start, page_req, page_pos, total_pages = meta
    pages = [2 * ((n + BLOCK - 1) // BLOCK) for n in seq_lens_cpu.tolist()]
    assert total_pages == sum(pages)
    assert page_start.tolist() == [0, pages[0]]
    assert page_req.tolist() == [0] * pages[0] + [1] * pages[1]
    assert page_pos.tolist() == list(range(pages[0])) + list(range(pages[1]))
    assert req_id.tolist() == [0] * 3 + [1] * 5
    assert abs_pos.tolist() == [5, 6, 7] + list(range(300, 305))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
