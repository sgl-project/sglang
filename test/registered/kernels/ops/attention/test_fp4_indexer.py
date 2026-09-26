from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import (
    apply_rotary_emb_triton,
    precompute_freqs_cis,
)
from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    compress_norm_rope_store,
    fused_q_indexer_rope_hadamard_fp4_quant,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    finish_paged_indexer_topk,
    fp4_index_logits_decode,
    fp4_index_logits_paged,
    quantize_fp4_indexer_tensor,
    store_fp4_index_k_cache,
)
from sglang.srt.utils import get_device, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_is_xpu = is_xpu()
if _is_xpu:
    from sgl_kernel import hadamard_transform
else:
    from sglang.kernels.ops.quantization.hadamard import hadamard_transform

HEAD_DIM = 128
FP4_DIM = HEAD_DIM // 2
GROUP_SIZE = 32
SCALE_GROUPS = HEAD_DIM // GROUP_SIZE
SCALE_BYTES = 4
PAGE_SIZE = 64
E2M1_MAX = 6.0


def _ceil_ue8m0_exp_ref(x: torch.Tensor) -> torch.Tensor:
    bits = x.to(torch.float32).contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    exp = exp + (mantissa != 0).to(torch.int32)
    return exp.clamp(1, 254)


def _fp4_e2m1_code_ref(x: torch.Tensor) -> torch.Tensor:
    ax = torch.minimum(x.abs(), torch.tensor(E2M1_MAX, device=x.device))
    idx = torch.zeros_like(ax, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        idx += (ax > threshold).to(torch.uint8)
    sign = ((x < 0) & (idx != 0)).to(torch.uint8) * 8
    return idx | sign


def _ref_quantize_fp4_indexer(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.contiguous().view(-1, HEAD_DIM).float()
    groups = x.view(-1, SCALE_GROUPS, GROUP_SIZE)
    scale_raw = (groups.abs().amax(dim=-1) / E2M1_MAX).clamp_min(1.0e-4)
    scale_exp = _ceil_ue8m0_exp_ref(scale_raw)
    scale = (scale_exp << 23).contiguous().view(torch.float32)

    scaled = (groups / scale.unsqueeze(-1)).view(-1, HEAD_DIM)
    code = _fp4_e2m1_code_ref(scaled)
    packed = (code[:, 0::2].to(torch.int16) | (code[:, 1::2].to(torch.int16) << 4)).to(
        torch.uint8
    )

    packed_sf = scale_exp[:, 0].clone()
    for group_id in range(1, SCALE_GROUPS):
        packed_sf |= scale_exp[:, group_id] << (8 * group_id)
    return packed, packed_sf


def _ref_store_fp4_index_cache(
    x_fp4: torch.Tensor,
    x_sf: torch.Tensor,
    loc: torch.Tensor,
    num_pages: int,
) -> torch.Tensor:
    expected = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=x_fp4.device,
        dtype=torch.uint8,
    )
    sf_shifts = torch.arange(0, 32, 8, device=x_fp4.device, dtype=torch.int32)
    for token_id in range(x_fp4.shape[0]):
        cache_loc = int(loc[token_id].item())
        page = cache_loc // PAGE_SIZE
        offset = cache_loc % PAGE_SIZE
        expected[page, offset * FP4_DIM : (offset + 1) * FP4_DIM] = x_fp4[token_id]
        sf_start = PAGE_SIZE * FP4_DIM + offset * SCALE_BYTES
        expected[page, sf_start : sf_start + SCALE_BYTES] = (
            (x_sf[token_id] >> sf_shifts) & 0xFF
        ).to(torch.uint8)
    return expected


@pytest.mark.parametrize("num_tokens", [1, 7, 96])
def test_quantize_fp4_indexer_tensor(num_tokens: int) -> None:
    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    x[0, :8] = torch.tensor(
        [-8.0, -6.0, -3.0, -1.5, 0.0, 0.5, 2.0, 8.0],
        device=get_device(),
        dtype=torch.bfloat16,
    )

    x_fp4, x_sf = quantize_fp4_indexer_tensor(x)
    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(x)

    torch.testing.assert_close(x_fp4.view(torch.uint8), ref_fp4)
    torch.testing.assert_close(x_sf, ref_sf)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Vectorized prefill dispatch targets Blackwell",
)
@pytest.mark.parametrize("rows", [4096, 4097, 16384, 524288])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rne", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_prefill_quantization_matches_single_row_and_replay(rows, dtype, rne, strided):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        _quantize_fp4_indexer_kernel,
    )

    x = torch.randn(rows, 256 if strided else 128, device="cuda", dtype=dtype)
    if strided:
        x = x[:, ::2]

    def reference():
        q = torch.empty(rows, 64, device="cuda", dtype=torch.int8)
        sf = torch.empty(rows, device="cuda", dtype=torch.int32)
        _quantize_fp4_indexer_kernel[(rows,)](
            x.contiguous(),
            q,
            sf,
            BLOCK_N=128,
            GROUP_N=32,
            RNE=rne,
        )
        return q, sf

    for _ in range(3):
        quantize_fp4_indexer_tensor(x, rne)
        reference()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = quantize_fp4_indexer_tensor(x, rne)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda", dtype=dtype
    )
    for scale in (0.0, 1e-6, 1.0, 1e3):
        x.normal_().mul_(scale)
        x[:2] = boundaries.repeat(16)
        x[1].neg_()
        graph.replay()
        for a, b in zip(actual, reference()):
            assert torch.equal(a, b)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="Vectorized prefill dispatch targets Blackwell",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rne", [False, True])
def test_prefill_quantization_nonfinite_group_replay(dtype, rne):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        _quantize_fp4_indexer_kernel,
    )

    rows = 4097  # Also exercise the final partial CTA.
    x = torch.randn(rows, HEAD_DIM, device="cuda", dtype=dtype)
    expected = (
        torch.empty(rows, FP4_DIM, device="cuda", dtype=torch.int8),
        torch.empty(rows, device="cuda", dtype=torch.int32),
    )
    for _ in range(3):
        quantize_fp4_indexer_tensor(x, rne)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = quantize_fp4_indexer_tensor(x, rne)
    for value in (float("inf"), float("-inf"), float("nan"), 0.0):
        x.normal_()
        x[:, 0] = value
        x[:, 32:64] = value
        x[-1, 96:] = value
        graph.replay()
        _quantize_fp4_indexer_kernel[(rows,)](
            x, *expected, BLOCK_N=HEAD_DIM, GROUP_N=GROUP_SIZE, RNE=rne
        )
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_fp4_index_cache_store_layout(num_tokens: int) -> None:
    torch.manual_seed(num_tokens)
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    x = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    loc = torch.randperm(num_pages * PAGE_SIZE, device=get_device())[:num_tokens].to(
        torch.int64
    )
    cache = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=get_device(),
        dtype=torch.uint8,
    )

    store_fp4_index_k_cache(x, cache, loc, page_size=PAGE_SIZE)

    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(x)
    expected = _ref_store_fp4_index_cache(ref_fp4, ref_sf, loc, num_pages)
    torch.testing.assert_close(cache, expected)


@pytest.mark.parametrize("num_tokens", [1, 16, 96])
def test_fp4_fused_norm_rope_store_layout(num_tokens: int) -> None:
    torch.manual_seed(num_tokens + 100)
    num_pages = max(1, (num_tokens + PAGE_SIZE - 1) // PAGE_SIZE)
    compress_ratio = 4
    kv = torch.randn(num_tokens, HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    norm_weight = torch.randn(HEAD_DIM, device=get_device(), dtype=torch.bfloat16)
    seq_lens = (
        torch.arange(1, num_tokens + 1, device=get_device(), dtype=torch.int64)
        * compress_ratio
    )
    req_pool_indices = torch.arange(num_tokens, device=get_device(), dtype=torch.int64)
    plan = CompressorDecodePlan.generate_legacy(
        compress_ratio, req_pool_indices, seq_lens
    )
    loc = torch.arange(num_tokens, device=get_device(), dtype=torch.int64)
    freqs_cis = precompute_freqs_cis(
        64, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
    ).to(get_device())
    cache = torch.zeros(
        num_pages,
        PAGE_SIZE * (FP4_DIM + SCALE_BYTES),
        device=get_device(),
        dtype=torch.uint8,
    )

    compress_norm_rope_store(
        kv.clone(),
        plan,
        norm_weight=norm_weight,
        norm_eps=1.0e-6,
        freq_cis=freqs_cis,
        out_loc=loc,
        kvcache=cache,
        page_size=PAGE_SIZE,
        use_fp4=True,
    )

    ref = kv.float()
    ref = ref * torch.rsqrt((ref * ref).sum(dim=-1, keepdim=True) / HEAD_DIM + 1.0e-6)
    ref = ref * norm_weight.float()
    freqs = torch.view_as_real(freqs_cis).flatten(-2)[
        (seq_lens - compress_ratio).long()
    ]
    rope = ref[:, 64:].reshape(num_tokens, 32, 2)
    freqs = freqs.reshape(num_tokens, 32, 2)
    rope_out = torch.empty_like(rope)
    rope_out[..., 0] = rope[..., 0] * freqs[..., 0] - rope[..., 1] * freqs[..., 1]
    rope_out[..., 1] = rope[..., 0] * freqs[..., 1] + rope[..., 1] * freqs[..., 0]
    ref[:, 64:] = rope_out.reshape(num_tokens, 64)
    ref = hadamard_transform(ref.contiguous(), scale=HEAD_DIM**-0.5)
    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(ref)

    expected = _ref_store_fp4_index_cache(
        ref_fp4,
        ref_sf,
        loc.to(torch.int64),
        num_pages,
    )
    torch.testing.assert_close(cache, expected)


@pytest.mark.skipif(
    _is_xpu,
    reason="fused_q_indexer_rope_hadamard_fp4_quant is not supported by Intel GPU",
)
@pytest.mark.parametrize("batch_size", [1, 5, 17])
def test_fp4_fused_q_indexer_rope_hadamard_quant(batch_size: int) -> None:
    torch.manual_seed(batch_size + 200)
    num_heads = 8
    rope_dim = 64
    weight_scale = HEAD_DIM**-0.5 * num_heads**-0.5
    q = torch.randn(
        batch_size, num_heads, HEAD_DIM, device=get_device(), dtype=torch.bfloat16
    )
    weight = torch.randn(
        batch_size, num_heads, device=get_device(), dtype=torch.bfloat16
    )
    positions = (
        torch.arange(batch_size, device=get_device(), dtype=torch.int32) * 7
    ) % 63
    freqs_cis = precompute_freqs_cis(rope_dim, 64, 0, 10000, 1, 32, 1).to(get_device())

    (q_fp4, q_sf), weights_out = fused_q_indexer_rope_hadamard_fp4_quant(
        q, weight, weight_scale, freqs_cis, positions
    )

    ref = q.clone()
    apply_rotary_emb_triton(ref[..., -rope_dim:], freqs_cis, positions=positions)
    ref = hadamard_transform(ref.contiguous(), scale=HEAD_DIM**-0.5)
    ref_fp4, ref_sf = _ref_quantize_fp4_indexer(ref.view(-1, HEAD_DIM))
    ref_fp4 = ref_fp4.view(batch_size, num_heads, FP4_DIM)
    ref_sf = ref_sf.view(batch_size, num_heads)

    torch.testing.assert_close(q_fp4.view(torch.uint8), ref_fp4)
    torch.testing.assert_close(q_sf, ref_sf)
    torch.testing.assert_close(weights_out.squeeze(-1), weight.float() * weight_scale)


def _make_logits_case(rows: int, heads: int, width: int, seed: int = 719):
    """Use exact dyadic inputs to isolate logits/masking from FP32 dot order."""
    torch.manual_seed(seed)
    device = get_device()
    q = (
        torch.randint(-8, 9, (rows, heads, HEAD_DIM), device=device).to(torch.bfloat16)
        / 8
    )
    weights = torch.randint(-8, 9, (rows, heads), device=device).to(torch.bfloat16) / 8
    # Six adjacent verify rows share one request, with shuffled physical pages.
    groups = (rows + 5) // 6
    padded = max(PAGE_SIZE, (width + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE)
    pages = torch.randperm(groups * padded // PAGE_SIZE, device=device)
    locations = (
        pages[:, None] * PAGE_SIZE + torch.arange(PAGE_SIZE, device=device)
    ).flatten()
    keys = (
        torch.randint(-8, 9, (groups * padded, HEAD_DIM), device=device).to(
            torch.bfloat16
        )
        / 8
    )
    table = torch.zeros(
        groups * padded // PAGE_SIZE,
        PAGE_SIZE * 68,
        device=device,
        dtype=torch.uint8,
    )
    store_fp4_index_k_cache(keys, table, locations, page_size=PAGE_SIZE, rne=True)
    slots = locations.reshape(groups, padded)[
        torch.arange(rows, device=device) // 6, :width
    ].contiguous()
    patterns = [0, 1, 63, 64, 65, max(0, width - 1), width]
    lens = torch.tensor(
        [min(width, patterns[i % len(patterns)]) for i in range(rows)],
        device=device,
        dtype=torch.int64,
    )
    lens[-1] = width
    return q, weights, slots, lens, table


def _reference_logits(q, weights, slots, lens, table):
    """Independent E2M1/UE8M0 decoding with the prescribed BF16 rounds."""
    rows, width = slots.shape
    position = torch.arange(width, device=q.device)
    # Invisible positions have no cache mapping contract.
    safe_slots = torch.where(position[None, :] < lens[:, None], slots, 0)
    page, offset = safe_slots // PAGE_SIZE, safe_slots % PAGE_SIZE
    payload = table[
        page[:, :, None],
        offset[:, :, None] * 64 + torch.arange(64, device=q.device),
    ]
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2).long()
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=q.device)
    values = levels[codes & 7]
    values = torch.where((codes & 8) != 0, -values, values)
    exponents = table[
        page[:, :, None],
        PAGE_SIZE * 64 + offset[:, :, None] * 4 + torch.arange(4, device=q.device),
    ].int()
    scales = torch.ldexp(
        torch.ones_like(exponents, dtype=torch.float32), exponents - 127
    )
    keys = (values * scales.repeat_interleave(32, dim=-1)).to(torch.bfloat16)
    scores = torch.bmm(q.double(), keys.double().transpose(1, 2)).to(torch.bfloat16)
    products = (scores.double().clamp_min(0) * weights.double()[:, :, None]).to(
        torch.bfloat16
    )
    logits = products.double().sum(dim=1).to(torch.bfloat16).float()
    return logits.masked_fill(position[None, :] >= lens[:, None], -torch.inf)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("rows", [1, 6, 13])
@pytest.mark.parametrize("width", [0, 1, 63, 64, 65, 129])
def test_fp4_logits_visibility_boundaries(heads, rows, width):
    args = _make_logits_case(rows, heads, width)
    actual = fp4_index_logits_decode(*args, PAGE_SIZE)
    expected = _reference_logits(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (rows, width)
    assert actual.dtype == torch.float32


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("rows", [1, 6])
def test_fp4_logits_noncontiguous_queries_and_weights(heads, rows):
    q, weights, slots, lens, table = _make_logits_case(rows, heads, 129)
    q_storage = torch.empty(rows, heads, HEAD_DIM * 2, device=q.device, dtype=q.dtype)
    q_view = q_storage[..., ::2]
    q_view.copy_(q)
    weight_storage = torch.empty(rows, heads * 2, device=q.device, dtype=weights.dtype)
    weight_view = weight_storage[:, ::2]
    weight_view.copy_(weights)
    actual = fp4_index_logits_decode(q_view, weight_view, slots, lens, table, PAGE_SIZE)
    expected = _reference_logits(q, weights, slots, lens, table)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    if rows == 1:
        # A singleton batch can be contiguous even with an arbitrary batch stride.
        q_view = q.as_strided(q.shape, (0, HEAD_DIM, 1))
        assert q_view.is_contiguous()
        actual = fp4_index_logits_decode(q_view, weights, slots, lens, table, PAGE_SIZE)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.skipif(_is_xpu, reason="CUDA graph replay requires a CUDA device")
def test_fp4_logits_graph_replay_updates_visibility_and_mapping(heads):
    args = _make_logits_case(13, heads, 193)
    for _ in range(3):
        fp4_index_logits_decode(*args, PAGE_SIZE)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fp4_index_logits_decode(*args, PAGE_SIZE)
    q, weights, slots, lens, table = args
    full_slots = slots.clone()
    for step, visible in enumerate([193, 0, 1, 63, 64, 65, 129, 193]):
        lens.copy_((visible - torch.arange(13, device=q.device) % 6).clamp_min(0))
        slots.copy_(full_slots.roll(step, dims=1))
        q.neg_()
        weights.neg_()
        table.copy_(table.roll(1, dims=0))
        graph.replay()
        expected = _reference_logits(*args)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        invalid = torch.arange(193, device=q.device)[None, :] >= lens[:, None]
        assert torch.isneginf(actual[invalid]).all().item()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_fp4_logits_invisible_tiles_ignore_nonfinite_queries(value):
    args = _make_logits_case(6, 32, 129)
    q, weights, slots, lens, table = args
    q.fill_(value)
    weights.fill_(value)
    lens.zero_()
    slots.fill_(-1)
    actual = fp4_index_logits_decode(*args, PAGE_SIZE)
    assert torch.isneginf(actual).all().item()


@pytest.mark.skipif(_is_xpu, reason="Paged decode uses CUDA persistent kernels")
@pytest.mark.parametrize(
    "batch,ratio,width,masked",
    [
        (6, 1, 193, False),
        (6, 2, 193, True),
        (6, 1, 16385, True),
        (6, 2, 8193, False),
        (6, 1, 32769, False),
        (64, 1, 32769, True),
    ],
)
def test_fp4_paged_logits_replay(batch, ratio, width, masked):
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import amax_topk_blocks
    from sglang.kernels.ops.attention.dsv4.topk import (
        plan_topk_v2,
        topk_transform_paged_v2,
    )

    q, weights, slots, lens, table = _make_logits_case(batch, 32, width)
    capacity = 1048580 // ratio
    req = torch.arange(batch, device=q.device, dtype=torch.int32)
    req_table = torch.full(
        (batch, capacity * ratio), -1, device=q.device, dtype=torch.int32
    )
    req_table[:, : width * ratio : ratio] = (slots * ratio).to(torch.int32)
    lengths = lens.to(torch.int32)
    candidate_mask = (
        torch.rand(batch, capacity, device=q.device) > 0.3 if masked else None
    )
    k = min(512, width)

    def run():
        plan = plan_topk_v2(lengths)
        scores = fp4_index_logits_paged(
            q,
            weights,
            req,
            req_table,
            lengths,
            table,
            PAGE_SIZE,
            capacity,
            ratio,
            candidate_mask,
        )
        indices = torch.empty((batch, k), dtype=torch.int32, device=q.device)
        topk_transform_paged_v2(scores, lengths, None, indices, 1, plan)
        pages = torch.empty((batch + 1, 528), dtype=torch.int32, device=q.device)[
            :, :512
        ]
        raw = torch.empty_like(pages) if ratio == 1 else None
        finish_paged_indexer_topk(
            indices, scores, lengths, req, req_table, pages, raw, ratio, masked
        )
        return scores, indices, pages, raw

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        scores, indices, pages, raw = run()
    for step, visible in enumerate([width, 0, 1, 63, 64, 65, width]):
        lens.copy_((visible - torch.arange(batch, device=q.device)).clamp_min(0))
        lengths.copy_(lens)
        req.copy_(req.roll(1))
        slots.copy_(slots.roll(step, dims=1))
        req_table[req.long(), : width * ratio : ratio] = (slots * ratio).to(torch.int32)
        q.neg_()
        # Poison the unused capacity: length-aware top-k must never select it.
        scores.fill_(torch.inf)
        graph.replay()
        expected = _reference_logits(q, weights, slots, lens, table)
        if masked:
            expected.masked_fill_(~candidate_mask[:, :width], -torch.inf)
        visible_mask = torch.arange(width, device=q.device)[None, :] < lens[:, None]
        torch.testing.assert_close(
            scores[:, :width][visible_mask], expected[visible_mask], atol=0, rtol=0
        )
        blocks = amax_topk_blocks(scores, lengths, (lengths + 7) // 8, 2048)
        assert (pages[batch] == -1).all().item()
        if raw is not None:
            assert (raw[batch] == -1).all().item()
        for row, length in enumerate(lens.tolist()):
            selected = indices[row][indices[row] >= 0].long()
            assert selected.numel() == selected.unique().numel() == min(k, length)
            assert not selected.numel() or selected.max().item() < length
            torch.testing.assert_close(
                expected[row, selected].sort().values,
                expected[row, :length].topk(min(k, length)).values.sort().values,
                atol=0,
                rtol=0,
            )
            kept = (
                selected[expected[row, selected] > -torch.inf] if masked else selected
            )
            kept = kept.sort().values
            expected_pages = torch.full_like(pages[row], -1)
            expected_pages[: kept.numel()] = slots[row, kept].to(torch.int32)
            torch.testing.assert_close(pages[row], expected_pages, rtol=0, atol=0)
            if raw is not None:
                expected_raw = torch.full_like(raw[row], -1)
                expected_raw[: kept.numel()] = kept.to(torch.int32)
                torch.testing.assert_close(raw[row], expected_raw, rtol=0, atol=0)
            nblocks = (length + 7) // 8
            chosen = blocks[row][blocks[row] >= 0].long()
            assert chosen.numel() == chosen.unique().numel() == min(2048, nblocks)
            if nblocks:
                assert chosen.max().item() < nblocks
                assert nblocks - 1 in chosen.tolist()
                padded = torch.full((nblocks * 8,), -torch.inf, device=q.device)
                padded[:length] = expected[row, :length]
                maxima = padded.view(-1, 8).amax(1)
                maxima[-1] = torch.inf
                torch.testing.assert_close(
                    maxima[chosen].sort().values,
                    maxima.topk(min(2048, nblocks)).values.sort().values,
                    atol=0,
                    rtol=0,
                )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Hopper paged indexer dispatch",
)
@pytest.mark.parametrize("ratio", [1, 2])
def test_hopper_indexer_backends_replay(ratio):
    from sglang.srt.layers.attention.dsv4.v41_indexer import scoring
    from sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks import (
        DenseBlocksBackend,
    )
    from sglang.srt.layers.attention.dsv4.v41_indexer.full_topk import FullTopKIndexer
    from sglang.srt.layers.attention.dsv4.v41_indexer.types import (
        DecodeInputs,
        Selection,
    )

    batch, width, capacity, topk = 12, 193, 32772, 64
    q, weights, slots, lens, table = _make_logits_case(batch, 32, width)
    consumer_q = -q
    req = torch.arange(batch, device=q.device, dtype=torch.int32) // 6
    req_table = torch.full(
        (2, capacity * ratio), -1, device=q.device, dtype=torch.int32
    )
    req_table[:, : width * ratio : ratio] = (slots[::6] * ratio).int()
    positions = (lens * ratio - 1).clamp_min(0)
    pool = SimpleNamespace(get_index_k_with_scale_buffer=lambda layer: table)
    indexer = SimpleNamespace(
        queries=lambda q, freqs: q, head_weights=lambda x: x, index_topk=topk
    )
    common = dict(
        indexer=indexer,
        layer_id=0,
        compress_ratio=ratio,
        freqs_cis=torch.zeros(width * ratio + 1, device=q.device),
        x=weights,
        positions=positions,
        req_rows=req,
        paged_metadata=SimpleNamespace(max_compressed_seq_len=capacity),
        is_verify=True,
    )
    inputs = DecodeInputs(q_lora=q, **common)
    consumer_inputs = DecodeInputs(q_lora=consumer_q, **common)
    kwargs = dict(
        token_to_kv_pool=pool, req_to_token=req_table, use_deep_gemm_prefill=False
    )
    full = FullTopKIndexer(**kwargs, use_deep_gemm_decode=False)
    candidates = DenseBlocksBackend(
        **kwargs, candidate_topk_blocks=4, candidate_block_size=8
    )
    outputs = [
        Selection(
            page_indices=torch.empty(
                (batch + 1, topk + 8), device=q.device, dtype=torch.int32
            ),
            raw_indices=torch.empty(
                (batch + 1, topk + 8), device=q.device, dtype=torch.int32
            )
            if ratio == 1
            else None,
        )
        for _ in range(3)
    ]
    captured_scores = []
    captured_masks = []

    def record_scores(*args, **kwargs):
        result = fp4_index_logits_paged(*args, **kwargs)
        captured_scores.append(result)
        captured_masks.append(args[-1])
        return result

    def run():
        full.topk_decode(inputs, outputs[0])
        published = candidates.publish_decode(inputs, outputs[1])
        candidates.consume_decode(consumer_inputs, published, outputs[2])
        return published

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with patch.object(scoring, "fp4_index_logits_paged", side_effect=record_scores):
        with torch.cuda.graph(graph):
            published = run()
    assert (
        len(captured_scores) == 3
    )  # all three runtime entrypoints took the paged path
    mask = published.decode_mask
    assert captured_masks[:2] == [None, None]
    assert captured_masks[2] is mask
    for step, visible in enumerate([193, 0, 1, 7, 8, 9, 65, 193]):
        # -1 positions exercise zero-length padded rows, also for ratio 1.
        lens.copy_((visible - torch.arange(batch, device=q.device) % 6).clamp_min(0))
        positions.copy_(lens * ratio - 1)
        req.copy_(req.roll(1))
        req_table[:, : width * ratio : ratio] = (slots[::6].roll(step, 1) * ratio).int()
        q.neg_()
        for scores in captured_scores:
            scores.fill_(torch.inf)
        for out in outputs:
            out.page_indices.fill_(12345)
            if out.raw_indices is not None:
                out.raw_indices.fill_(12345)
        graph.replay()
        current_slots = req_table[req.long(), : width * ratio : ratio].long() // ratio
        source_scores = _reference_logits(q, weights, current_slots, lens, table)
        consumer_scores = _reference_logits(
            consumer_q, weights, current_slots, lens, table
        )
        for row, length in enumerate(lens.tolist()):
            blocks = published.blocks[row]
            blocks = blocks[blocks >= 0].long()
            nblocks = (length + 7) // 8
            assert blocks.numel() == blocks.unique().numel() == min(4, nblocks)
            if nblocks:
                assert nblocks - 1 in blocks.tolist()
                maxima = torch.full((nblocks * 8,), -torch.inf, device=q.device)
                maxima[:length] = source_scores[row, :length]
                maxima = maxima.view(-1, 8).amax(-1)
                maxima[-1] = torch.inf
                torch.testing.assert_close(
                    maxima[blocks].sort().values,
                    maxima.topk(min(4, nblocks)).values.sort().values,
                )
            keep = torch.isin(torch.arange(width, device=q.device) // 8, blocks)
            torch.testing.assert_close(mask[row, :width], keep)
            consumer_scores[row].masked_fill_(~keep, -torch.inf)
            for out, scores in zip(
                outputs, (source_scores, source_scores, consumer_scores)
            ):
                pages = out.page_indices[row]
                pages = pages[pages >= 0]
                # The shuffled request mapping is bijective within a row.
                selected = (current_slots[row, :, None] == pages[None, :]).nonzero()[
                    :, 0
                ]
                count = min(topk, int(torch.isfinite(scores[row]).sum()))
                assert pages.numel() == selected.unique().numel() == count
                torch.testing.assert_close(
                    scores[row, selected].sort().values,
                    scores[row].topk(count).values.sort().values,
                )
                expected_pages = torch.full_like(out.page_indices[row], -1)
                expected_pages[:count] = current_slots[
                    row, selected.sort().values
                ].int()
                torch.testing.assert_close(out.page_indices[row], expected_pages)
                if out.raw_indices is not None:
                    expected_raw = torch.full_like(out.raw_indices[row], -1)
                    expected_raw[:count] = selected.sort().values.int()
                    torch.testing.assert_close(out.raw_indices[row], expected_raw)
        for out in outputs:
            assert (out.page_indices[batch] == -1).all().item()
            if out.raw_indices is not None:
                assert (out.raw_indices[batch] == -1).all().item()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
