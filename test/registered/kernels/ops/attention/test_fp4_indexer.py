from __future__ import annotations

import sys

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
    quantize_fp4_indexer_tensor,
    store_fp4_index_k_cache,
)
from sglang.srt.utils import get_device, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="DeepGEMM MXFP4 requires Blackwell",
)
@pytest.mark.parametrize("width", [17, 65539, 1048579])
@pytest.mark.parametrize("rows", [9, 37])
def test_candidate_fp4_indexer(width, rows):
    from deep_gemm import fp8_fp4_mqa_logits

    from sglang.kernels.ops.attention.dsv4.candidate_fp4_indexer import (
        candidate_fp4_mqa_logits,
    )
    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        select_candidate_blocks,
    )

    torch.manual_seed(910)
    heads, group, budget = 32, 8, 2048
    q = quantize_fp4_indexer_tensor(
        torch.randn(rows * heads, 128, device="cuda", dtype=torch.bfloat16), rne=True
    )
    q = (q[0].view(rows, heads, 64), q[1].view(rows, heads))
    # Independent per-32 scales and negative head weights exercise the FP32
    # reduction, rather than only checking selected positions on positive scores.
    keys = torch.randn(width, 4, 32, device="cuda", dtype=torch.bfloat16)
    keys *= torch.tensor([0.25, 0.5, 2, 4], device="cuda")[None, :, None]
    k = quantize_fp4_indexer_tensor(keys.flatten(1), rne=True)
    weights = torch.randn(rows, heads, device="cuda")
    lens = torch.tensor(
        [0, 1, 7, 8, 9, 16, width - 1] + [width] * (rows - 7),
        device="cuda",
        dtype=torch.int32,
    )
    width_aligned = (width + 3) // 4 * 4
    baseline = fp8_fp4_mqa_logits(
        q, k, weights, torch.zeros_like(lens), lens, False, width_aligned
    )
    visible = torch.arange(width, device="cuda")[None, :] < lens[:, None]
    reference = baseline[:, :width].masked_fill(~visible, -torch.inf)
    # Keep the baseline block-selection tie behavior, including all-masked rows.
    reference[1:6] = reference[1:6].masked_fill(visible[1:6], 0)
    block_ids = select_candidate_blocks(
        reference, lens[:, None], budget, group, return_indices=True
    )
    reference_mask = select_candidate_blocks(reference, lens[:, None], budget, group)
    positions = (
        (block_ids[:, :, None] * group + torch.arange(group, device="cuda"))
        .flatten(1)
        .long()
    )
    actual_mask = torch.zeros(
        (rows, ((width + group - 1) // group + 1) * group),
        device="cuda",
        dtype=torch.bool,
    )
    actual_mask.scatter_(1, positions, True)
    torch.testing.assert_close(actual_mask[:, :width], reference_mask)

    actual = candidate_fp4_mqa_logits(q, k, weights, block_ids, lens, group)
    expected = baseline.gather(1, positions.clamp_max(width_aligned - 1))
    expected.masked_fill_(positions >= lens[:, None], -torch.inf)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Replaying with changed causal lengths must not reuse scores or read padding.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = candidate_fp4_mqa_logits(q, k, weights, block_ids, lens, group)
    original_lens = lens.clone()
    lens.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isneginf(replayed).all()
    lens.copy_(original_lens)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
    block_ids[:, 0] = -1
    invalid = candidate_fp4_mqa_logits(q, k, weights, block_ids, lens, group)
    assert torch.isneginf(invalid[:, :group]).all()
    torch.testing.assert_close(invalid[:, group:], expected[:, group:], rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="DeepGEMM MXFP4 requires Blackwell",
)
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize(
    "case",
    ["dense", "compact", "mixed", "below", "at", "zero_compressed", "empty_query"],
)
def test_candidate_prefill_mapping(ratio, case):
    from types import SimpleNamespace as NS
    from unittest import mock

    from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        select_candidate_blocks,
    )

    torch.manual_seed(910)
    context = 32771 if case == "dense" else 262147
    other = 8197 if case in ("dense", "mixed") else context
    if case in ("below", "at"):
        context = other = 262144 - (case == "below")
    lengths, q_lengths = [context * ratio, 0, 0, other * ratio], [33, 0, 0, 17]
    if case == "zero_compressed":
        lengths[2], q_lengths[2] = 1, 1
    if case == "empty_query":
        # Even a small context without query rows keeps the batch dense.
        lengths[2] = 1
    compact = all(lc == 0 or lc >= 262144 for lc in (s // ratio for s in lengths))
    tokens, topk = sum(q_lengths), 512
    compressed = [length // ratio for length in lengths]
    positions = torch.cat(
        [torch.arange(s - t, s, device="cuda") for s, t in zip(lengths, q_lengths)]
    )
    lens = ((positions + 1) // ratio).int()
    slots = sum(compressed)
    keys = quantize_fp4_indexer_tensor(
        torch.randn(slots, 128, device="cuda", dtype=torch.bfloat16), rne=True
    )
    request_map = torch.zeros(
        len(lengths), max(lengths), device="cuda", dtype=torch.int64
    )
    physical = torch.randperm(slots, device="cuda")
    starts, offset = [], 0
    for b, count in enumerate(compressed):
        starts.append(offset)
        request_map[b, torch.arange(count, device="cuda") * ratio] = (
            physical[offset : offset + count] * ratio
        )
        offset += count
    q_lens = torch.tensor(q_lengths, device="cuda")
    offsets = torch.repeat_interleave(
        torch.tensor(starts, device="cuda", dtype=torch.int32), q_lens
    )
    batch = NS(
        seq_lens_cpu=lengths, req_pool_indices=torch.arange(len(lengths), device="cuda")
    )
    indexer = NS(
        queries=lambda q, freqs: q,
        head_weights=lambda x: x,
        index_topk=topk,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
    )
    layer = NS(
        compress_ratio=ratio,
        indexer=indexer,
        layer_id=0,
        freqs_cis=torch.zeros(max(lengths), 1, device="cuda"),
    )
    backend = backend_module.DeepseekV4AttnBackend.__new__(
        backend_module.DeepseekV4AttnBackend
    )
    pages = torch.empty(tokens, topk, device="cuda", dtype=torch.int32)
    raw = torch.empty_like(pages)
    backend.req_to_token = request_map
    backend.token_to_kv_pool = NS(
        get_low_ratio_index_k_fp4=lambda _, loc: (keys[0][loc], keys[1][loc])
    )
    backend.forward_metadata = NS(
        core_metadata=NS(
            sparse_page_indices=lambda r: pages, sparse_raw_indices=lambda r: raw
        )
    )
    backend.forward_metadata.candidate_metadata = None
    published = []
    for stage in ["full", "source", "consumer1", "consumer2"]:
        source, consume = stage == "source", stage.startswith("consumer")
        query = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(tokens, 32, device="cuda")
        indexer.is_candidate_source, indexer.uses_candidates = source, consume
        packed_q = quantize_fp4_indexer_tensor(query.flatten(0, 1), rne=True)
        scores = backend_module._dense_fp4_mqa_logits(
            (packed_q[0].view(tokens, 32, 64), packed_q[1].view(tokens, 32)),
            (keys[0][physical], keys[1][physical]),
            weights,
            offsets,
            offsets + lens,
            (max(compressed) + 3) // 4 * 4,
        )
        start = 0
        for b, (count, q_len) in enumerate(zip(compressed, q_lengths)):
            rows = slice(start, start + q_len)
            local = scores[rows, :count]
            local.masked_fill_(
                torch.arange(count, device="cuda")[None, :] >= lens[rows, None],
                -torch.inf,
            )
            if source:
                published.append(
                    select_candidate_blocks(local, lens[rows, None], 2048, 8)
                    if count
                    else None
                )
            elif consume and count:
                local.masked_fill_(~published[b], -torch.inf)
            start += q_len
        selected = torch.empty_like(raw)
        backend_module.topk_transform_ragged_v2(
            scores, lens, out_offsets=offsets, out_indices=selected
        )
        if consume:
            selected = backend_module.mask_topk_scores(scores, selected, offsets)
        sentinel = torch.iinfo(torch.int32).max
        selected = selected.masked_fill(selected < 0, sentinel).sort(-1).values
        chosen = selected != sentinel
        expected_pages = torch.where(
            chosen, physical[selected.clamp_max(slots - 1)], -1
        ).int()
        expected_raw = torch.where(chosen, selected - offsets[:, None], -1)
        from sglang.kernels.ops.attention.dsv4 import candidate_fp4_indexer

        with (
            mock.patch.object(
                backend_module,
                "_dense_fp4_mqa_logits",
                wraps=backend_module._dense_fp4_mqa_logits,
            ) as dense_call,
            mock.patch.object(
                candidate_fp4_indexer,
                "candidate_fp4_mqa_logits",
                wraps=candidate_fp4_indexer.candidate_fp4_mqa_logits,
            ) as compact_call,
        ):
            backend._low_ratio_index_topk_dense(
                layer, weights, query, positions, batch, q_lens, q_lengths
            )
        # Sources/full scoring always retain one upstream full-batch allocation.
        assert dense_call.call_count == (0 if consume and compact else 1)
        if dense_call.call_count:
            assert dense_call.call_args.args[0][0].shape[0] == tokens
            assert dense_call.call_args.args[-1] == (max(compressed) + 3) // 4 * 4
        assert compact_call.call_count == (
            sum(bool(lc and q) for lc, q in zip(compressed, q_lengths))
            if consume and compact
            else 0
        )
        if source:
            masks = backend.forward_metadata.candidate_metadata.request_masks
            for mask, lc, q_len in zip(masks, compressed, q_lengths):
                if lc and q_len:
                    assert mask.dtype == (torch.int32 if compact else torch.bool)
        torch.testing.assert_close(pages, expected_pages, rtol=0, atol=0)
        torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)
        if consume:
            # Replay the actual late-layer-tail switch and compact/dense consumer
            # using fewer rows; compare both outputs to the full-request result.
            # Drop every request except the first: mixed batches must not switch
            # to IDs merely because only their long request remains in the tail.
            tail_lens = [min(q_lengths[0], 3)] + [0] * (len(q_lengths) - 1)
            tail_rows, offset = [], 0
            for count, tail_len in zip(q_lengths, tail_lens):
                tail_rows.extend(range(offset + count - tail_len, offset + count))
                offset += count
            tail_rows = torch.tensor(tail_rows, device="cuda")
            tail_pages = torch.empty_like(pages[tail_rows])
            tail_raw = torch.empty_like(raw[tail_rows])
            full_core = backend.forward_metadata.core_metadata
            full_core.low_ratios = (ratio,)
            full_core.sparse_topk_lengths = lambda r: None
            backend.forward_metadata.core_attn_metadata = full_core
            tail_core = NS(
                low_ratios=(ratio,),
                sparse_page_indices=lambda r: tail_pages,
                sparse_raw_indices=lambda r: tail_raw,
                sparse_topk_lengths=lambda r: None,
            )
            backend.tail_forward_metadata = NS(
                candidate_metadata=None,
                core_metadata=tail_core,
                core_attn_metadata=tail_core,
                late_layer_tail=NS(
                    cp_metadata=None,
                    extend_seq_lens_cpu=tail_lens,
                    real_rows=lambda buf: buf[tail_rows],
                ),
            )
            backend.token_to_kv_pool.request_window = None
            batch.attn_cp_metadata = None
            with (
                mock.patch.object(
                    backend_module, "get_local_dp_buffer_len", return_value=0
                ),
                mock.patch.object(backend_module, "set_local_dp_buffer_len"),
            ):
                saved = backend.enter_late_layer_tail(batch)
                backend._low_ratio_index_topk_dense(
                    layer,
                    weights[tail_rows],
                    query[tail_rows],
                    positions[tail_rows],
                    batch,
                    torch.tensor(tail_lens, device="cuda"),
                    tail_lens,
                )
                backend.exit_late_layer_tail(saved, batch)
            torch.testing.assert_close(
                tail_pages, expected_pages[tail_rows], rtol=0, atol=0
            )
            torch.testing.assert_close(
                tail_raw, expected_raw[tail_rows], rtol=0, atol=0
            )


@pytest.mark.parametrize("dtype", [torch.bool, torch.int32])
def test_candidate_prefill_tail_masks(dtype):
    from types import SimpleNamespace as NS
    from unittest import mock

    from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
    from sglang.srt.layers.attention.dsv4.candidate_indexer import CandidateMasks

    # The same current-main tail switch must slice reusable dense masks and
    # compact block IDs, including a request with no tail rows.
    masks = [torch.ones(5, 8, dtype=dtype), torch.ones(3, 4, dtype=dtype)]
    core = NS(low_ratios=())
    full = NS(
        candidate_metadata=CandidateMasks(request_masks=masks), core_attn_metadata=core
    )
    tail = NS(
        candidate_metadata=None,
        core_attn_metadata=core,
        late_layer_tail=NS(cp_metadata=None, extend_seq_lens_cpu=[2, 0]),
    )
    backend = backend_module.DeepseekV4AttnBackend.__new__(
        backend_module.DeepseekV4AttnBackend
    )
    backend.forward_metadata, backend.tail_forward_metadata = full, tail
    backend.token_to_kv_pool = NS(request_window=None)
    batch = NS(attn_cp_metadata=None)
    with (
        mock.patch.object(backend_module, "get_local_dp_buffer_len", return_value=0),
        mock.patch.object(backend_module, "set_local_dp_buffer_len"),
    ):
        saved = backend.enter_late_layer_tail(batch)
        actual = backend.forward_metadata.candidate_metadata.request_masks
        torch.testing.assert_close(actual[0], masks[0][-2:])
        assert actual[0].data_ptr() == masks[0][-2:].data_ptr()
        assert actual[1].shape == (0, 4)
        backend.exit_late_layer_tail(saved, batch)
        assert backend.forward_metadata is full
        assert full.candidate_metadata.request_masks is masks


@pytest.mark.parametrize("scalar_length", [False, True])
def test_candidate_selector_default_unchanged(scalar_length):
    import torch.nn.functional as F

    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        select_candidate_blocks,
    )

    torch.manual_seed(911)
    logits = torch.randn(5, 73)
    lens = 73 if scalar_length else torch.tensor([0, 1, 8, 17, 73])[:, None]
    logits.masked_fill_(torch.arange(73) >= lens, -torch.inf)
    logits[-1].zero_()  # Tied blocks use the same upstream torch.topk call.
    before = logits.clone()
    pooled = F.pad(logits, (0, 7), value=-torch.inf).unflatten(-1, (-1, 8)).amax(-1)
    pooled.masked_fill_(torch.arange(10) == (lens - 1) // 8, torch.inf)
    top = pooled.topk(3, dim=-1)
    expected = (
        torch.zeros_like(pooled, dtype=torch.bool)
        .scatter_(-1, top.indices, top.values > -torch.inf)
        .repeat_interleave(8, dim=-1)[:, :73]
    )
    torch.testing.assert_close(select_candidate_blocks(logits, lens, 3, 8), expected)
    ids = select_candidate_blocks(logits, lens, 3, 8, return_indices=True)
    actual = torch.zeros(5, 11, dtype=torch.bool).scatter_(1, ids.long(), True)
    torch.testing.assert_close(actual.repeat_interleave(8, dim=-1)[:, :73], expected)
    torch.testing.assert_close(logits, before)


def test_compact_source_structure():
    import ast
    import inspect
    import textwrap

    from sglang.kernels.ops.attention.dsv4 import candidate_fp4_indexer
    from sglang.srt.layers.attention import deepseek_v4_backend as backend

    assert backend._TORCH_INDEXER_SCORE_BUDGET_BYTES == 1 << 30
    assert not hasattr(backend, "_DENSE_INDEXER_SCORE_BUDGET_BYTES")
    assert not hasattr(candidate_fp4_indexer, "_candidate_scores_kernel")
    assert not hasattr(candidate_fp4_indexer, "select_candidate_block_indices")
    cls = backend.DeepseekV4AttnBackend
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(cls._low_ratio_index_topk_dense))
    )
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    dense = [
        n
        for n in calls
        if isinstance(n.func, ast.Name) and n.func.id == "_dense_fp4_mqa_logits"
    ]
    assert len(dense) == 1
    # The source's full-batch dense allocation is not nested in any new loop.
    assert any(
        isinstance(n, ast.Assign) and n.value is dense[0] for n in tree.body[0].body
    )
    publication = inspect.getsource(cls._publish_or_consume_candidates)
    assert "_TORCH_INDEXER_SCORE_BUDGET_BYTES // (lc * 4)" in publication
    assert "for start in range(0, t_len, step)" in publication


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("lengths,q_lengths", [([0, 1], [0, 1]), ([262144, 0], [0, 0])])
def test_candidate_empty_prefill(lengths, q_lengths):
    from types import SimpleNamespace as NS
    from unittest import mock

    from sglang.srt.layers.attention import deepseek_v4_backend as backend

    obj = backend.DeepseekV4AttnBackend.__new__(backend.DeepseekV4AttnBackend)
    pages = torch.empty(sum(q_lengths), 512, device="cuda", dtype=torch.int32)
    raw = torch.empty_like(pages)
    obj.forward_metadata = NS(
        core_metadata=NS(
            sparse_page_indices=lambda r: pages, sparse_raw_indices=lambda r: raw
        )
    )
    obj.token_to_kv_pool = NS()
    obj.req_to_token = torch.zeros(2, max(lengths), device="cuda", dtype=torch.int64)
    layer = NS(
        compress_ratio=2, indexer=NS(is_candidate_source=True, uses_candidates=False)
    )
    batch = NS(seq_lens_cpu=lengths, req_pool_indices=torch.arange(2, device="cuda"))
    pos = torch.zeros(sum(q_lengths), device="cuda", dtype=torch.int64)
    with mock.patch.object(backend, "_dense_fp4_mqa_logits") as dense:
        for source in (True, False):
            layer.indexer.is_candidate_source = source
            layer.indexer.uses_candidates = not source
            obj._low_ratio_index_topk_dense(
                layer,
                None,
                None,
                pos,
                batch,
                torch.tensor(q_lengths, device="cuda"),
                q_lengths,
            )
        dense.assert_not_called()
    assert (pages == -1).all() and (raw == -1).all()
    assert len(obj.forward_metadata.candidate_metadata.request_masks) == 2


@pytest.mark.parametrize("mode", ["torch", "sm90_decode"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_candidate_default_callers(mode, ratio):
    from types import SimpleNamespace as NS
    from unittest import mock

    from sglang.srt.layers.attention import deepseek_v4_backend as backend
    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        select_candidate_blocks,
    )

    torch.manual_seed(912)
    lengths = torch.tensor([17, 24])
    positions = lengths * ratio - 1
    physical = torch.randperm(48).reshape(2, 24)
    obj = backend.DeepseekV4AttnBackend.__new__(backend.DeepseekV4AttnBackend)
    obj.req_to_token = physical.repeat_interleave(ratio, dim=1) * ratio
    pages = torch.empty(2, 6, dtype=torch.int32)
    raw = torch.empty_like(pages)
    obj.forward_metadata = NS(
        candidate_metadata=None,
        core_metadata=NS(
            sparse_page_indices=lambda r: pages, sparse_raw_indices=lambda r: raw
        ),
        c1_indexer_metadata=NS(max_compressed_seq_len=24),
        c2_indexer_metadata=NS(max_compressed_seq_len=24),
    )
    obj.token_to_kv_pool = NS(
        get_index_k_with_scale_buffer=lambda _: torch.empty(1, 68 * 64),
        get_low_ratio_index_k_dequant=lambda _, loc: torch.empty(len(loc), 128),
    )
    indexer = NS(
        queries=lambda q, f: q,
        head_weights=lambda x: x,
        scores=lambda q, k, w: q[:, 0, : len(k)].clone(),
        index_topk=6,
        candidate_topk_blocks=2,
        candidate_block_size=4,
    )
    layer = NS(
        compress_ratio=ratio, indexer=indexer, layer_id=0, freqs_cis=torch.zeros(48, 1)
    )
    saved_masks = None
    for source in (True, False):
        indexer.is_candidate_source, indexer.uses_candidates = source, not source
        scores = torch.randn(2, 24).masked_fill(
            torch.arange(24) >= lengths[:, None], -torch.inf
        )
        expected_masks = select_candidate_blocks(scores, lengths[:, None], 2, 4)
        with (
            mock.patch.object(
                backend, "select_candidate_blocks", wraps=select_candidate_blocks
            ) as selector,
            mock.patch.object(
                backend, "fp4_index_logits_decode", return_value=scores.clone()
            ),
        ):
            method = getattr(obj, "_low_ratio_index_topk_" + mode)
            method(
                layer, torch.ones(2, 1), scores[:, None, :], torch.arange(2), positions
            )
        assert selector.call_count == ((2 if mode == "torch" else 1) if source else 0)
        assert all("return_indices" not in c.kwargs for c in selector.call_args_list)
        if source:
            saved_masks = expected_masks
            candidate = obj.forward_metadata.candidate_metadata
            if mode == "torch":
                for i, mask in enumerate(candidate.request_masks):
                    assert mask.dtype == torch.bool
                    torch.testing.assert_close(
                        mask, expected_masks[i : i + 1, : lengths[i]]
                    )
            else:
                assert candidate.mask.dtype == torch.bool
                torch.testing.assert_close(candidate.mask, expected_masks)
        else:
            scores.masked_fill_(~saved_masks, -torch.inf)
        expected = scores.topk(6, dim=-1, sorted=False).indices
        # The forced partial last block may leave fewer than six candidates.
        expected.masked_fill_(scores.gather(1, expected) == -torch.inf, 24)
        expected = expected.sort(-1).values
        valid = expected < lengths[:, None]
        torch.testing.assert_close(raw, torch.where(valid, expected, -1).int())
        torch.testing.assert_close(
            pages,
            torch.where(valid, physical.gather(1, expected.clamp_max(23)), -1).int(),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="DeepGEMM MXFP4 requires Blackwell",
)
def test_candidate_fp4_large_output_offset():
    from deep_gemm import fp8_fp4_mqa_logits

    from sglang.kernels.ops.attention.dsv4.candidate_fp4_indexer import (
        candidate_fp4_mqa_logits,
    )

    # 8 GiB output: the last row starts at element 2**31, without allocating
    # a dense reference for all rows. Never execute the old int32-offset kernel.
    torch.manual_seed(913)
    rows, heads, width = 131073, 32, 16383
    q = quantize_fp4_indexer_tensor(
        torch.randn(heads, 128, device="cuda", dtype=torch.bfloat16), rne=True
    )
    q = (
        q[0].repeat(rows, 1).view(rows, heads, 64),
        q[1].repeat(rows).view(rows, heads),
    )
    k = quantize_fp4_indexer_tensor(
        torch.randn(width, 128, device="cuda", dtype=torch.bfloat16), rne=True
    )
    weights = torch.randn(rows, heads, device="cuda")
    ids = torch.arange(2048, device="cuda", dtype=torch.int32)[None, :].expand(rows, -1)
    lens = torch.full((rows,), width, device="cuda", dtype=torch.int32)
    samples = torch.tensor([0, 1, 31, 32, 65535, 131070, 131071, 131072], device="cuda")
    lens[samples[:3]] = torch.tensor(
        [0, 7, width - 1], device="cuda", dtype=torch.int32
    )
    actual = candidate_fp4_mqa_logits(q, k, weights, ids, lens, 8)
    expected = fp8_fp4_mqa_logits(
        (q[0][samples], q[1][samples]),
        k,
        weights[samples],
        torch.zeros_like(lens[samples]),
        lens[samples],
        False,
        16384,
    )
    expected.masked_fill_(
        torch.arange(16384, device="cuda")[None, :] >= lens[samples, None], -torch.inf
    )
    torch.testing.assert_close(actual[samples], expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
