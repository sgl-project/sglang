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
def test_candidate_fp4_indexer(width):
    from deep_gemm import fp8_fp4_mqa_logits

    from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
        candidate_block_logits,
    )
    from sglang.kernels.ops.attention.dsv4.candidate_fp4_indexer import (
        candidate_fp4_mqa_logits,
        select_candidate_block_indices,
    )
    from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks

    torch.manual_seed(910)
    rows, heads, group, budget = 9, 32, 8, 2048
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
        [0, 1, 7, 8, 9, 16, width - 1, width, width], device="cuda", dtype=torch.int32
    )
    width_aligned = (width + 3) // 4 * 4
    baseline = fp8_fp4_mqa_logits(
        q, k, weights, torch.zeros_like(lens), lens, False, width_aligned
    )
    visible = torch.arange(width, device="cuda")[None, :] < lens[:, None]
    reference = baseline[:, :width].masked_fill(~visible, -torch.inf)
    # Keep the baseline block-selection tie behavior, including all-masked rows.
    reference[1:6] = reference[1:6].masked_fill(visible[1:6], 0)
    block_ids = select_candidate_block_indices(reference.clone(), lens, budget, group)
    reference_mask = select_candidate_blocks(reference, lens[:, None], budget, group)
    # The shared block reducer must retain the paged decode helper's behavior.
    masked, published = candidate_block_logits(
        reference, lens, topk_blocks=budget, block_size=group, published=None
    )
    torch.testing.assert_close(masked, reference)
    torch.testing.assert_close(published, reference_mask)
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
    lens.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isneginf(replayed).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="DeepGEMM MXFP4 requires Blackwell",
)
@pytest.mark.parametrize("ratio", [1, 2])
def test_candidate_prefill_mapping(ratio):
    from types import SimpleNamespace as NS
    from unittest import mock

    from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
    from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks

    torch.manual_seed(910)
    lengths, q_lengths = [262147 * ratio, 0, 1, 8197 * ratio], [33, 0, 1, 17]
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
    backend.candidate_masks = None
    published = []
    for source in [True, False, False]:
        query = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(tokens, 32, device="cuda")
        indexer.is_candidate_source, indexer.uses_candidates = source, not source
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
            elif count:
                local.masked_fill_(~published[b], -torch.inf)
            start += q_len
        selected = torch.empty_like(raw)
        backend_module.topk_transform_ragged_v2(
            scores, lens, out_offsets=offsets, out_indices=selected
        )
        if not source:
            selected = backend_module._mask_topk_scores(scores, selected, offsets)
        sentinel = torch.iinfo(torch.int32).max
        selected = selected.masked_fill(selected < 0, sentinel).sort(-1).values
        chosen = selected != sentinel
        expected_pages = torch.where(
            chosen, physical[selected.clamp_max(slots - 1)], -1
        ).int()
        expected_raw = torch.where(chosen, selected - offsets[:, None], -1)
        budget = ((max(compressed) + 3) // 4 * 4) * 4 * 7
        dense_logits = backend_module._dense_fp4_mqa_logits

        def bounded_logits(*args):
            result = dense_logits(*args)
            assert result.numel() * result.element_size() <= budget
            return result

        with (
            mock.patch.object(
                backend_module, "_DENSE_INDEXER_SCORE_BUDGET_BYTES", budget
            ),
            mock.patch.object(backend_module, "_dense_fp4_mqa_logits", bounded_logits),
        ):
            backend._low_ratio_index_topk_dense(
                layer, weights, query, positions, batch, q_lens, q_lengths
            )
        torch.testing.assert_close(pages, expected_pages, rtol=0, atol=0)
        torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
