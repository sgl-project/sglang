"""Dispatch invariants and fused FP8 row-mover/activation parity."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.lora.moe.activation_delta import act_delta  # noqa: E402
from sglang.kernels.ops.lora.moe.dispatch_masked import (  # noqa: E402
    dispatch_fill_masked_bf16,
)
from sglang.kernels.ops.moe.ep_moe_kernels import (  # noqa: E402
    moe_ep_deepgemm_preprocess,
)
from sglang.srt.lora.moe.base_gemm_provider.base import (  # noqa: E402
    expected_rows_per_expert,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused LoRA stages require CUDA"
)


def _routed_topk_ids(
    num_tokens: int, top_k: int, num_experts: int, *, seed: int
) -> torch.Tensor:
    """Distinct experts per token (the capacity invariant both kernels rely on)."""
    generator = torch.Generator().manual_seed(seed)
    scores = torch.rand((num_tokens, num_experts), generator=generator)
    return torch.topk(scores, top_k, dim=1).indices.to(torch.int32)


def _rand_hidden(num_tokens: int, hidden: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((num_tokens, hidden), generator=generator).to(torch.bfloat16)


def _masked_buffers(
    num_tokens: int, top_k: int, num_experts: int, hidden: int, *, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m_max = (num_tokens // 256 + 1) * 256
    return (
        torch.empty(num_experts, dtype=torch.int32, device=device),
        torch.empty(num_tokens * top_k, dtype=torch.int32, device=device),
        torch.empty((num_experts, m_max, hidden), dtype=torch.bfloat16, device=device),
    )


def _assert_matches_reference(
    topk_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    num_local_experts: int,
    top_k: int,
) -> None:
    ref_masked, ref_expected, _ref_s2d, _ref_slab, ref_scale = (
        moe_ep_deepgemm_preprocess(
            topk_ids,
            num_local_experts,
            hidden_states,
            top_k,
            None,
            output_dtype=torch.bfloat16,
        )
    )
    masked, pair_to_row, slab = _masked_buffers(
        hidden_states.shape[0],
        top_k,
        num_local_experts,
        hidden_states.shape[1],
        device=hidden_states.device,
    )
    pair_to_row.fill_(-777)
    dispatch_fill_masked_bf16(
        hidden_states,
        topk_ids,
        top_k,
        masked_m_out=masked,
        pair_to_row_out=pair_to_row,
        rows_out=slab,
    )
    assert ref_scale is None
    assert expected_rows_per_expert(topk_ids.numel(), num_local_experts) == ref_expected
    assert torch.equal(masked, ref_masked)

    hidden = hidden_states.shape[1]
    m_max = slab.shape[1]
    flat_ids = topk_ids.view(-1).long()
    valid = flat_ids >= 0
    assert bool(torch.all(pair_to_row[~valid] == -1))
    experts = flat_ids[valid]
    histogram = torch.bincount(experts, minlength=num_local_experts).to(torch.int32)
    assert torch.equal(masked, histogram)

    dst = pair_to_row.long()[valid]
    region_lo = experts * m_max
    assert bool(torch.all(dst >= region_lo))
    assert bool(torch.all(dst < region_lo + masked.long()[experts]))
    assert dst.unique().numel() == dst.numel()

    tokens = torch.div(
        torch.nonzero(valid, as_tuple=True)[0], top_k, rounding_mode="floor"
    )
    assert torch.equal(slab.view(-1, hidden)[dst], hidden_states[tokens])


@pytest.mark.parametrize(
    ("num_tokens", "top_k", "num_experts", "hidden", "skewed_expert"),
    ((8, 2, 4, 128, None), (192, 8, 16, 320, None), (250, 1, 8, 128, 3)),
    ids=("decode-single-block-ref", "prefill-multi-block-ref", "skewed-one-expert"),
)
def test_matches_two_kernel_composition(
    num_tokens: int, top_k: int, num_experts: int, hidden: int, skewed_expert
) -> None:
    """The skewed case sends all traffic to one expert: masked_m near m_max,
    every other expert empty."""
    device = torch.device("cuda")
    if skewed_expert is None:
        topk_ids = _routed_topk_ids(num_tokens, top_k, num_experts, seed=0xD15)
    else:
        topk_ids = torch.full((num_tokens, top_k), skewed_expert, dtype=torch.int32)
    hidden_states = _rand_hidden(num_tokens, hidden, seed=0xF111).to(device)
    _assert_matches_reference(topk_ids.to(device), hidden_states, num_experts, top_k)


def test_sentinel_pairs_are_skipped() -> None:
    """-1 pairs take no slot, map to pair_to_row == -1 (the finalize gate), copy nothing."""
    device = torch.device("cuda")
    num_tokens, top_k, num_experts, hidden = 64, 4, 8, 128
    topk_ids = _routed_topk_ids(num_tokens, top_k, num_experts, seed=0x5E17)
    drop = torch.rand(topk_ids.shape, generator=torch.Generator().manual_seed(7)) < 0.3
    assert bool(drop.any())
    topk_ids[drop] = -1
    topk_ids = topk_ids.to(device)
    hidden_states = _rand_hidden(num_tokens, hidden, seed=0xB0B).to(device)
    _assert_matches_reference(topk_ids, hidden_states, num_experts, top_k)


@pytest.mark.parametrize("hidden", [128, 1536, 7168])
@pytest.mark.parametrize("domain", ["masked", "contiguous", "contiguous_fold"])
@pytest.mark.parametrize("captured", [False, True])
def test_fp8_row_movers_match_upstream_quant(hidden, domain, captured) -> None:
    # contiguous_fold: the layout leaves slots in the map and the fill finalizes it.
    from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
        contiguous_m_pad_ceiling,
        dispatch_fill_rows_contiguous_fp8,
        dispatch_layout_contiguous,
    )
    from sglang.kernels.ops.lora.moe.dispatch_masked import dispatch_fill_masked_fp8
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    num_tokens, top_k, num_experts = 16, 4, 8
    device = torch.device("cuda")
    ids = _routed_topk_ids(num_tokens, top_k, num_experts, seed=5).to(device)
    ids[0] = -1
    ids[1, 2] = -1
    x = _rand_hidden(num_tokens, hidden, seed=6).to(device)
    x[2].zero_()
    x[3, ::128] = 32
    counts = torch.empty(num_experts, dtype=torch.int32, device=device)
    mapping = torch.empty(num_tokens * top_k, dtype=torch.int32, device=device)
    if domain == "masked":
        shape = (num_experts, num_tokens, hidden)
    else:
        offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        shape = (contiguous_m_pad_ceiling(mapping.numel(), num_experts, 8), hidden)
    rows = torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)
    scales = torch.empty(
        (*shape[:-1], hidden // 128), dtype=torch.float32, device=device
    )

    def dispatch():
        if domain == "masked":
            dispatch_fill_masked_fp8(
                x,
                ids,
                top_k,
                masked_m_out=counts,
                pair_to_row_out=mapping,
                rows_fp8_out=rows,
                scale_out=scales,
            )
        else:
            fold = domain == "contiguous_fold"
            dispatch_layout_contiguous(
                x,
                ids,
                num_experts,
                top_k,
                8,
                seg_counts_out=counts,
                seg_offsets_out=offsets,
                pair_to_row_out=mapping,
                finalize=not fold,
            )
            dispatch_fill_rows_contiguous_fp8(
                x,
                ids,
                mapping,
                rows_fp8_out=rows,
                scale_out=scales,
                seg_offsets=offsets if fold else None,
            )

    if captured:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            dispatch()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            dispatch()
        run = graph.replay
    else:
        run = dispatch
    for step in range(2):
        rows.fill_(0.25)
        scales.fill_(-777)
        if step:
            x.mul_(0.5)
            ids.copy_(ids.roll(1, dims=0))
        run()
        ref_q, ref_s = sglang_per_token_group_quant_fp8(x, 128)
        valid = ids.view(-1) >= 0
        tokens = torch.arange(num_tokens, device=device).repeat_interleave(top_k)[valid]
        dst = mapping[valid].long()
        flat_rows = rows.view(-1, hidden)
        flat_scales = scales.view(-1, hidden // 128)
        # A single 128-wide group takes upstream's per-token kernel, whose
        # small-batch variant saturates the codes of an all-zero row (its scale
        # is zero, so every code dequantizes to zero); compare those by scale.
        live = ref_s[tokens].ne(0).any(dim=-1)
        assert torch.equal(
            flat_rows[dst][live].view(torch.uint8),
            ref_q[tokens][live].view(torch.uint8),
        )
        # The movers write zero codes for a zero-scale row (a NaN code would
        # not dequantize to zero), so those rows must hold exactly zero.
        assert bool(torch.all(flat_rows[dst][~live].float() == 0))
        assert torch.equal(flat_scales[dst], ref_s[tokens])
        assert bool(torch.all(mapping[~valid] == -1))
        unused = torch.ones(flat_rows.size(0), dtype=torch.bool, device=device)
        unused[dst] = False
        assert bool(torch.all(flat_rows[unused].float() == 0.25))
        assert bool(torch.all(flat_scales[unused] == -777))


@pytest.mark.parametrize("k", [384, 640, 2048])
@pytest.mark.parametrize("layout", ["flat", "masked"])
def test_per_token_row_quantizer_matches_upstream(k, layout) -> None:
    """Several rows per program; rows past the live bound stay untouched."""
    from sglang.kernels.ops.lora.moe.fp8_quant import quantize_rows_per_token_fp8
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    device = torch.device("cuda")
    num_slabs, m_max = 6, 40  # 240 rows: whole programs of rows, a partial last one
    num_rows = num_slabs * m_max
    x = _rand_hidden(num_rows, k, seed=9).to(device)
    x[2].zero_()
    x[3, ::64] = 32
    q = torch.empty(num_rows, k, dtype=torch.float8_e4m3fn, device=device)
    q.fill_(0.25)
    s = torch.full((num_rows,), -777.0, dtype=torch.float32, device=device)
    row = torch.arange(num_rows, device=device)
    if layout == "masked":
        masked_m = torch.tensor([40, 0, 7, 33, 1, 40], dtype=torch.int32, device=device)
        quantize_rows_per_token_fp8(
            x.view(num_slabs, m_max, k),
            q_out=q.view(num_slabs, m_max, k),
            scale_out=s.view(num_slabs, m_max),
            masked_m=masked_m,
        )
        live = row % m_max < masked_m.repeat_interleave(m_max)
    else:
        live_rows = torch.tensor([131], dtype=torch.int32, device=device)
        quantize_rows_per_token_fp8(x, q_out=q, scale_out=s, live_rows=live_rows)
        live = row < 131
    ref_q, ref_s = sglang_per_token_group_quant_fp8(x, k)
    ref_s = ref_s.view(-1)
    # Upstream's per-token kernel saturates the codes of an all-zero row; the
    # row quantizer writes zero codes for it (its scale is zero either way).
    nonzero = live & ref_s.ne(0)
    assert torch.equal(q[nonzero].view(torch.uint8), ref_q[nonzero].view(torch.uint8))
    assert torch.equal(s[live], ref_s[live])
    assert bool(torch.all(q[live & ~nonzero].float() == 0))
    assert bool(torch.all(q[~live].float() == 0.25))
    assert bool(torch.all(s[~live] == -777))


@pytest.mark.parametrize(
    "inter,group,with_delta",
    [
        (128, 128, False),
        (128, 128, True),
        (384, 128, True),
        (1152, 128, False),
        # Whole rows: one scale per row, including rows narrower or wider than
        # the kernel's 512-wide block.
        (384, 384, True),
        (640, 640, False),
    ],
)
def test_fused_act_quant_matches_unfused(inter, group, with_delta):
    from sglang.kernels.ops.lora.moe.fp8_quant import quantize_rows_per_token_fp8
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, top_k, num_experts = 7, 3, 8
    num_pairs = num_tokens * top_k
    gateup = torch.randn(num_pairs, 2 * inter, dtype=torch.bfloat16, device=device)
    delta = (
        0.1
        * torch.randn(num_tokens, top_k, 2 * inter, dtype=torch.bfloat16, device=device)
        if with_delta
        else None
    )
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    topk_ids[0, 0] = -1
    pair_to_row = torch.arange(num_pairs, dtype=torch.int32, device=device)

    def run(act_quant):
        act = torch.zeros(num_pairs, inter, dtype=torch.bfloat16, device=device)
        pairs = torch.zeros(
            num_tokens, top_k, inter, dtype=torch.bfloat16, device=device
        )
        act_delta(
            gateup,
            delta,
            act,
            pairs,
            pair_to_row,
            topk_ids,
            num_local_experts=num_experts,
            act_quant=act_quant,
        )
        return act, pairs

    ref_act, ref_pairs = run(None)
    act_q = torch.zeros(num_pairs, inter, dtype=torch.float8_e4m3fn, device=device)
    act_s = torch.zeros(num_pairs, inter // group, dtype=torch.float32, device=device)
    fused_act, fused_pairs = run((act_q, act_s, group))
    assert torch.equal(fused_act, ref_act)
    assert torch.equal(fused_pairs, ref_pairs)

    if group == inter:
        ref_q = torch.empty_like(act_q)
        ref_s = torch.empty(num_pairs, dtype=torch.float32, device=device)
        quantize_rows_per_token_fp8(ref_act, q_out=ref_q, scale_out=ref_s)
        ref_s = ref_s.view(num_pairs, 1)
    else:
        ref_q, ref_s = sglang_per_token_group_quant_fp8(ref_act, group)
    valid_rows = (topk_ids.flatten() >= 0).nonzero(as_tuple=True)[0]
    assert torch.equal(act_s[valid_rows], ref_s[valid_rows])
    assert torch.equal(
        act_q[valid_rows].view(torch.uint8), ref_q[valid_rows].view(torch.uint8)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
