"""Masked dispatch invariants and FP8 row-mover parity with upstream quantization."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

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

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused masked dispatch requires CUDA"
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
    ("num_tokens", "top_k", "num_experts", "hidden"),
    ((8, 2, 4, 128), (192, 8, 16, 320)),
    ids=("decode-single-block-ref", "prefill-multi-block-ref"),
)
def test_matches_two_kernel_composition(
    num_tokens: int, top_k: int, num_experts: int, hidden: int
) -> None:
    device = torch.device("cuda")
    topk_ids = _routed_topk_ids(num_tokens, top_k, num_experts, seed=0xD15).to(device)
    hidden_states = _rand_hidden(num_tokens, hidden, seed=0xF111).to(device)
    _assert_matches_reference(topk_ids, hidden_states, num_experts, top_k)


def test_sentinel_pairs_are_skipped() -> None:
    """-1 pairs take no slot, map to pair_to_row == -1 (the finalize gate), copy nothing."""
    device = torch.device("cuda")
    num_tokens, top_k, num_experts, hidden = 64, 4, 8, 128
    topk_ids = _routed_topk_ids(num_tokens, top_k, num_experts, seed=0x5E17)
    drop = torch.rand(topk_ids.shape, generator=torch.Generator().manual_seed(7)) < 0.3
    topk_ids[drop] = -1
    topk_ids = topk_ids.to(device)
    hidden_states = _rand_hidden(num_tokens, hidden, seed=0xB0B).to(device)
    _assert_matches_reference(topk_ids, hidden_states, num_experts, top_k)

    m_max = (num_tokens // 256 + 1) * 256
    pair_to_row_out = torch.full(
        (num_tokens * top_k,), -777, dtype=torch.int32, device=device
    )
    masked_m_out = torch.empty(num_experts, dtype=torch.int32, device=device)
    gateup_input_out = torch.empty(
        (num_experts, m_max, hidden), dtype=torch.bfloat16, device=device
    )
    dispatch_fill_masked_bf16(
        hidden_states,
        topk_ids,
        top_k,
        masked_m_out=masked_m_out,
        pair_to_row_out=pair_to_row_out,
        rows_out=gateup_input_out,
    )
    invalid = topk_ids.view(-1) < 0
    assert bool(invalid.any())
    assert bool(torch.all(pair_to_row_out[invalid] == -1))


def test_skewed_routing_fills_one_expert() -> None:
    """All traffic to one expert: masked_m near m_max, every other expert empty."""
    device = torch.device("cuda")
    num_tokens, top_k, num_experts, hidden = 250, 1, 8, 128
    topk_ids = torch.full((num_tokens, top_k), 3, dtype=torch.int32, device=device)
    hidden_states = _rand_hidden(num_tokens, hidden, seed=0xACE).to(device)
    _assert_matches_reference(topk_ids, hidden_states, num_experts, top_k)

    masked, pair_to_row, slab = _masked_buffers(
        num_tokens, top_k, num_experts, hidden, device=device
    )
    dispatch_fill_masked_bf16(
        hidden_states,
        topk_ids,
        top_k,
        masked_m_out=masked,
        pair_to_row_out=pair_to_row,
        rows_out=slab,
    )
    counts = masked.cpu()
    assert counts[3].item() == num_tokens
    assert counts.sum().item() == num_tokens


def test_empty_batch() -> None:
    device = torch.device("cuda")
    num_experts, top_k, hidden = 4, 2, 64
    topk_ids = torch.empty((0, top_k), dtype=torch.int32, device=device)
    hidden_states = torch.empty((0, hidden), dtype=torch.bfloat16, device=device)
    masked, pair_to_row, slab = _masked_buffers(
        0, top_k, num_experts, hidden, device=device
    )
    dispatch_fill_masked_bf16(
        hidden_states,
        topk_ids,
        top_k,
        masked_m_out=masked,
        pair_to_row_out=pair_to_row,
        rows_out=slab,
    )
    assert expected_rows_per_expert(0, num_experts) == 1
    assert torch.equal(masked, torch.zeros_like(masked))


@pytest.mark.parametrize("hidden", [128, 512, 1536, 3072, 7168])
@pytest.mark.parametrize("domain", ["masked", "contiguous"])
@pytest.mark.parametrize("captured", [False, True])
def test_fp8_row_movers_match_upstream_quant(hidden, domain, captured) -> None:
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
            dispatch_layout_contiguous(
                x,
                ids,
                num_experts,
                top_k,
                8,
                seg_counts_out=counts,
                seg_offsets_out=offsets,
                pair_to_row_out=mapping,
            )
            dispatch_fill_rows_contiguous_fp8(
                x, ids, mapping, rows_fp8_out=rows, scale_out=scales
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
