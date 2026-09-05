"""dispatch_fill_masked_bf16 must match the bf16 branch of moe_ep_deepgemm_preprocess;
slot order is nondeterministic in both, so checks are order-independent invariants."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.moe.ep_moe_kernels import (  # noqa: E402
    moe_ep_deepgemm_preprocess,
)
from sglang.srt.lora.moe.base_gemm_provider.base import (  # noqa: E402
    expected_rows_per_expert,
)
from sglang.srt.lora.moe.kernels.dispatch_masked import (  # noqa: E402
    dispatch_fill_masked_bf16,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

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
    """-1 pairs take no slot, keep their pair_to_row entry untouched, copy nothing."""
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
    assert bool(torch.all(pair_to_row_out[invalid] == -777))


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
