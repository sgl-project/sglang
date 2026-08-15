"""GPU invariance coverage for the engine's fused S1 dispatch kernel.

``fused_masked_preprocess`` must be an exact drop-in for the bf16 branch of
``moe_ep_deepgemm_preprocess``.  Slot order within an expert is
atomic-arrival nondeterministic in BOTH implementations, so the checks are
order-independent invariants rather than direct ``src2dst`` equality:
bitwise-equal ``masked_m`` histograms, every valid pair's destination inside
its expert region with no duplicates, and slab rows bitwise-equal to their
source token rows.  Sentinel pairs, skewed routing, the >1024-pair regime
(the reference's multi-block path), and the empty batch are the blind spots
the runner numerics suite cannot reach.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from sglang.kernels.ops.moe.ep_moe_kernels import (  # noqa: E402
    moe_ep_deepgemm_preprocess,
)
from sglang.srt.lora.moe.base_gemm_provider.masked_dispatch import (  # noqa: E402
    fused_masked_preprocess,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

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
    masked, expected, src2dst, slab, scale = fused_masked_preprocess(
        topk_ids,
        num_local_experts,
        hidden_states,
        top_k,
        None,
        output_dtype=torch.bfloat16,
    )
    assert ref_scale is None and scale is None
    assert expected == ref_expected
    assert torch.equal(masked, ref_masked)

    hidden = hidden_states.shape[1]
    m_max = slab.shape[1]
    flat_ids = topk_ids.view(-1).long()
    valid = flat_ids >= 0
    experts = flat_ids[valid]
    histogram = torch.bincount(experts, minlength=num_local_experts).to(torch.int32)
    assert torch.equal(masked, histogram)

    dst = src2dst.long()[valid]
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
    """-1 pairs take no slot, keep their src2dst entry untouched, copy nothing."""
    device = torch.device("cuda")
    num_tokens, top_k, num_experts, hidden = 64, 4, 8, 128
    topk_ids = _routed_topk_ids(num_tokens, top_k, num_experts, seed=0x5E17)
    drop = torch.rand(topk_ids.shape, generator=torch.Generator().manual_seed(7)) < 0.3
    topk_ids[drop] = -1
    topk_ids = topk_ids.to(device)
    hidden_states = _rand_hidden(num_tokens, hidden, seed=0xB0B).to(device)
    _assert_matches_reference(topk_ids, hidden_states, num_experts, top_k)

    # Workspace out-tensor contract: pinned buffers are used in place, and a
    # poisoned src2dst proves no store happens on invalid lanes.
    m_max = (num_tokens // 256 + 1) * 256
    src2dst_out = torch.full(
        (num_tokens * top_k,), -777, dtype=torch.int32, device=device
    )
    masked_m_out = torch.empty(num_experts, dtype=torch.int32, device=device)
    gateup_input_out = torch.empty(
        (num_experts, m_max, hidden), dtype=torch.bfloat16, device=device
    )
    masked, _expected, src2dst, slab, _scale = fused_masked_preprocess(
        topk_ids,
        num_experts,
        hidden_states,
        top_k,
        None,
        output_dtype=torch.bfloat16,
        masked_m_out=masked_m_out,
        src2dst_out=src2dst_out,
        gateup_input_out=gateup_input_out,
    )
    assert masked is masked_m_out
    assert src2dst is src2dst_out
    assert slab is gateup_input_out
    invalid = topk_ids.view(-1) < 0
    assert bool(invalid.any())
    assert bool(torch.all(src2dst[invalid] == -777))


def test_skewed_routing_fills_one_expert() -> None:
    """All traffic to one expert: masked_m near m_max, every other expert empty."""
    device = torch.device("cuda")
    num_tokens, top_k, num_experts, hidden = 250, 1, 8, 128
    topk_ids = torch.full((num_tokens, top_k), 3, dtype=torch.int32, device=device)
    hidden_states = _rand_hidden(num_tokens, hidden, seed=0xACE).to(device)
    _assert_matches_reference(topk_ids, hidden_states, num_experts, top_k)

    masked, _expected, _src2dst, _slab, _scale = fused_masked_preprocess(
        topk_ids, num_experts, hidden_states, top_k, None
    )
    counts = masked.cpu()
    assert counts[3].item() == num_tokens
    assert counts.sum().item() == num_tokens


def test_empty_batch() -> None:
    device = torch.device("cuda")
    num_experts, top_k, hidden = 4, 2, 64
    topk_ids = torch.empty((0, top_k), dtype=torch.int32, device=device)
    hidden_states = torch.empty((0, hidden), dtype=torch.bfloat16, device=device)
    masked, expected, src2dst, slab, scale = fused_masked_preprocess(
        topk_ids, num_experts, hidden_states, top_k, None
    )
    assert scale is None
    assert expected == (topk_ids.numel() - 1) // num_experts + 1
    assert torch.equal(masked, torch.zeros_like(masked))
    assert src2dst.numel() == 0
    assert slab.shape == (num_experts, 256, hidden)
