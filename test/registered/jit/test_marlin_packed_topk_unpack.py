"""Precision test for the fused packed-topk unpack triton kernel used by the
Marlin MoE runner (fused-gate-topk support).

The FlashInfer / Inkling fused gate emits PackedTopKOutput -- int32
``(expert_id << 16) | bf16-weight-bits``. The Marlin runner reads topk_ids /
topk_weights separately, so it unpacks with a single Triton launch. This test
checks the kernel is bit-identical to the torch elementwise reference and that
pack -> unpack round-trips, across shapes / top_k / num_experts / weight
distributions.
"""

import sys

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.marlin import _fused_unpack_packed_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _torch_unpack(packed: torch.Tensor):
    ids = (packed >> 16).to(torch.int32)
    w = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16).to(torch.float32)
    return ids, w


def _torch_pack(ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # inverse of the unpack, matching trtllm_lora_temp/topk_pack._pack_topk_kernel
    wbits = weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    return (ids.to(torch.int32) << 16) | wbits


@pytest.mark.parametrize(
    "num_tokens,top_k",
    [
        (1, 1),
        (1, 2),
        (3, 6),
        (8, 2),
        (127, 6),
        (512, 6),
        (631, 2),
        (1024, 8),
        (2048, 8),
    ],
)
@pytest.mark.parametrize("num_experts", [2, 8, 64, 256])
@pytest.mark.parametrize("wdist", ["uniform", "edge", "tiny"])
def test_unpack_matches_reference_and_roundtrips(num_tokens, top_k, num_experts, wdist):
    torch.manual_seed(0)
    ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device="cuda"
    )
    if wdist == "uniform":
        w = torch.rand(num_tokens, top_k, device="cuda")
    elif wdist == "edge":
        choices = torch.tensor([0.0, 1.0, 0.5, 0.999, 1e-3], device="cuda")
        w = choices[
            torch.randint(0, choices.numel(), (num_tokens, top_k), device="cuda")
        ]
    else:
        w = torch.rand(num_tokens, top_k, device="cuda") * 1e-3

    packed = _torch_pack(ids, w)
    t_ids, t_w = _fused_unpack_packed_topk(packed)
    r_ids, r_w = _torch_unpack(packed)

    # bit-identical to the torch elementwise reference
    assert torch.equal(t_ids, r_ids)
    assert torch.equal(t_w, r_w)
    # round-trip: ids exact, weights recover the bf16-rounded originals
    assert torch.equal(t_ids, ids)
    torch.testing.assert_close(
        t_w, w.to(torch.bfloat16).to(torch.float32), rtol=0, atol=0
    )
    assert t_ids.dtype == torch.int32 and t_w.dtype == torch.float32
    assert t_ids.shape == (num_tokens, top_k) and t_w.shape == (num_tokens, top_k)


def test_unpack_empty():
    packed = torch.empty((0, 2), dtype=torch.int32, device="cuda")
    ids, w = _fused_unpack_packed_topk(packed)
    assert ids.shape == (0, 2) and w.shape == (0, 2)
    assert ids.dtype == torch.int32 and w.dtype == torch.float32


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
