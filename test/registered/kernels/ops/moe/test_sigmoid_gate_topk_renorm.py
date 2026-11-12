import sys

import pytest
import torch

from sglang.kernels.ops.moe.sigmoid_gate_topk_renorm import sigmoid_gate_topk_renorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def make_inputs(k, shared):
    torch.manual_seed(50100)
    routed = max(k, 16)
    logits = torch.randn(4, routed + shared, device="cuda", dtype=torch.float32)
    bias = torch.randn(routed, device="cuda", dtype=torch.float32)
    scale = torch.tensor([0.75], device="cuda", dtype=torch.float32)
    return logits, bias, scale


@pytest.mark.parametrize("k,shared", [(5, 4), (6, 3), (8, 1), (3, 2)])
@pytest.mark.parametrize("packed", [False, True])
def test_rejects_mismatched_padded_axes(k, shared, packed):
    """Unsupported shared-expert counts must fail before Triton compilation."""
    logits, bias, scale = make_inputs(k, shared)
    with pytest.raises(ValueError, match="same next power of two"):
        sigmoid_gate_topk_renorm(
            logits, k, shared, 1.25, scale, bias, return_packed_topk=packed
        )


@pytest.mark.parametrize("k,shared", [(5, 3), (6, 2), (8, 0), (5, 1)])
@pytest.mark.parametrize("packed", [False, True])
def test_matching_padded_axes_match_reference(k, shared, packed):
    logits, bias, scale = make_inputs(k, shared)
    routed = bias.numel()
    ref_indices = (logits[:, :routed].sigmoid() + bias).topk(k, dim=1).indices
    active = torch.cat(
        (logits.gather(1, ref_indices), logits[:, routed:]), dim=1
    ).sigmoid()
    ref_weights = active / active.sum(dim=1, keepdim=True) * 1.25 * scale

    weights, indices, shared_weights, packed_ids = sigmoid_gate_topk_renorm(
        logits, k, shared, 1.25, scale, bias, return_packed_topk=packed
    )
    torch.cuda.synchronize()
    if packed:
        assert weights is None and indices is None
        indices = packed_ids >> 16
        weights = (packed_ids & 0xFFFF).to(torch.int16).view(torch.bfloat16).float()
        expected = ref_weights[:, :k].to(torch.bfloat16).float()
        torch.testing.assert_close(weights, expected, rtol=1e-2, atol=1e-3)
    else:
        assert packed_ids is None
        torch.testing.assert_close(weights, ref_weights[:, :k], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(indices.long(), ref_indices)
    torch.testing.assert_close(shared_weights, ref_weights[:, k:], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
