import itertools

import pytest
import torch

from sglang.jit_kernel.kimi_k2_moe_fused_gate import kimi_k2_moe_fused_gate


NUM_EXPERTS = 384

M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
TOPK_LIST = [1, 2, 4, 6]
RENORMALIZE_LIST = [True, False]

configs = list(itertools.product(M_LIST, TOPK_LIST, RENORMALIZE_LIST))


def _reference_kimi_k2_moe_fused_gate(
    input_tensor, bias, topk, renormalize, routed_scaling_factor,
    apply_routed_scaling_factor_on_output
):
    sigmoid_vals = torch.sigmoid(input_tensor)
    biased_scores = sigmoid_vals + bias.unsqueeze(0)

    _, top_indices = torch.topk(biased_scores, topk, dim=-1)

    output_scores = torch.gather(sigmoid_vals, 1, top_indices)

    if renormalize:
        score_sum = output_scores.sum(dim=-1, keepdim=True)
        output_scores = output_scores / score_sum.clamp(min=torch.finfo(score_sum.dtype).eps)
        if apply_routed_scaling_factor_on_output:
            output_scores *= routed_scaling_factor

    return output_scores, top_indices.int()


@pytest.mark.parametrize("m, topk, renormalize", configs)
def test_kimi_k2_moe_fused_gate(m, topk, renormalize):
    routed_scaling_factor = 2.0
    apply_scaling = True

    input_tensor = torch.randn(m, NUM_EXPERTS, dtype=torch.float32, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device="cuda") * 0.1

    output, indices = kimi_k2_moe_fused_gate(
        input_tensor, bias, topk, renormalize,
        routed_scaling_factor, apply_scaling,
    )

    ref_output, ref_indices = _reference_kimi_k2_moe_fused_gate(
        input_tensor, bias, topk, renormalize,
        routed_scaling_factor, apply_scaling,
    )

    # Expert selection order may differ; compare as sets per row
    for i in range(m):
        jit_set = set(indices[i].tolist())
        ref_set = set(ref_indices[i].tolist())
        assert jit_set == ref_set, (
            f"Row {i}: JIT selected {jit_set} but ref selected {ref_set}"
        )

    # Sort by expert index for score comparison
    jit_sorted = torch.sort(indices, dim=-1)
    ref_sorted = torch.sort(ref_indices, dim=-1)
    jit_scores_sorted = torch.gather(output, 1, jit_sorted.indices)
    ref_scores_sorted = torch.gather(ref_output, 1, ref_sorted.indices)

    torch.testing.assert_close(
        jit_scores_sorted, ref_scores_sorted, rtol=1e-4, atol=1e-4
    )


def test_kimi_k2_moe_fused_gate_wrong_experts():
    input_tensor = torch.randn(4, 256, dtype=torch.float32, device="cuda")
    bias = torch.randn(256, dtype=torch.float32, device="cuda")

    with pytest.raises(Exception):
        kimi_k2_moe_fused_gate(input_tensor, bias, 2, False, 1.0, False)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
