import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")


def ref_quant_scatter(hidden_states, src2dst, topk_ids_flat, gateup_shape, scale_shape, topk, group_size=128):
    """Reference: separate quant + scatter."""
    num_tokens, hidden_size = hidden_states.shape
    fp8_max = 448.0
    eps = 1e-10

    # Step 1: quant
    flat = hidden_states.float().reshape(-1, group_size)
    absmax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    scales = (absmax / fp8_max).squeeze(-1)
    quantized = (flat / absmax * fp8_max).clamp(-fp8_max, fp8_max)
    quantized = quantized.reshape(num_tokens, hidden_size).to(torch.float8_e4m3fn)
    scales = scales.reshape(num_tokens, hidden_size // group_size)

    # Step 2: scatter
    gateup = torch.zeros(gateup_shape, dtype=torch.float8_e4m3fn, device=hidden_states.device)
    gateup_scale = torch.zeros(scale_shape, dtype=torch.float32, device=hidden_states.device)

    gateup_flat = gateup.view(-1, hidden_size)
    scale_flat = gateup_scale.view(-1, scales.shape[-1])

    for slot in range(num_tokens * topk):
        expert_id = topk_ids_flat[slot].item()
        if expert_id < 0:
            continue
        src_token = slot // topk
        dst_row = src2dst[slot].item()
        gateup_flat[dst_row] = quantized[src_token]
        scale_flat[dst_row] = scales[src_token]

    return gateup, gateup_scale


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 16])
@pytest.mark.parametrize("topk", [2, 4, 8])
@pytest.mark.parametrize("num_experts", [8, 64, 128])
def test_fused_quant_scatter(num_tokens, topk, num_experts):
    from sglang.jit_kernel.fused_quant_scatter import fused_quant_scatter

    hidden_size = 256
    group_size = 128
    m_max = (num_tokens * topk // num_experts + 1) * 2 + 4  # enough padding

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device="cuda")

    # Build src2dst: simple sequential mapping
    topk_ids_flat = topk_ids.flatten()
    src2dst = torch.zeros_like(topk_ids_flat)
    expert_counts = torch.zeros(num_experts, dtype=torch.int32)
    for i in range(topk_ids_flat.numel()):
        eid = topk_ids_flat[i].item()
        if eid >= 0:
            src2dst[i] = eid * m_max + expert_counts[eid]
            expert_counts[eid] += 1

    gateup_shape = (num_experts, m_max, hidden_size)
    scale_shape = (num_experts, m_max, hidden_size // group_size)

    # JIT kernel
    gateup = torch.zeros(gateup_shape, dtype=torch.float8_e4m3fn, device="cuda")
    gateup_scale = torch.zeros(scale_shape, dtype=torch.float32, device="cuda")
    fused_quant_scatter(
        hidden_states, src2dst, topk_ids_flat, gateup, gateup_scale,
        topk=topk, group_size=group_size,
    )

    # Reference
    ref_gateup, ref_scale = ref_quant_scatter(
        hidden_states, src2dst, topk_ids_flat,
        gateup_shape, scale_shape, topk, group_size,
    )

    # Compare scales
    torch.testing.assert_close(gateup_scale, ref_scale, rtol=1e-3, atol=1e-5)

    # Compare quantized values
    torch.testing.assert_close(gateup.float(), ref_gateup.float(), rtol=0.1, atol=1.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
