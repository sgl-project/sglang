"""SAIL activations for the DeepGEMM MoE runners.

Every platform runs the plain SiLU between GEMM1 and GEMM2. The models SAIL
serves also need the oai-swiglu gating, so that variant lives here and the
shared runner keeps its upstream activation path.
"""

import torch


def apply_oai_swiglu(
    gateup_output: torch.Tensor,
    gemm1_alpha: float,
    gemm1_clamp_limit: float,
    gate_up_interleaved: bool,
) -> torch.Tensor:
    """Run FP32 oai-swiglu according to the weight layout; the BF16 upcast keeps the
    clamp from mutating the GEMM1 output."""
    assert gateup_output.dtype == torch.bfloat16, (
        f"expected bf16 gateup_output, got {gateup_output.dtype}"
    )
    if gate_up_interleaved:
        gate = gateup_output[..., ::2].float()
        up = gateup_output[..., 1::2].float()
    else:
        n = gateup_output.shape[-1]
        gate = gateup_output[..., : n // 2].float()
        up = gateup_output[..., n // 2 :].float()
    gate = gate.clamp_(max=gemm1_clamp_limit)
    up = up.clamp_(min=-gemm1_clamp_limit, max=gemm1_clamp_limit)
    return (gate * torch.sigmoid(gate * gemm1_alpha) * (up + 1)).to(torch.bfloat16)
