from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.kernels.jit import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_kimi_k2_moe_fused_gate_module() -> Module:
    return load_jit(
        "kimi_k2_moe_fused_gate",
        cuda_files=["trtllm_lora_temp/kimi_k2_moe_fused_gate.cuh"],
        cuda_wrappers=[
            ("kimi_k2_moe_fused_gate", "KimiK2MoEFusedGateKernel::run"),
        ],
    )


def kimi_k2_moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float | None = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kimi K2 MoE fused gate (num_expert_group=1, DeepSeek noaux_tc routing).

    Supports num_experts in {256, 384} and topk <= 8. input and bias are CUDA
    tensors of float32, bfloat16, or float16 (dtypes may differ between the two);
    they are widened to fp32 inside the kernel, so callers no longer need to
    upcast bf16/fp16 router logits or correction bias on the host. Returns
    (output_weights, expert_indices).
    """
    _supported = (torch.float32, torch.bfloat16, torch.float16)
    assert (
        input.dtype in _supported
    ), f"input must be float32/bfloat16/float16, got {input.dtype}"
    assert (
        bias.dtype in _supported
    ), f"bias must be float32/bfloat16/float16, got {bias.dtype}"
    assert input.ndim == 2, "input must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert input.size(1) == bias.size(0), "input and bias must have same num_experts"

    num_rows = input.size(0)
    device = input.device

    output = torch.empty(num_rows, topk, dtype=torch.float32, device=device)
    indices = torch.empty(num_rows, topk, dtype=torch.int32, device=device)

    module = _jit_kimi_k2_moe_fused_gate_module()
    module.kimi_k2_moe_fused_gate(
        input,
        bias,
        output,
        indices,
        topk,
        renormalize,
        float(routed_scaling_factor) if routed_scaling_factor is not None else 1.0,
        apply_routed_scaling_factor_on_output,
    )

    return output, indices
