"""Cake Kimi-K3 SiTU fused MoE (TP8-local, NVFP4 group-16) via FlashInfer.

FlashInfer entries: ``flashinfer.fused_moe.cutlass_fused_moe(backend="cake")``,
``flashinfer.fused_moe.cutlass_fused_moe_workspace_size(backend="cake")`` and
``flashinfer.fused_moe.cake_fused_moe_prepare_workspace`` (implementation
``flashinfer.fused_moe.cake_kimi_k3_situ``, JIT module
``flashinfer.jit.cake_kimi_k3_situ``). Contract at FlashInfer ``46340689a5ab``:
fixed geometry ``H=3584, I=384, E=896, top_k=16``; BF16 ``input [T, 3584]``,
int32 ``token_selected_experts [T, 16]``, BF16 ``token_final_scales [T, 16]``,
uint8 packed NVFP4 weights ``fc1 [896, 768, 1792]`` / ``fc2 [896, 3584, 192]``
in the TRTLLM shuffled group-16 layout (``TrtllmFp4Config.prepare_weights``
with ``activation=SiTU()``), six NVFP4 ``quant_scales`` (``[1]`` f32 input
scale, ``[896, 768, 224]`` FC1 block scales, ``[896]`` f32 FC1 dequant,
``[896]`` f32 FC2 input scale, ``[896, 3584, 24]`` FC2 block scales, ``[896]``
f32 FC2 dequant), BF16 ``output [T, 3584]`` and a caller-owned 1-D uint8
``workspace_buffer`` (128-byte aligned). ``activation_type=Situ``,
``tp_size=8`` (local weights), ``ep_size=1``, ``use_fused_finalize=True``;
``1 <= T <= 16384``. Built for sm_100a / sm_103a.

CUDA graphs: size the workspace once with ``..._workspace_size`` and call
``..._prepare_workspace(workspace, num_tokens)`` OUTSIDE graph capture for
every token count that will be submitted (the prepare stores a per-token-count
program key on the buffer and loads the module); the fused call itself is one
packed FFI submission and is capturable. The prepare helper allocates nothing.

Not supported here: other geometries, biases, ``input_sf`` / swiglu
parameters, EP, all-to-all, min-latency, block-scale / w4-group / mxfp8 /
packed / humming variants, ``enable_pdl=False``, ``profile_ids``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

from sglang.kernels.cake_kernels._support import SM100, SM103, cuda_tensor_on
from sglang.kernels.cake_kernels.moe_common import (
    contiguous_cuda,
    cuda_device_in,
    current_cuda_index,
    modules_available,
)

if TYPE_CHECKING:
    import torch

FI_MODULE = "flashinfer.fused_moe.cake_kimi_k3_situ"
FI_JIT_MODULE = "flashinfer.jit.cake_kimi_k3_situ"
ARCHS = (SM100, SM103)
HIDDEN = 3584
INTERMEDIATE = 384
NUM_EXPERTS = 896
TOP_K = 16
TP_SIZE = 8
MAX_TOKENS = 16384
WEIGHT_LAYOUT = "trtllm_shuffled_nvfp4_group16"


def supports_kimi_k3_situ_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    quant_scales: Sequence[torch.Tensor],
) -> bool:
    """Admission check mirroring the FlashInfer SiTU contract; never raises."""
    import torch

    try:
        if not (
            modules_available(FI_MODULE, FI_JIT_MODULE)
            and cuda_tensor_on(input, ARCHS)
            and input.dtype == torch.bfloat16
            and input.ndim == 2
            and input.is_contiguous()
            and 1 <= input.shape[0] <= MAX_TOKENS
            and input.shape[1] == HIDDEN
        ):
            return False
        num_tokens = int(input.shape[0])
        if not (
            contiguous_cuda(
                token_selected_experts, shape=(num_tokens, TOP_K), dtype=torch.int32
            )
            and contiguous_cuda(
                token_final_scales, shape=(num_tokens, TOP_K), dtype=torch.bfloat16
            )
            and contiguous_cuda(
                fc1_expert_weights,
                shape=(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 2),
                dtype=torch.uint8,
            )
            and contiguous_cuda(
                fc2_expert_weights,
                shape=(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 2),
                dtype=torch.uint8,
            )
        ):
            return False
        if not isinstance(quant_scales, (list, tuple)) or len(quant_scales) != 6:
            return False
        qx, sf1, decode1, qa, sf2, decode2 = quant_scales
        sf_dtype = (torch.uint8, torch.float8_e4m3fn)
        return (
            contiguous_cuda(qx, dtype=torch.float32)
            and qx.numel() == 1
            and contiguous_cuda(
                sf1,
                shape=(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN // 16),
                dtype=sf_dtype,
            )
            and contiguous_cuda(
                sf2, shape=(NUM_EXPERTS, HIDDEN, INTERMEDIATE // 16), dtype=sf_dtype
            )
            and all(
                contiguous_cuda(t, shape=(NUM_EXPERTS,), dtype=torch.float32)
                for t in (decode1, qa, decode2)
            )
            and len(
                {
                    t.device
                    for t in (
                        input,
                        token_selected_experts,
                        token_final_scales,
                        fc1_expert_weights,
                        fc2_expert_weights,
                        *quant_scales,
                    )
                }
            )
            == 1
        )
    except Exception:
        return False


def supports_kimi_k3_situ_workspace(
    workspace_buffer: torch.Tensor, num_tokens: int
) -> bool:
    """``True`` when ``workspace_buffer`` can be prepared for ``num_tokens``."""
    import torch

    try:
        return (
            modules_available(FI_MODULE, FI_JIT_MODULE)
            and isinstance(num_tokens, int)
            and not isinstance(num_tokens, bool)
            and 1 <= num_tokens <= MAX_TOKENS
            and cuda_tensor_on(workspace_buffer, ARCHS)
            and workspace_buffer.dtype in (torch.uint8, torch.int8)
            and workspace_buffer.ndim == 1
            and workspace_buffer.is_contiguous()
            and workspace_buffer.data_ptr() % 128 == 0
        )
    except Exception:
        return False


def kimi_k3_situ_fused_moe_workspace_size(
    max_num_tokens: int,
    *,
    tp_rank: int = 0,
    device: Optional[torch.device] = None,
) -> int:
    """Bytes of the SiTU workspace for up to ``max_num_tokens`` (host metadata only)."""
    import torch

    from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
    from flashinfer.tllm_enums import ActivationType

    return cutlass_fused_moe_workspace_size(
        max_num_tokens,
        HIDDEN,
        INTERMEDIATE,
        NUM_EXPERTS,
        TOP_K,
        x_dtype=torch.bfloat16,
        weight_dtype=torch.uint8,
        output_dtype=torch.bfloat16,
        activation_type=ActivationType.Situ,
        tp_size=TP_SIZE,
        tp_rank=tp_rank,
        ep_size=1,
        ep_rank=0,
        use_fused_finalize=True,
        device=device,
        backend="cake",
    )


def kimi_k3_situ_fused_moe_prepare_workspace(
    workspace_buffer: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Prepare ``workspace_buffer`` for ``num_tokens`` outside graph capture.

    Returns the same buffer. Must run once per token count on the submitting
    stream before :func:`kimi_k3_situ_fused_moe` is called for that count.
    """
    from flashinfer.fused_moe import cake_fused_moe_prepare_workspace

    return cake_fused_moe_prepare_workspace(
        workspace_buffer,
        num_tokens,
        backend="cake",
        weight_layout=WEIGHT_LAYOUT,
    )


def kimi_k3_situ_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    quant_scales: List[torch.Tensor],
    *,
    output: torch.Tensor,
    workspace_buffer: torch.Tensor,
    tp_rank: int = 0,
    enable_pdl: Optional[bool] = None,
    situ_beta: Optional[torch.Tensor] = None,
    situ_linear_beta: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
) -> torch.Tensor:
    """Forward to ``cutlass_fused_moe(backend="cake")``; returns ``output``.

    ``situ_beta`` / ``situ_linear_beta`` (f32 ``[896]``) override the prepared
    per-expert 4.0 / 25.0 defaults. ``enable_pdl`` must not be ``False``.
    """
    import torch

    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.tllm_enums import ActivationType

    return cutlass_fused_moe(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc2_expert_weights,
        torch.bfloat16,
        list(quant_scales),
        tp_size=TP_SIZE,
        tp_rank=tp_rank,
        ep_size=1,
        ep_rank=0,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        enable_pdl=enable_pdl,
        activation_type=ActivationType.Situ,
        use_fused_finalize=True,
        workspace_buffer=workspace_buffer,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        backend="cake",
    )


def device_supports_kimi_k3_situ(device: Optional[torch.device] = None) -> bool:
    """Arch + module probe without tensors (for allocation-time decisions)."""
    try:
        return modules_available(FI_MODULE, FI_JIT_MODULE) and cuda_device_in(
            current_cuda_index(device), ARCHS
        )
    except Exception:
        return False
