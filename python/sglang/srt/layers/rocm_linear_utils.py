from functools import lru_cache
from typing import Optional, Tuple

import torch
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cat_and_cache_mla
from aiter.ops.triton.fused_qk_concat import fused_qk_rope_cat
from aiter.tuned_gemm import tgemm

from sglang.kernels.ops.gemm.router_gemv_hip import rocm_router_gemv_split_k
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.runtime_context import get_exec, get_forward, get_lora
from sglang.srt.utils import is_gfx95_supported

try:
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import (
        gemm_a16w16 as _aiter_auto_bf16_gemm,
    )
except ImportError:
    _aiter_auto_bf16_gemm = None

__all__ = [
    "fused_fp8_bmm_rope_cat_and_cache_mla",
    "fused_qk_rope_cat",
    "fused_qk_rope_cat_and_cache_mla",
]

# This module is imported wherever AITER is on, gfx942 included, but the fused
# bmm+rope+cache op is gfx95-only. Import it behind the same predicate its one
# caller gates on, so an aiter build without the op cannot take down every
# DeepSeek import on another card. The name stays bound either way.
if is_gfx95_supported():
    from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
        fused_fp8_bmm_rope_cat_and_cache_mla,
    )
else:
    fused_fp8_bmm_rope_cat_and_cache_mla = None


def aiter_dsv3_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """Use aiter tuned GEMM dispatcher (tgemm.mm) to automatically select the GEMM kernel."""
    return tgemm.mm(hidden_states, weight.detach(), otype=hidden_states.dtype)


@lru_cache(maxsize=8)
def _query_auto_device_supported(device: torch.device) -> bool:
    props = torch.cuda.get_device_properties(device)
    return (
        props.gcnArchName.split(":")[0] == "gfx950"
        and props.multi_processor_count == 256
    )


def aiter_bf16_query_gemm(layer, q: torch.Tensor) -> Optional[torch.Tensor]:
    """Use the ordinary AITER automatic GEMM for qualified canonical q_b operands.

    None keeps the existing linear caller. Quantized, shuffled and adapter-owned
    weights retain their caller; no packed storage is reinterpreted here.
    """
    if (
        _aiter_auto_bf16_gemm is None
        or get_exec().kernel.bf16_gemm_backend != "auto"
        or get_exec().deterministic.enable_deterministic_inference
        or get_lora().enable_lora
        or get_forward().sp_active
        or type(layer.quant_method) is not UnquantizedLinearMethod
        or getattr(layer, "set_lora", False)
        or getattr(layer, "scheme", None) is not None
        or layer.bias is not None
        or layer.gather_output
        or type(q) is not torch.Tensor
        or q.ndim != 2
        or q.shape[0] not in (64, 128)
        or q.shape[1] != 2048
    ):
        return None
    weight = layer.weight
    if (
        type(weight.data) is not torch.Tensor
        or tuple(weight.shape) != (2048, 2048)
        or q.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or q.device.type != "cuda"
        or q.device != weight.device
        or q.device.index != torch.cuda.current_device()
        or q.stride() != (2048, 1)
        or weight.stride() != (2048, 1)
        or q.data_ptr() % 16
        or weight.data_ptr() % 16
        or getattr(weight, "is_shuffled", False) is not False
        or getattr(weight, "aiter_layout", None) is not None
        or not _query_auto_device_supported(q.device)
    ):
        return None
    return _aiter_auto_bf16_gemm(q, weight, dtype=torch.bfloat16)


def rocm_dsv3_router_split_k(
    gate, hidden_states: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """The ROCm decode router for gate (a MoEGate): an fp32 logits buffer plus
    the split-K partials whose fixed-order sum fills it, or None when gate.forward
    applies. Only TopK.forward_cuda(..., router_logits_partials=partials) may read
    the buffer: it sums the partials into it inside the fused gate launch."""
    num_tokens = hidden_states.shape[0]
    if not 0 < num_tokens <= gate.rocm_router_max_tokens:
        return None
    if get_exec().deterministic.enable_deterministic_inference:
        return None
    partials = rocm_router_gemv_split_k(hidden_states, gate.weight)
    logits = torch.empty(
        (num_tokens, gate.weight.shape[0]),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return logits, partials


def get_dsv3_gemm_output_zero_allocator_size(
    n_routed_experts: int, num_moe_layers: int, allocate_size: int, embedding_dim: int
):
    if embedding_dim != 7168 or n_routed_experts != 256:
        return 0

    per_layer_size = 256 * (allocate_size + n_routed_experts)

    return num_moe_layers * per_layer_size
