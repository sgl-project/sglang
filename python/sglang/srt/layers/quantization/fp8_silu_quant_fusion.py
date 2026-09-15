"""SiLU+mul and the down_proj activation quant as one aiter kernel, on ROCm.

An FP8 MLP computes ``act_fn(gate_up)`` in bf16 and then lets ``down_proj`` quantize it
per token. aiter's ``silu_and_mul_quant`` does both in one pass and hands ``down_proj``
the ``(fp8, per-token scale)`` tuple, which :func:`owns_input` and :func:`apply_linear`
route to the aiter PTPC GEMM -- the only FP8 GEMM here that takes a scale per row.

:func:`claim_mlp` claims only an MLP whose ``down_proj`` is online per-token FP8, in
practice the layers ``--enable-dense-fp8`` promotes. Anywhere else :func:`owns_act_fn`
is False and the MLP keeps its bf16 activation and quantizing ``down_proj``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear
from sglang.srt.utils import get_bool_env_var, is_hip

IS_HIP = is_hip()

aiter = None
if IS_HIP and get_bool_env_var("SGLANG_USE_AITER"):
    try:
        import aiter
        from aiter import dtypes as aiter_dtypes
    except Exception:
        aiter = None

ENABLED = aiter is not None

_OWNS_ATTR = "_fp8_silu_quant_fused"


def _flag_enabled() -> bool:
    from sglang.srt.runtime_context import get_exec

    try:
        return get_exec().kernel.enable_dense_fp8
    except ValueError:
        return False


def _eligible(mlp: torch.nn.Module, quant_config: object) -> bool:
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

    if not _flag_enabled():
        return False
    method = getattr(mlp.down_proj, "quant_method", None)
    # use_aiter_fp8_per_token is set in __init__ and implies the bpreshuffle per-token
    # path; use_per_token_if_dynamic only appears in process_weights_after_loading.
    return (
        isinstance(method, Fp8LinearMethod)
        and getattr(method, "use_aiter_fp8_per_token", False)
        and getattr(method.quant_config, "activation_scheme", None) == "dynamic"
        and getattr(quant_config, "dequantization_config", None) is None
    )


def claim_mlp(mlp: torch.nn.Module, quant_config: object) -> None:
    """Note on ``mlp`` whether its activation may fold into down_proj's quant."""
    setattr(mlp, _OWNS_ATTR, ENABLED and _eligible(mlp, quant_config))


def owns_act_fn(mlp: torch.nn.Module, gate_up: torch.Tensor) -> bool:
    """Whether ``mlp`` runs its activation and down_proj quant as one kernel."""
    # An empty batch would hand the kernel a zero-row GEMM.
    return getattr(mlp, _OWNS_ATTR, False) and gate_up.shape[0] > 0


def silu_and_mul_quant(gate_up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``silu(gate) * up`` quantized per token: the input ``down_proj`` then expects."""
    num_tokens = gate_up.shape[0]
    half = gate_up.shape[-1] // 2
    out_fp8 = torch.empty(
        (num_tokens, half), dtype=aiter_dtypes.fp8, device=gate_up.device
    )
    scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=gate_up.device)
    aiter.silu_and_mul_quant(out_fp8, gate_up, scale, half)
    return out_fp8, scale


def owns_input(linear_method, x) -> bool:
    """Whether ``x`` is this fusion's ``(fp8, per-token scale)`` tuple.

    A per-tensor tuple from the fused RMSNorm+quant kernels carries a 0-D or 1-D scale
    and stays on the ``apply_fp8_linear`` path below it.
    """
    return (
        ENABLED
        and linear_method.use_per_token_if_dynamic
        and isinstance(x, tuple)
        and x[1].dim() >= 2
    )


def apply_linear(
    linear_method,
    layer: torch.nn.Module,
    x: Tuple[torch.Tensor, torch.Tensor],
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """``layer``'s GEMM over an activation this fusion already quantized."""
    # The PTPC GEMM wants the weight as (N, K); Fp8LinearMethod stores it as (K, N).
    return apply_fp8_ptpc_linear(
        input=(x[0], x[1]),
        weight=layer.weight.T,
        weight_scale=layer.weight_scale,
        input_scale=layer.input_scale,
        bias=bias,
        cutlass_fp8_supported=linear_method.cutlass_fp8_supported,
        use_per_token_if_dynamic=linear_method.use_per_token_if_dynamic,
    )
