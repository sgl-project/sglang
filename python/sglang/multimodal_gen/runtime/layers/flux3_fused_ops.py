# SPDX-License-Identifier: Apache-2.0
"""FLUX.3 projection operators with PrimsTS and generic implementations.

Each operator is a :class:`~sglang.kernels.fused_op.BaseFusedOp`.
``forward_native`` is the generic SGLang path: the layer's own quant method
plus SGLang's fused QK-norm/RoPE and SwiGLU kernels, on any device and format.
``forward_flashinfer`` runs the FlashInfer PrimsTS epilogue that the layer's
FLUX.3 quant method (:mod:`...quantization.flux3_prims_ts`) prepared at load
time. FlashInfer is selected per call, only on SM10x and only for layers whose
method planned the matching role, so call sites never branch on the backend.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.spec import FormatSignature, KernelBackend
from sglang.multimodal_gen.runtime.layers.layernorm import apply_qk_norm_rope
from sglang.multimodal_gen.runtime.layers.quantization.flux3_prims_ts import (
    FLUX3_PRIMS_TS_CAPABILITIES,
    Flux3PrimsTsRole,
)


def _prims_ts_method(layer: nn.Module, role: Flux3PrimsTsRole):
    """The layer's quant method if it planned ``role`` for PrimsTS, else None."""
    method = getattr(layer, "quant_method", None)
    if getattr(method, "prims_ts_role", None) is role:
        return method
    return None


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    from sgl_kernel import silu_and_mul

    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    silu_and_mul(x.view(-1, 2 * d), out.view(-1, d))
    return out


def _unpack_qkv(qkv: torch.Tensor, local_heads: int):
    """Return Q/K/V views from packed or separate-section GEMM output."""
    if qkv.ndim == 4 and qkv.shape[0] == 3:
        _, batch, seqlen, width = qkv.shape
        return tuple(
            tensor.view(batch, seqlen, local_heads, width // local_heads)
            for tensor in qkv.unbind(0)
        )
    batch, seqlen, packed = qkv.shape
    head_dim = packed // (3 * local_heads)
    return qkv.view(batch, seqlen, 3, local_heads, head_dim).unbind(dim=2)


class _Flux3PrimsTsOp(BaseFusedOp):
    priority = (KernelBackend.FLASHINFER, KernelBackend.TORCH)
    capabilities = {KernelBackend.FLASHINFER: FLUX3_PRIMS_TS_CAPABILITIES}
    format_signature = FormatSignature(
        supported_dtypes=("bfloat16",),
        description="FLUX.3 projection; layer-prepared PrimsTS weights",
    )

    def _flashinfer_eligible(
        self, layer: nn.Module, x: torch.Tensor, role: Flux3PrimsTsRole
    ) -> bool:
        return (
            x.is_cuda
            and _prims_ts_method(layer, role) is not None
            and super().backend_eligible(KernelBackend.FLASHINFER)
        )


class Flux3LinearOp(_Flux3PrimsTsOp):
    """``layer(x)`` for an attention, MLP-down, or unfused QKV projection."""

    op = "gemm.flux3_linear"
    descriptions = {
        KernelBackend.FLASHINFER: "PrimsTS FP8/NVFP4 dense GEMM.",
        KernelBackend.TORCH: "Generic SGLang linear (layer quant method).",
    }

    def backend_eligible(self, backend, layer, x, quantized=None) -> bool:
        if backend is KernelBackend.FLASHINFER:
            return self._flashinfer_eligible(layer, x, Flux3PrimsTsRole.LINEAR)
        return super().backend_eligible(backend)

    def forward_native(self, layer, x, quantized=None) -> torch.Tensor:
        # The generic method quantizes its own input; ``quantized`` is a
        # PrimsTS operand and does not apply here.
        out, _ = layer(x)
        return out

    def forward_flashinfer(self, layer, x, quantized=None) -> torch.Tensor:
        return layer.quant_method.apply_prims_ts_linear(layer, x, quantized)


class Flux3QKVQKNormRopeOp(_Flux3PrimsTsOp):
    """Packed QKV projection, Q/K RMSNorm, and RoPE -> ``(B, L, H, D)`` q, k, v."""

    op = "gemm.flux3_qkv_qknorm_rope"
    descriptions = {
        KernelBackend.FLASHINFER: "PrimsTS QKV GEMM with QK-norm/RoPE epilogue.",
        KernelBackend.TORCH: "FLUX.3 linear, then fused QK-norm/RoPE kernel.",
    }

    def backend_eligible(
        self,
        backend,
        layer,
        x,
        norm,
        cos_sin_cache,
        positions,
        local_heads,
        quantized=None,
    ) -> bool:
        if backend is KernelBackend.FLASHINFER:
            return self._flashinfer_eligible(layer, x, Flux3PrimsTsRole.QKV_QKNORM_ROPE)
        return super().backend_eligible(backend)

    def forward_native(
        self,
        layer,
        x,
        norm,
        cos_sin_cache,
        positions,
        local_heads,
        quantized=None,
    ):
        q, k, v = _unpack_qkv(flux3_linear(layer, x, quantized), local_heads)
        q, k = apply_qk_norm_rope(
            q,
            k,
            norm.query_norm,
            norm.key_norm,
            q.shape[-1],
            cos_sin_cache,
            is_neox=False,
            positions=positions,
            allow_strided_qk=True,
        )
        return q, k, v

    def forward_flashinfer(
        self,
        layer,
        x,
        norm,
        cos_sin_cache,
        positions,
        local_heads,
        quantized=None,
    ):
        qkv = layer.quant_method.apply_prims_ts_qkv_qknorm_rope(
            layer,
            x,
            norm.query_norm.weight,
            norm.key_norm.weight,
            cos_sin_cache,
            positions,
            quantized,
        )
        return _unpack_qkv(qkv, local_heads)


class Flux3GatedMLPOp(_Flux3PrimsTsOp):
    """``mlp_out(silu(gate) * value)`` with ``[gate | value] = mlp_in(x)``."""

    op = "gemm.flux3_gated_mlp"
    descriptions = {
        KernelBackend.FLASHINFER: "PrimsTS MLP-up with SwiGLU epilogue.",
        KernelBackend.TORCH: "Generic MLP-up, silu_and_mul, FLUX.3 linear.",
    }

    def backend_eligible(self, backend, mlp_in, mlp_out, x, quantized=None) -> bool:
        if backend is KernelBackend.FLASHINFER:
            return self._flashinfer_eligible(mlp_in, x, Flux3PrimsTsRole.SWIGLU)
        return super().backend_eligible(backend)

    def forward_native(self, mlp_in, mlp_out, x, quantized=None) -> torch.Tensor:
        mlp, _ = mlp_in(x)
        return flux3_linear(mlp_out, _silu_and_mul(mlp))

    def forward_flashinfer(self, mlp_in, mlp_out, x, quantized=None) -> torch.Tensor:
        up = mlp_in.quant_method
        down = _prims_ts_method(mlp_out, Flux3PrimsTsRole.LINEAR)
        if (
            down is not None
            and up.packed_activation_format is not None
            and down.packed_activation_format == up.packed_activation_format
        ):
            # Hand MLP-down the packed SwiGLU output without a BF16 round trip.
            packed = up.apply_prims_ts_swiglu(mlp_in, x, quantized, packed_output=True)
            return down.apply_prims_ts_linear_from_packed(mlp_out, packed)
        return flux3_linear(mlp_out, up.apply_prims_ts_swiglu(mlp_in, x, quantized))


_FLUX3_LINEAR = register_fused_op(Flux3LinearOp(), __name__, "_FLUX3_LINEAR")
_FLUX3_QKV_QKNORM_ROPE = register_fused_op(
    Flux3QKVQKNormRopeOp(), __name__, "_FLUX3_QKV_QKNORM_ROPE"
)
_FLUX3_GATED_MLP = register_fused_op(Flux3GatedMLPOp(), __name__, "_FLUX3_GATED_MLP")


def flux3_linear(layer: nn.Module, x: torch.Tensor, quantized=None) -> torch.Tensor:
    """Run one FLUX.3 projection; ``quantized`` is an optional shared operand."""
    return _FLUX3_LINEAR(layer, x, quantized)


def flux3_qkv_qknorm_rope(
    layer: nn.Module,
    x: torch.Tensor,
    norm: nn.Module,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    local_heads: int,
    quantized=None,
):
    """Project packed QKV and apply Q/K RMSNorm plus interleaved RoPE."""
    return _FLUX3_QKV_QKNORM_ROPE(
        layer, x, norm, cos_sin_cache, positions, local_heads, quantized
    )


def flux3_gated_mlp(
    mlp_in: nn.Module, mlp_out: nn.Module, x: torch.Tensor, quantized=None
) -> torch.Tensor:
    """Run FLUX.3's gated (SwiGLU) MLP."""
    return _FLUX3_GATED_MLP(mlp_in, mlp_out, x, quantized)


def flux3_shared_activation(x: torch.Tensor, qkv: nn.Module, mlp_in: nn.Module):
    """Quantize ``x`` once when QKV and fused MLP-up both run PrimsTS on it.

    Returns ``None`` when reuse does not apply; consumers then quantize their
    own input.
    """
    if not x.is_cuda:
        return None
    up = _prims_ts_method(mlp_in, Flux3PrimsTsRole.SWIGLU)
    qkv_method = getattr(qkv, "quant_method", None)
    if (
        up is None
        or getattr(qkv_method, "prims_ts_role", None) is None
        or qkv_method.activation_format != up.activation_format
    ):
        return None
    return up.quantize_prims_ts_activation(mlp_in, x)
