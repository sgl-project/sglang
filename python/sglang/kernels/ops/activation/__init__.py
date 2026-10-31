"""Activation kernels.

``silu_and_mul`` (``silu(x[..., :d]) * x[..., d:]``) is a
:class:`~sglang.kernels.fused_op.BaseFusedOp` with a pure-``torch`` reference
(``forward_native``) plus AOT (``sgl_kernel``), JIT CUDA and aiter backends
behind one ``(input, out)`` signature. The JIT backend additionally accepts
``expert_ids`` / ``expert_step`` — call ``forward_jit`` directly when those are
needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch


class SiluAndMulOp(BaseFusedOp):
    """``out = silu(input[..., :d]) * input[..., d:]`` with ``d = input.shape[-1] // 2``."""

    op = "activation.silu_and_mul"
    # AOT spans CUDA+HIP; JIT is CUDA; AITER is an opt-in HIP path. By priority,
    # CUDA resolves to JIT and HIP resolves to AOT (matching production
    # defaults); AITER is registered and HIP-eligible but sits below AOT, so it
    # is available for explicit/forced selection without changing the default.
    priority = (
        KernelBackend.JIT,
        KernelBackend.AOT,
        KernelBackend.AITER,
        KernelBackend.TORCH,
    )
    capabilities = {
        KernelBackend.AOT: frozenset(
            {CapabilityRequirement.CUDA, CapabilityRequirement.HIP}
        ),
        KernelBackend.JIT: CUDA_ONLY,
        KernelBackend.AITER: frozenset({CapabilityRequirement.HIP}),
    }

    def forward_native(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        import torch.nn.functional as F

        d = input.shape[-1] // 2
        result = F.silu(input[..., :d]) * input[..., d:]
        if out is None:
            return result
        out.copy_(result)
        return out

    def forward_aot(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        from sgl_kernel import silu_and_mul as aot_silu_and_mul

        return aot_silu_and_mul(input, out)

    def forward_jit(
        self,
        input: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        expert_ids: Optional[torch.Tensor] = None,
        expert_step: int = 1,
    ) -> torch.Tensor:
        from sglang.kernels.ops.activation.activation import (
            silu_and_mul as jit_silu_and_mul,
        )

        return jit_silu_and_mul(input, out, expert_ids, expert_step)

    def forward_aiter(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        import torch
        from aiter import silu_and_mul as _aiter_silu_and_mul

        d = input.shape[-1] // 2
        if out is None:
            out = torch.empty(
                (*input.shape[:-1], d), dtype=input.dtype, device=input.device
            )
        # aiter's ROCm silu_and_mul: (out, input, limit); limit=0.0 = no clamp,
        # matching the standard (unclamped) gated-SiLU used elsewhere.
        _aiter_silu_and_mul(out, input, 0.0)
        return out


_SILU_AND_MUL = register_fused_op(SiluAndMulOp(), __name__, "_SILU_AND_MUL")


def silu_and_mul(
    input: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out = silu(input[..., :d]) * input[..., d:]`` with ``d = input.shape[-1] // 2``."""
    return _SILU_AND_MUL(input, out)


def situ_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:
    from .kimi_k3_situ_and_mul import situ_and_mul as impl

    return impl(input, out, beta, linear_beta)


def situ_and_mul_masked(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
    topk: int = 8,
    expected_m: Optional[int] = None,
) -> torch.Tensor:
    from .kimi_k3_situ_and_mul import situ_and_mul_masked as impl

    return impl(input, out, masked_m, beta, linear_beta, topk, expected_m)


def situ_and_mul_post_quant_mxfp4(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
) -> None:
    from .kimi_k3_situ_and_mul import situ_and_mul_post_quant_mxfp4 as impl

    impl(input, output, output_scale, beta, linear_beta)


def situ_and_mul_masked_post_quant_mxfp4(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: Optional[float],
    topk: int = 8,
    expected_m: Optional[int] = None,
) -> None:
    from .kimi_k3_situ_and_mul import situ_and_mul_masked_post_quant_mxfp4 as impl

    impl(input, output, output_scale, masked_m, beta, linear_beta, topk, expected_m)


__all__ = [
    "situ_and_mul",
    "situ_and_mul_masked",
    "situ_and_mul_post_quant_mxfp4",
    "situ_and_mul_masked_post_quant_mxfp4",
    "SiluAndMulOp",
    "silu_and_mul",
]


# Triton kernel migrated into this group (from layers/triton_ops/softcap);
# registered for inventory. Import it from its module.
register_kernel(
    KernelSpec(
        op="activation.softcap_inplace_logits",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.activation.softcap:softcap_inplace_logits",
    )
)


register_kernel(
    KernelSpec(
        op="activation.softcap_to_float32_logits",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.activation.softcap:softcap_to_float32_logits",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="activation.cast_to_float32_logits",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.activation.softcap:cast_to_float32_logits",
        capabilities=CUDA_ONLY,
    )
)


# Kernels introduced with Kimi-K3, inventoried by logical operator group.
register_kernel(
    KernelSpec(
        op="activation.situ_and_mul",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.activation.kimi_k3_situ_and_mul:situ_and_mul",
        capabilities=CUDA_ONLY,
    )
)


# The new SiTU producers still belong to activation; device implementations stay
# lazily imported.
for _fn in (
    "situ_and_mul_masked",
    "situ_and_mul_post_quant_mxfp4",
    "situ_and_mul_masked_post_quant_mxfp4",
):
    register_kernel(
        KernelSpec(
            op=f"activation.{_fn}",
            backend=KernelBackend.JIT,
            target=f"sglang.kernels.ops.activation.kimi_k3_situ_and_mul:{_fn}",
            capabilities=CUDA_ONLY,
        )
    )
del _fn


# MXFP8-quantizing SwiGLU-OAI producers; registered for inventory.
for _fn in ("swiglu_oai_mxfp8", "swiglu_oai_mxfp8_deepgemm"):
    register_kernel(
        KernelSpec(
            op=f"activation.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.activation.swiglu_mxfp8_swizzled:{_fn}",
        )
    )
del _fn
# SAIL goes through the CUDA interface; the dedicated-instruction constraint is
# checked by each implementation's cached factory.
for _fn in (
    "silu_and_mul_post_quant_mxfp4",
    "silu_and_mul_post_per_token_quant_fp8",
    "silu_and_mul_masked_post_quant_mxfp4",
):
    register_kernel(
        KernelSpec(
            op=f"activation.{_fn}",
            backend=KernelBackend.JIT,
            target=f"sglang.kernels.ops.activation.silu_mul_quant:{_fn}",
            capabilities=CUDA_ONLY,
        )
    )
del _fn
for _fn in ("silu_and_mul_clamp", "silu_and_mul_contig_post_quant"):
    register_kernel(
        KernelSpec(
            op=f"activation.{_fn}",
            backend=KernelBackend.JIT,
            target=f"sglang.kernels.ops.activation.silu_and_mul_clamp:{_fn}",
            capabilities=frozenset(
                {CapabilityRequirement.CUDA, CapabilityRequirement.HIP}
            ),
        )
    )
del _fn
