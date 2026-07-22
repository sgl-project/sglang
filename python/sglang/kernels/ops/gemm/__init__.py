"""GEMM and fused-GEMM kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_CUDA = frozenset({CapabilityRequirement.CUDA})

register_kernel(
    KernelSpec(
        op="gemm.fp8_scaled_mm",
        backend=KernelBackend.AOT,
        target="sgl_kernel:fp8_scaled_mm",
        format_signature=FormatSignature(
            supported_dtypes=("float8_e4m3fn",),
            description="C = (A_fp8 @ B_fp8) * scales_a * scales_b (+ bias)",
        ),
        description="FP8 scaled matmul (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.bmm_fp8",
        backend=KernelBackend.FLASHINFER,
        target="sglang.srt.layers.quantization.fp8_utils:bmm_fp8",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("float8_e4m3fn", "float8_e5m2"),
            description="batched (3D) per-tensor-scale FP8 matmul: D = A_fp8 @ B_fp8 * A_scale * B_scale",
        ),
        description="Batched FP8 matmul (flashinfer cuBLAS backend, torch.compile-safe wrapper).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_fused_a_gemm",
        backend=KernelBackend.AOT,
        target="sgl_kernel:dsv3_fused_a_gemm",
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="DeepSeek-V3 fused QKV-A GEMM",
        ),
        description="DeepSeek-V3 fused-A GEMM (sgl_kernel wheel).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_fused_a_gemm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm._jit_dsv3_fused_a_gemm:dsv3_fused_a_gemm",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="DeepSeek-V3 fused QKV-A GEMM (drop-in with AOT signature)",
        ),
        description="DeepSeek-V3 fused-A GEMM (sglang.jit_kernel).",
    )
)
register_kernel(
    KernelSpec(
        op="gemm.dsv3_router_gemm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.gemm._jit_dsv3_router_gemm:dsv3_router_gemm",
        capabilities=_CUDA,
        format_signature=FormatSignature(
            supported_dtypes=("bfloat16",),
            description="DeepSeek-V3 router GEMM; num_tokens in [1, 16]",
        ),
        description="DeepSeek-V3 router GEMM (sglang.jit_kernel, JIT-only).",
    )
)


def fp8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 scaled matmul: ``(mat_a @ mat_b) * scales_a * scales_b (+ bias)``."""
    return get_kernel("gemm.fp8_scaled_mm", KernelBackend.AOT)(
        mat_a, mat_b, scales_a, scales_b, out_dtype, bias
    )


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched (3D) per-tensor-scale FP8 matmul, via flashinfer's cuBLAS backend."""
    return get_kernel("gemm.bmm_fp8", KernelBackend.FLASHINFER)(
        A, B, A_scale, B_scale, dtype, out
    )


def dsv3_fused_a_gemm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DeepSeek-V3 fused QKV-A GEMM."""
    return get_kernel("gemm.dsv3_fused_a_gemm", KernelBackend.AOT)(mat_a, mat_b, output)


def dsv3_router_gemm(
    hidden_states: torch.Tensor,
    router_weights: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DeepSeek-V3 router GEMM (JIT-backed). ``out_dtype`` defaults to bfloat16."""
    impl = get_kernel("gemm.dsv3_router_gemm", KernelBackend.JIT)
    if out_dtype is None:
        return impl(hidden_states, router_weights, output=output)
    return impl(hidden_states, router_weights, out_dtype, output)


class Nvfp4GemmOp(BaseFusedOp):
    """NVFP4 dense GEMM: one FlashInfer ``mm_fp4`` sub-kernel per backend.

    All four backends share one underlying call (``flashinfer.mm_fp4``) with
    a different ``backend=`` string; each is still registered as its own
    :class:`KernelBackend` so callers (e.g. ``--fp4-gemm-backend``) can force
    one explicitly via ``forward(..., backend=KernelBackend.FLASHINFER_CUTLASS)``.
    A Marlin weight-only fallback exists but is a structurally different,
    dequant-based code path selected upstream by the caller, so it is not a
    backend of this op.
    """

    op = "gemm.nvfp4"
    priority = (
        KernelBackend.FLASHINFER_CUTEDSL,
        KernelBackend.FLASHINFER_CUTLASS,
        KernelBackend.FLASHINFER_TRTLLM,
        KernelBackend.FLASHINFER_CUDNN,
    )
    capabilities = {
        KernelBackend.FLASHINFER_CUTEDSL: _CUDA,
        KernelBackend.FLASHINFER_CUTLASS: _CUDA,
        KernelBackend.FLASHINFER_TRTLLM: _CUDA,
        KernelBackend.FLASHINFER_CUDNN: _CUDA,
    }
    descriptions = {
        KernelBackend.FLASHINFER_CUTEDSL: "FlashInfer CuTe DSL NVFP4 GEMM (SM100).",
        KernelBackend.FLASHINFER_CUTLASS: "FlashInfer CUTLASS NVFP4 GEMM.",
        KernelBackend.FLASHINFER_TRTLLM: "FlashInfer TRTLLM NVFP4 GEMM.",
        KernelBackend.FLASHINFER_CUDNN: "FlashInfer cuDNN NVFP4 GEMM.",
    }

    def forward_native(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_sf: torch.Tensor,
        weight_sf: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype,
        out_features: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "gemm.nvfp4: no pure-torch reference; requires flashinfer's mm_fp4 "
            "(or a weight-only Marlin fallback, selected separately by the caller)."
        )

    def _call_flashinfer(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_sf: torch.Tensor,
        weight_sf: torch.Tensor,
        alpha: torch.Tensor,
        out_dtype: torch.dtype,
        out_features: int,
        backend: str,
    ) -> torch.Tensor:
        from flashinfer import mm_fp4

        # `out_features` is only used by callers for fake-mode shape inference
        # (see modelopt_quant.fp4_gemm); flashinfer's `mm_fp4` doesn't take it.
        return mm_fp4(
            input, weight, input_sf, weight_sf, alpha, out_dtype, backend=backend
        )

    def forward_flashinfer_cutedsl(self, *args, **kwargs) -> torch.Tensor:
        return self._call_flashinfer(*args, **kwargs, backend="cute-dsl")

    def forward_flashinfer_cutlass(self, *args, **kwargs) -> torch.Tensor:
        return self._call_flashinfer(*args, **kwargs, backend="cutlass")

    def forward_flashinfer_trtllm(self, *args, **kwargs) -> torch.Tensor:
        return self._call_flashinfer(*args, **kwargs, backend="trtllm")

    def forward_flashinfer_cudnn(self, *args, **kwargs) -> torch.Tensor:
        return self._call_flashinfer(*args, **kwargs, backend="cudnn")


_NVFP4_GEMM = register_fused_op(Nvfp4GemmOp(), __name__, "_NVFP4_GEMM")


def nvfp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
    backend: Optional[KernelBackend] = None,
) -> torch.Tensor:
    """Dense NVFP4 GEMM via FlashInfer's ``mm_fp4``.

    Pass ``backend=`` to force a specific FlashInfer sub-kernel; omit it to
    auto-select.
    """
    return _NVFP4_GEMM.forward(
        input,
        weight,
        input_sf,
        weight_sf,
        alpha,
        out_dtype,
        out_features,
        backend=backend,
    )


__all__ = [
    "fp8_scaled_mm",
    "bmm_fp8",
    "dsv3_fused_a_gemm",
    "dsv3_router_gemm",
    "Nvfp4GemmOp",
    "nvfp4_gemm",
]


# LoRA SGMV Triton kernels migrated into this group (from lora/triton_ops);
# registered for inventory. Import them from their modules.
_TRITON_KERNELS = [
    ("chunked_embedding_lora_a", "chunked_embedding_lora_a_forward"),
    ("chunked_sgmv_expand", "chunked_sgmv_lora_expand_forward"),
    ("chunked_sgmv_shrink", "chunked_sgmv_lora_shrink_forward"),
    ("embedding_lora_a", "embedding_lora_a_fwd"),
    ("gate_up_lora_b", "gate_up_lora_b_fwd"),
    ("qkv_lora_b", "qkv_lora_b_fwd"),
    ("sgemm_lora_a", "sgemm_lora_a_fwd"),
    ("sgemm_lora_b", "sgemm_lora_b_fwd"),
    ("kv_b_lora_absorbed", "step_a_q_fwd"),
    ("kv_b_lora_absorbed", "step_b_q_fwd"),
    ("kv_b_lora_absorbed", "step_a_v_fwd"),
    ("kv_b_lora_absorbed", "step_b_v_fwd"),
]
for _mod, _fn in _TRITON_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"gemm.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.gemm.{_mod}:{_fn}",
        )
    )
del _mod, _fn
