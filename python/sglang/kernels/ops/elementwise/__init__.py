"""Generic elementwise / fused-pointwise kernels.

Home for cross-cutting pointwise kernels that do not belong to a single
functional group: the fused-pointwise Triton collection (``elementwise``:
sigmoid-mul, gated-activation and fused-rmsnorm variants shared across models)
and the ``add_constant`` JIT reference kernel used by the developer guide.
Individual functions register (or are imported) under the functional op id
they logically belong to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

__all__ = ["BeamAttentionGateOp"]


# Public entry points inventoried by logical operator group (RFC #29630).
register_kernel(
    KernelSpec(
        op="elementwise.fused_qwen4_gate_value",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.elementwise.qwen4_gate:fused_qwen4_gate_value",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)
register_kernel(
    KernelSpec(
        op="elementwise.fused_qwen4_gate_reduce",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.elementwise.qwen4_gate:fused_qwen4_gate_reduce",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)
register_kernel(
    KernelSpec(
        op="elementwise.row_scale_bf16",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.elementwise.row_scale:row_scale_bf16",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)


class BeamAttentionGateOp(BaseFusedOp):
    """Apply a per-head bias and sigmoid gate to flat attention output."""

    op = "elementwise.beam_attention_gate"
    capabilities = {KernelBackend.TRITON: {CapabilityRequirement.CUDA}}
    format_signature = FormatSignature(supported_dtypes=("float16", "bfloat16"))

    def _torch_compile_forward(self, num_tokens: int):
        # Native compilation can remove the required intermediate rounding.
        return None

    def forward_torch_compile(self, *args, **kwargs):
        import torch

        if self._compiled_native is None:
            # Register and warm the opaque op before tracing its dispatch wrapper.
            self.forward_triton(*args, **kwargs)
            self._compiled_native = torch.compile(
                self.forward_triton, options={"emulate_precision_casts": True}
            )
        return self._compiled_native(*args, **kwargs)

    def forward_native(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        gate_bias: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
        import torch

        heads = gate_bias.numel()
        multiplier = torch.sigmoid(
            gate.reshape(-1, heads, head_dim) + gate_bias[None, :, None]
        )
        return (attention_output.reshape(-1, heads, head_dim) * multiplier).reshape_as(
            attention_output
        )

    def forward_triton(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        gate_bias: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
        import torch

        if not attention_output.is_cuda or attention_output.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            return self.forward_native(attention_output, gate, gate_bias, head_dim)
        from sglang.kernels.ops.elementwise.beam_fused import fused_attention_gate

        return fused_attention_gate(attention_output, gate, gate_bias, head_dim)


_BEAM_ATTENTION_GATE = register_fused_op(
    BeamAttentionGateOp(), __name__, "_BEAM_ATTENTION_GATE"
)
