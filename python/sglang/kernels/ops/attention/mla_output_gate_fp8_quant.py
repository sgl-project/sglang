"""K3 MLA output gate fused with per-token FP8 quant.

Only the ROCm PTPC ``o_proj`` consumes ``(fp8, scale)`` directly, so this sits
beside ``mla_output_gate`` rather than inside it: the bf16-only gate keeps its
own JIT module and is not rebuilt for a kernel CUDA never launches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.ops.attention.mla_output_gate import covered as _gate_covered

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_THREADS: int = 256


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(_THREADS, is_arch_support_pdl())
    return load_jit(
        "kimi_k3_mla_output_gate_fp8_quant_" + str(_THREADS),
        *args,
        cuda_files=["kimi_k3/mla_output_gate_fp8_quant.cuh"],
        cuda_wrappers=[("run", f"MlaOutputGateFp8QuantKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def covered(x: torch.Tensor, gate: torch.Tensor) -> bool:
    # The kernel is per-token, so unlike the flat gate it also needs 2D [T, H].
    return _gate_covered(x, gate) and x.dim() == 2 and x.shape[-1] % 8 == 0


def kimi_k3_mla_output_gate_fp8_quant(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate multiply + per-token FP8 quant. Returns ``(fp8, [T, 1] scale)``."""
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

    out_q = torch.empty(x.shape, dtype=fp8_dtype, device=x.device)
    out_s = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    _jit_module().run(x, gate, out_q.view(torch.uint8), out_s)
    return out_q, out_s.unsqueeze(1)
