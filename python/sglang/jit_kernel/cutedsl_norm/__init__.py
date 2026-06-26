"""CuTe-DSL RMSNorm kernels vendored from FlashInfer (flashinfer/norm/kernels/rmsnorm.py).

Provides PDL-enabled RMSNorm variants for the q/k norm use-case:
- rmsnorm_cute: 2D RMSNorm (also Gemma variant via weight_bias=1.0)
- qk_rmsnorm_cute: per-head RMSNorm for 3D tensors [batch, heads, head_dim]
- rmsnorm_quant_cute: RMSNorm + FP8 E4M3 quantization
"""

from .rmsnorm import (
    QKRMSNormKernel,
    RMSNormFusedParallelKernel,
    RMSNormKernel,
    RMSNormQuantKernel,
    qk_rmsnorm_cute,
    rmsnorm_cute,
    rmsnorm_fused_parallel_cute,
    rmsnorm_quant_cute,
)

__all__ = [
    "RMSNormKernel",
    "QKRMSNormKernel",
    "RMSNormFusedParallelKernel",
    "RMSNormQuantKernel",
    "rmsnorm_cute",
    "qk_rmsnorm_cute",
    "rmsnorm_fused_parallel_cute",
    "rmsnorm_quant_cute",
]
