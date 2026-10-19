"""Layer-normalization kernels."""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import (
    CUDA_ONLY,
    CapabilityRequirement,
    KernelBackend,
    KernelSpec,
)

# Triton / TileLang kernels migrated from srt/layers top-level strays
# (RFC #29630, Phase 2.5); registered for inventory.
_PHASE25_KERNELS = [
    ("gemma4_fused_ops", "gemma_qkv_rmsnorm", "triton"),
    ("mhc_head", "fused_hc_head", "triton"),
    ("hy4_ihc", "fused_hy4_ihc_pre", "triton"),
    ("hy4_ihc", "fused_hy4_ihc_post", "triton"),
]
for _mod, _fn, _bk in _PHASE25_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"layernorm.{_fn}",
            backend=KernelBackend(_bk),
            target=f"sglang.kernels.ops.layernorm.{_mod}:{_fn}",
        )
    )
del _mod, _fn, _bk

# Fused hyper-connection combine / norm kernels for small speculative batches.
_HC_NORM_KERNELS = [
    ("hc_combine_norm", "hc_combine_norm"),
    ("mhc_post_split_h", "mhc_post_split_h"),
    ("hc_combine_norm", "hc_combine_norm_mxfp8"),
    ("mxfp8_epilogue", "rmsnorm_mxfp8"),
]
for _mod, _fn in _HC_NORM_KERNELS:
    register_kernel(
        KernelSpec(
            op=f"layernorm.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.layernorm.{_mod}:{_fn}",
        )
    )
del _mod, _fn

# The fused-rmsnorm variants physically live in the shared fused-pointwise
# collection (sglang.kernels.ops.elementwise.elementwise) but stay layernorm ops.
for _fn in ("fused_dual_residual_rmsnorm", "fused_rmsnorm"):
    register_kernel(
        KernelSpec(
            op=f"layernorm.{_fn}",
            backend=KernelBackend.TRITON,
            target=f"sglang.kernels.ops.elementwise.elementwise:{_fn}",
        )
    )
del _fn


# Public entry points inventoried by logical operator group (RFC #29630).
register_kernel(
    KernelSpec(
        op="layernorm.rmsnorm_hf",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.layernorm.rmsnorm_hf:rmsnorm_hf",
        capabilities=CUDA_ONLY,
    )
)
register_kernel(
    KernelSpec(
        op="layernorm.grouped_gemma_rmsnorm",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm:grouped_gemma_rmsnorm",
        capabilities=CUDA_ONLY,
    )
)

register_kernel(
    KernelSpec(
        op="layernorm.rmsnorm_fanout",
        backend=KernelBackend.CUTE_DSL,
        target="sglang.kernels.ops.layernorm.rmsnorm_fanout:rmsnorm_fanout",
        capabilities=frozenset(
            {CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 9))}
        ),
    )
)
