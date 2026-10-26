"""Multimodal input-processing kernels; implementations are imported lazily."""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import CUDA_ONLY, KernelBackend, KernelSpec

__all__ = ["process"]


# Public entry points inventoried by logical operator group (RFC #29630).
register_kernel(
    KernelSpec(
        op="mm.normalize_and_patchify",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.mm.process.image:normalize_and_patchify",
        capabilities=CUDA_ONLY,
    )
)

register_kernel(
    KernelSpec(
        op="mm.lanczos_resize",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.mm.process.lanczos:lanczos_resize",
        capabilities=CUDA_ONLY,
    )
)

register_kernel(
    KernelSpec(
        op="mm.normalize_and_patchify",
        backend=KernelBackend.TORCH,
        target="sglang.kernels.ops.mm.process.image:_normalize_and_patchify_torch",
    )
)
