"""Embedding kernels."""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import KernelBackend, KernelSpec

register_kernel(
    KernelSpec(
        op="embeddings.vocab_parallel_embedding",
        backend=KernelBackend.TRITON,
        target=(
            "sglang.kernels.ops.embeddings.vocab_parallel_embedding:"
            "vocab_parallel_embedding"
        ),
    )
)

register_kernel(
    KernelSpec(
        op="embeddings.host_embedding_gather",
        backend=KernelBackend.TRITON,
        target=(
            "sglang.kernels.ops.embeddings.host_embedding_gather:host_embedding_gather"
        ),
    )
)

__all__ = []
