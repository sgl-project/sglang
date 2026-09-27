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
        op="embeddings.engram_gather",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.embeddings.engram_gather:engram_gather",
    )
)

register_kernel(
    KernelSpec(
        op="embeddings.engram_hash_ids",
        backend=KernelBackend.TRITON,
        target="sglang.kernels.ops.embeddings.engram_hash:engram_hash_ids",
    )
)

__all__ = []
