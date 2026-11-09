"""Select the NPU sparse-KV implementation before allocating its state."""


def create_sparse_kv_manager(
    req_pool, allocator, sparse_context_len, *, fused, options
):
    if fused:
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_manager import (
            SparseKVCacheManager,
        )

        return SparseKVCacheManager(req_pool, allocator, sparse_context_len, **options)

    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
        SparseKVCacheManager,
    )

    return SparseKVCacheManager(req_pool, allocator, sparse_context_len)


def get_sparse_kv_attention(*, fused):
    if fused:
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.fused_attention import (
            forward_sparsity_driven_kv_offload,
        )
    else:
        from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.attention import (
            forward_sparsity_driven_kv_offload,
        )

    return forward_sparsity_driven_kv_offload
