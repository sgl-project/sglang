"""NPU KV residency operators backed by the ``sgl_kernel_npu`` wheel."""


def restore_short_kv_history_inplace(
    host,
    cache,
    req_to_token,
    requests,
    lengths,
    first_positions,
    old_begin,
    old_end,
    next_begin,
    next_end,
    next_first,
    capacity,
    context_len,
    *,
    src_ptr,
    block_dim=48,
):
    """Restore absent short-request history into resident KV cache in place."""
    from sgl_kernel_npu.sparsity_driven_kv_offload import (
        restore_short_kv_history_inplace as impl,
    )

    return impl(
        host,
        cache,
        req_to_token,
        requests,
        lengths,
        first_positions,
        old_begin,
        old_end,
        next_begin,
        next_end,
        next_first,
        capacity,
        context_len,
        src_ptr=src_ptr,
        block_dim=block_dim,
    )
