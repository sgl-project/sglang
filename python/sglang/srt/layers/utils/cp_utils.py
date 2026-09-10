"""Import-only shims for deprecated platform backends awaiting CP refactoring.

The legacy CP algorithms have been removed. These names keep the retained
NPU/MUSA attention backends importable for non-CP inference; calling them fails.
"""


def _deprecated_platform_cp():
    raise ValueError(
        "Prefill CP on HIP/NPU/MUSA is deprecated; CP support will be refactored soon."
    )


def cp_all_gather_rerange_output(input_tensor, cp_size, forward_batch, stream):
    _deprecated_platform_cp()


def cp_all_gather_rerange_kv_cache(input_tensor, cp_size, forward_batch, stream):
    _deprecated_platform_cp()

def cp_all_gather_rerange_kv_cache_launch(input_tensor, cp_size, forward_batch):
    """Launch async all-gather for KV cache. Returns (handle, input_tensor_full).

    Caller MUST call cp_all_gather_rerange_kv_cache_finalize() after handle.wait().
    """
    max_len = forward_batch.attn_cp_metadata.max_rank_len[0]
    pad_size = max_len - input_tensor.shape[0]
    if pad_size > 0:
        input_tensor = F.pad(
            input_tensor, (0, 0, 0, pad_size), mode="constant", value=0
        )
    group = get_parallel().attn_cp_group
    with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
        input_tensor_full = torch.empty(
            max_len * cp_size,
            input_tensor.shape[1],
            device=input_tensor.device,
            dtype=input_tensor.dtype,
            )

    handle = torch.distributed.all_gather_into_tensor(
        input_tensor_full, input_tensor,
        group=group.device_group, async_op=True,
    )
    return handle, input_tensor_full

def cp_all_gather_rerange_kv_cache_finalize(input_tensor_full, forward_batch):
    """Finalize async all-gather: remove padding + rearrange."""
    outputs_list_max = list(
        torch.split(
            input_tensor_full, forward_batch.attn_cp_metadata.max_rank_len, dim=0
        )
    )
    outputs = torch.cat(
        [
            outputs_list_max[index][:per_rank_len]
            for index, per_rank_len in enumerate(
            forward_batch.attn_cp_metadata.per_rank_actual_token
        )
        ],
        dim=0,
    )
    outputs_list = list(
        torch.split(outputs, forward_batch.attn_cp_metadata.reverse_split_len, dim=0)
    )
    output_tensor = torch.cat(
        [outputs_list[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index],
        dim=0,
    )
    return output_tensor

def cp_allgather_and_save_kv_cache(forward_batch, layer, k, v, cp_size, swa_loc=None):
    _deprecated_platform_cp()
