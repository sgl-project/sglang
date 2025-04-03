import torch


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    cache_size: int,
    block_quota: int = 4,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer(
        src_k, dst_k, src_v, dst_v, src_indices, dst_indices, cache_size, block_quota
    )


def transfer_kv_all_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    cache_size: int,
    num_layers: int,
    src_layer_offset: int,
    dst_layer_offset: int,
    block_quota: int = 4,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        cache_size,
        num_layers,
        src_layer_offset,
        dst_layer_offset,
        block_quota,
    )


def transfer_kv_to_cpu_all_layer_naive(
    host_indices: torch.Tensor,
    host_k_buffer: torch.Tensor,
    host_v_buffer: torch.Tensor,
    device_indices: torch.Tensor,
    device_k_buffer: torch.Tensor,
    device_v_buffer: torch.Tensor,
    page_size: int,
    layer_num: int,
):
    torch.ops.sgl_kernel.transfer_kv_to_cpu_all_layer_naive(
        host_indices,
        host_k_buffer,
        host_v_buffer,
        device_indices,
        device_k_buffer,
        device_v_buffer,
        page_size,
        layer_num,
    )


def transfer_kv_to_gpu_per_layer_naive(
    host_indices: torch.Tensor,
    host_k_buffer: torch.Tensor,
    host_v_buffer: torch.Tensor,
    device_indices: torch.Tensor,
    device_k_buffer: torch.Tensor,
    device_v_buffer: torch.Tensor,
    page_size: int,
    layer_id: int,
):
    torch.ops.sgl_kernel.transfer_kv_to_gpu_per_layer_naive(
        host_indices,
        host_k_buffer,
        host_v_buffer,
        device_indices,
        device_k_buffer,
        device_v_buffer,
        page_size,
        layer_id,
    )
