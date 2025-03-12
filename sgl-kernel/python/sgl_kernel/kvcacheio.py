import torch


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    cache_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer(
        src_k, dst_k, src_v, dst_v, src_indices, dst_indices, cache_size
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
    )
