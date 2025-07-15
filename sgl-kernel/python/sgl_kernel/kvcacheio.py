import torch


def transfer_kv_per_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    io_backend: str,
    page_size: int,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    if io_backend == "kernel":
        torch.ops.sgl_kernel.transfer_kv_per_layer(
            src_k,
            dst_k,
            src_v,
            dst_v,
            src_indices,
            dst_indices,
            item_size,
            block_quota,
            num_warps_per_block,
        )
    elif io_backend == "direct":
        torch.ops.sgl_kernel.transfer_kv_per_layer_direct(
            src_k, dst_k, src_v, dst_v, src_indices, dst_indices, page_size
        )
    else:
        raise ValueError(f"Unsupported io backend")


def transfer_kv_all_layer(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    io_backend: str,
    page_size: int,
    item_size: int,
    num_layers: int,
    src_layer_offset: int,
    dst_layer_offset: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    if io_backend == "kernel":
        torch.ops.sgl_kernel.transfer_kv_all_layer(
            src_k,
            dst_k,
            src_v,
            dst_v,
            src_indices,
            dst_indices,
            item_size,
            num_layers,
            src_layer_offset,
            dst_layer_offset,
            block_quota,
            num_warps_per_block,
        )
    elif io_backend == "direct":
        torch.ops.sgl_kernel.transfer_kv_all_layer_direct(
            src_k, dst_k, src_v, dst_v, src_indices, dst_indices, page_size, num_layers
        )
    else:
        raise ValueError(f"Unsupported io backend")


def transfer_kv_per_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    io_backend: str,
    page_size: int,
    item_size: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    if io_backend == "kernel":
        torch.ops.sgl_kernel.transfer_kv_per_layer_mla(
            src,
            dst,
            src_indices,
            dst_indices,
            item_size,
            block_quota,
            num_warps_per_block,
        )
    elif io_backend == "direct":
        torch.ops.sgl_kernel.transfer_kv_per_layer_mla_direct(
            src, dst, src_indices, dst_indices, page_size
        )
    else:
        raise ValueError(f"Unsupported io backend")


def transfer_kv_all_layer_mla(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    io_backend: str,
    page_size: int,
    item_size: int,
    num_layers: int,
    src_layer_offset: int,
    dst_layer_offset: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    if io_backend == "kernel":
        torch.ops.sgl_kernel.transfer_kv_all_layer_mla(
            src,
            dst,
            src_indices,
            dst_indices,
            item_size,
            num_layers,
            src_layer_offset,
            dst_layer_offset,
            block_quota,
            num_warps_per_block,
        )
    elif io_backend == "direct":
        torch.ops.sgl_kernel.transfer_kv_all_layer_mla_direct(
            src, dst, src_indices, dst_indices, page_size, num_layers
        )
    else:
        raise ValueError(f"Unsupported io backend")
