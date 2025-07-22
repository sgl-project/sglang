from typing import List

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
            item_size * src_k.element_size(),  # todo, hot fix for compatibility
            block_quota,
            num_warps_per_block,
        )
    elif io_backend == "direct":
        torch.ops.sgl_kernel.transfer_kv_direct(
            [src_k, src_v], [dst_k, dst_v], src_indices, dst_indices, page_size
        )
    else:
        raise ValueError(f"Unsupported io backend")


def transfer_kv_per_layer_pf_lf(
    src_k: torch.Tensor,
    dst_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_pf_lf(
        src_k,
        dst_k,
        src_v,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer(
    src_k_layers: torch.Tensor,
    dst_k_layers: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    io_backend: str,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    if io_backend == "kernel":
        torch.ops.sgl_kernel.transfer_kv_all_layer(
            src_k_layers,
            dst_k_layers,
            src_v_layers,
            dst_v_layers,
            src_indices,
            dst_indices,
            item_size,
            num_layers,
            block_quota,
            num_warps_per_block,
        )
    elif io_backend == "direct":
        raise NotImplementedError("Deprecated interface")
    else:
        raise ValueError(f"Unsupported io backend")


def transfer_kv_all_layer_lf_pf(
    src_k_layers: torch.Tensor,
    dst_k: torch.Tensor,
    src_v_layers: torch.Tensor,
    dst_v: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_lf_pf(
        src_k_layers,
        dst_k,
        src_v_layers,
        dst_v,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_direct(
    src_layers: List[torch.Tensor],
    dst_layers: List[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    page_size: int,
):
    torch.ops.sgl_kernel.transfer_kv_direct(
        src_layers, dst_layers, src_indices, dst_indices, page_size
    )


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
            item_size * src.element_size(),  # todo, hot fix for compatibility
            block_quota,
            num_warps_per_block,
        )
    elif io_backend == "direct":
        torch.ops.sgl_kernel.transfer_kv_direct(
            [src], [dst], src_indices, dst_indices, page_size
        )
    else:
        raise ValueError(f"Unsupported io backend")


def transfer_kv_per_layer_mla_pf_lf(
    src: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    src_layout_dim: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    torch.ops.sgl_kernel.transfer_kv_per_layer_mla_pf_lf(
        src,
        dst,
        src_indices,
        dst_indices,
        item_size,
        src_layout_dim,
        block_quota,
        num_warps_per_block,
    )


def transfer_kv_all_layer_mla(
    src_layers: torch.Tensor,
    dst_layers: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    io_backend: str,
    item_size: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    if io_backend == "kernel":
        torch.ops.sgl_kernel.transfer_kv_all_layer_mla(
            src_layers,
            dst_layers,
            src_indices,
            dst_indices,
            item_size,
            num_layers,
            block_quota,
            num_warps_per_block,
        )
    elif io_backend == "direct":
        raise NotImplementedError("Deprecated interface")
    else:
        raise ValueError(f"Unsupported io backend")


def transfer_kv_all_layer_mla_lf_pf(
    src_layers: torch.Tensor,
    dst: torch.Tensor,
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    item_size: int,
    dst_layout_dim: int,
    num_layers: int,
    block_quota: int = 2,
    num_warps_per_block: int = 32,
):
    torch.ops.sgl_kernel.transfer_kv_all_layer_mla_lf_pf(
        src_layers,
        dst,
        src_indices,
        dst_indices,
        item_size,
        dst_layout_dim,
        num_layers,
        block_quota,
        num_warps_per_block,
    )
