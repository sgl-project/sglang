import torch

from sglang.srt.layers.dcp.layout import remap_dcp_sparse_indices


def remap_dcp_sparse_indices_npu(
    topk_indices: torch.Tensor, dcp_size: int, dcp_rank: int, interleave_size: int
) -> torch.Tensor:
    op = getattr(torch.ops.npu, "sgl_dcp_sparse_index_remap", None)
    if (
        op is not None
        and topk_indices.device.type == "npu"
        and topk_indices.dtype == torch.int32
        and topk_indices.ndim == 2
        and topk_indices.shape[1] == 2048
        and topk_indices.is_contiguous()
        and dcp_size in (2, 4)
        and 0 <= dcp_rank < dcp_size
        and 0 < interleave_size <= 65536
        and (interleave_size & (interleave_size - 1)) == 0
    ):
        return op(topk_indices, dcp_size, dcp_rank, interleave_size)
    return remap_dcp_sparse_indices(topk_indices, dcp_size, dcp_rank, interleave_size)
