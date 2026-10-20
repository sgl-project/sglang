from __future__ import annotations

import torch


def scatter_req_token_ids_torch_reference(
    *,
    flat_in: torch.Tensor,
    offsets: torch.Tensor,
    req_pool_indices: torch.Tensor,
    pool_out: torch.Tensor,
) -> None:
    """Plain-PyTorch reference for :func:`launch_scatter_req_token_ids_kernel`."""
    bs = int(req_pool_indices.shape[0])
    offsets_host = offsets.detach().cpu().tolist()
    req_pool_indices_host = req_pool_indices.detach().cpu().tolist()
    flat_host = flat_in.detach().cpu()
    pool_max_context_len = int(pool_out.shape[1])

    for r in range(bs):
        start = int(offsets_host[r])
        end = int(offsets_host[r + 1])
        if end <= start:
            continue
        rp = int(req_pool_indices_host[r])
        seg = flat_host[start:end].to(torch.int32)
        write_len = min(int(seg.shape[0]), pool_max_context_len)
        if write_len <= 0:
            continue
        pool_out[rp, :write_len] = seg[:write_len].to(pool_out.device)
