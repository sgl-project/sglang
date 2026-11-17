"""Temporary chronological KV for prefill over a DCP-sharded MLA cache."""

import torch

from sglang.srt.utils.common import ceil_div


class DcpPrefillWorkspace:
    """One request's KV, reused across layers and prefill batches."""

    def __init__(self, *, context_len, page_size, dcp_size, kv_dim, dtype, device):
        num_pages = ceil_div(context_len, page_size)
        gather_rows = ceil_div(context_len, dcp_size) * dcp_size
        self.kv = torch.empty(
            (max(num_pages * page_size, gather_rows), 1, kv_dim),
            dtype=dtype,
            device=device,
        )
        self.gathered = self.kv.new_empty((gather_rows, 1, kv_dim))
        self.page_table = torch.arange(
            num_pages, dtype=torch.int32, device=device
        ).unsqueeze(0)

    def gather_prefix(self, cache, local_rows, prefix_len, group):
        if prefix_len == 0:
            return

        rank_rows = ceil_div(prefix_len, group.world_size)
        local = self.kv[:rank_rows]
        live_rows = local_rows.numel()
        torch.index_select(cache, 0, local_rows, out=local[:live_rows])
        local[live_rows:].zero_()

        gather_rows = rank_rows * group.world_size
        gathered = self.gathered[:gather_rows]
        group.all_gather_into_tensor(gathered, local)
        # The wire is rank-major. Causal attention requires sequence order.
        rank_major = gathered.view(group.world_size, rank_rows, *cache.shape[1:])
        token_major = rank_major.transpose(0, 1)
        self.kv[:gather_rows].view_as(token_major).copy_(token_major)
