"""Page-striped pooled index keys for CUDA DCP.

The KV allocator assigns a request complete DCP superpages. Within each
superpage consecutive 64-key index pages therefore have predictable owners
even when successive superpages of the same request are physically disjoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_dcp import kpool_selection_group
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.runtime_context import get_parallel


def padded_rows(tensor, start, width):
    """Return a view for a full partition. Repeat a valid row in padding."""
    part = tensor[start : start + width]
    if part.shape[0] == width:
        return part
    return torch.cat((part, tensor[-1:].expand(width - len(part), *tensor.shape[1:])))


@dataclass(frozen=True)
class KPoolPageFetch:
    local_pages: torch.Tensor
    staged_page_ids: torch.Tensor


@dataclass(frozen=True)
class KPoolPageShard:
    size: int
    rank: int
    world_size: int
    world_rank: int
    group: dist.ProcessGroup
    exchange: str

    @classmethod
    def create(cls, device):
        size = envs.SGLANG_OPT_DSA_KPOOL_INDEX_CACHE_SHARDS.get()
        if size == 1:
            return None
        parallel = get_parallel()
        dcp_size = parallel.attn_dcp_size
        if size < 2 or dcp_size % size:
            raise ValueError("Pooled index cache shards must divide the DCP group size")
        exchange = envs.SGLANG_OPT_DSA_KPOOL_INDEX_EXCHANGE.get()
        if exchange not in ("logits", "candidates"):
            raise ValueError("Pooled index exchange must be logits or candidates")
        dcp = parallel.dcp_group
        rank = dcp.rank_in_group
        group = dcp.device_group
        if size < dcp_size:
            start = rank // size * size
            group = dist.new_group(
                ranks=dcp.ranks[start : start + size],
                backend="nccl",
                use_local_synchronization=True,
            )
        # Initialize NCCL before the first decode CUDA graph capture.
        dist.all_reduce(torch.ones(1, device=device), group=group)
        rows = kpool_selection_group(parallel)
        return cls(
            size, rank % size, rows.world_size, rows.rank_in_group, group, exchange
        )

    def owned_pages(self, page_ids):
        return page_ids[page_ids % self.size == self.rank] // self.size

    def plan_fetch(self, page_ids, *, aligned=False):
        if aligned:
            # Every request starts on a DCP superpage. If all preceding
            # requests end on a stripe boundary, superpage starts and positions
            # in the staged tensor are known without an on-device sort or sync.
            superpages = page_ids[:: self.size].long() // self.size
            pos = torch.arange(
                page_ids.numel(), device=page_ids.device, dtype=page_ids.dtype
            )
            mapped = pos.remainder(self.size) * len(superpages) + pos // self.size
        else:
            safe = page_ids.long().clamp_min(0)
            superpages, inverse = torch.unique(safe // self.size, return_inverse=True)
            mapped = safe.remainder(self.size) * len(superpages) + inverse
        return KPoolPageFetch(superpages, mapped.to(page_ids.dtype).contiguous())

    def fetch_pages(self, buffer, page_ids, plan=None):
        """Stage only the requested superpages for the ragged prefill kernel."""
        if page_ids.numel() == 0:
            return buffer, page_ids
        if plan is None:
            plan = self.plan_fetch(page_ids)
        local = buffer.index_select(0, plan.local_pages)
        gathered = local.new_empty((len(local) * self.size, *local.shape[1:]))
        dist.all_gather_into_tensor(gathered, local, group=self.group)
        return gathered, plan.staged_page_ids

    def decode_layout(self, pooled_table, pooled_lens, page_size):
        rows = pooled_lens.numel()
        stride = (rows + self.world_size - 1) // self.world_size
        team_start = (self.world_rank - self.rank) * stride
        team_width = self.size * stride
        table = padded_rows(pooled_table, team_start, team_width)
        lens = padded_rows(pooled_lens, team_start, team_width)
        local_lens = get_dcp_lens(lens, self.size, self.rank, interleave_size=page_size)
        pages = max(1, (table.shape[1] + self.size - 1) // self.size)
        local_table = table[:, self.rank :: self.size] // self.size
        if local_table.shape[1] < pages:
            local_table = torch.nn.functional.pad(
                local_table, (0, pages - local_table.shape[1])
            )
        # Padded graph rows and short prompts may have no page on this shard.
        valid = (
            torch.arange(pages, device=table.device) * page_size < local_lens[:, None]
        )
        local_table = torch.where(valid, local_table, 0).contiguous()
        if pages == 1:
            # PyTorch permits an arbitrary stride on a singleton dimension.
            # DeepGEMM requires a unit column stride even for a one-page table.
            local_table = local_table[:, 0].unsqueeze(1)
        return stride, team_start, local_lens, local_table

    def transpose_scores(self, local, local_lens, stride, page_size, topk):
        """Exchange history columns for request rows and restore logical token order."""
        rows, cols = local.shape
        if self.exchange == "logits":
            send = local.contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send, group=self.group)
            pages = cols // page_size
            return (
                recv.view(self.size, stride, pages, page_size)
                .permute(1, 2, 0, 3)
                .reshape(stride, cols * self.size)
            )

        # A global winner must be among its shard's k best valid groups.
        valid = torch.arange(cols, device=local.device) < local_lens[:, None]
        scores, ids = local.masked_fill(~valid, -torch.inf).topk(
            min(topk, cols), sorted=False
        )
        ids = (
            (ids // page_size * self.size + self.rank) * page_size + ids % page_size
        ).int()
        send = torch.stack((scores.float().view(torch.int32), ids), dim=-1)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        candidates = (
            recv.view(self.size, stride, -1, 2).transpose(0, 1).reshape(stride, -1, 2)
        )
        # Reuse the serving kernel for global selection, page-table mapping and
        # unfinished-pool tails. IDs from distinct page stripes are disjoint.
        full = local.new_full((stride, cols * self.size), -torch.inf)
        return full.scatter_(
            1,
            candidates[..., 1].long(),
            candidates[..., 0].contiguous().view(torch.float32),
        )
