"""Per-forward/capture-bucket ownership of Ascend KeyPool write inputs and scratch."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import torch


@dataclass(frozen=True)
class AscendKPoolWorkspace:
    """Own stable buffers; accept normalized tensors, never a backend or batch.

    The last operator request consumes graph padding. Real requests retain their
    computed TH rows even when invalid or unaccepted. Persistent tails are BF16;
    only the per-layer CANN state is FP32.
    """

    num_tokens: int
    cmp_ratio: int
    head_dim: int
    index_page_size: int
    tail_capacity: int
    mode: Literal["normal", "mtp"]
    max_query_len: int

    query_start_loc: torch.Tensor
    start_pos: torch.Tensor
    commit_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    valid_requests: torch.Tensor
    state_start_pos: torch.Tensor
    state_block_table: torch.Tensor
    state_cache: torch.Tensor
    pooled_slot_mapping: torch.Tensor
    tail_write_indices: torch.Tensor
    tail_state_indices: torch.Tensor
    tail_token_indices: torch.Tensor
    _history_indices: torch.Tensor
    _history_valid: torch.Tensor

    @staticmethod
    def _buffer_specs(batch, tokens, ratio, width, write_width, max_query_len):
        requests = batch + 1
        return {
            "query_start_loc": ((batch + 2,), torch.int32),
            "start_pos": ((batch,), torch.int32),
            "commit_lens": ((batch,), torch.int32),
            "req_pool_indices": ((batch,), torch.int32),
            "valid_requests": ((batch,), torch.bool),
            "state_start_pos": ((requests,), torch.int32),
            "state_block_table": (
                (requests, 1 + (max_query_len + ratio - 1) // ratio),
                torch.int32,
            ),
            "state_cache": ((1 + 2 * requests, ratio, 2 * width), torch.float32),
            "pooled_slot_mapping": (
                (min(tokens, tokens // ratio + requests),),
                torch.int64,
            ),
            "tail_write_indices": ((requests, write_width), torch.int64),
            "tail_state_indices": ((requests, write_width), torch.int64),
            "tail_token_indices": ((requests, write_width), torch.int64),
            "_history_indices": ((batch, ratio), torch.int64),
            "_history_valid": ((batch, ratio), torch.bool),
        }

    @classmethod
    def allocate(
        cls,
        *,
        batch_size,
        num_tokens,
        cmp_ratio,
        head_dim,
        index_page_size,
        tail_capacity,
        mode,
        max_query_len,
        device,
    ):
        if mode not in ("normal", "mtp"):
            raise ValueError("KeyPool mode must be normal or mtp")
        if cmp_ratio <= 1 or tail_capacity < cmp_ratio or index_page_size <= 0:
            raise ValueError("invalid KeyPool cache geometry")
        write_width = max_query_len if mode == "mtp" else cmp_ratio
        if mode == "mtp" and tail_capacity < cmp_ratio + write_width:
            raise ValueError("KeyPool rollback ring must cover r+N rows")
        buffers = {
            name: torch.zeros(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in cls._buffer_specs(
                batch_size, num_tokens, cmp_ratio, head_dim, write_width, max_query_len
            ).items()
        }
        self = cls(
            num_tokens=num_tokens,
            cmp_ratio=cmp_ratio,
            head_dim=head_dim,
            index_page_size=index_page_size,
            tail_capacity=tail_capacity,
            mode=mode,
            max_query_len=max_query_len,
            **buffers,
        )
        request = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        self.state_block_table.copy_(
            (batch_size + 2 + request)[:, None].expand_as(self.state_block_table)
        )
        self.state_block_table[:, 0].copy_(request + 1)
        return self

    @property
    def batch_size(self) -> int:
        return self.req_pool_indices.numel()

    @classmethod
    def estimate_bytes(cls, batch: int, tokens: int, ratio: int, width: int) -> int:
        """Conservative buffer/adapter bound; CANN workspace is profiled separately."""
        specs = cls._buffer_specs(
            batch, tokens, ratio, width, max(ratio, tokens // max(batch, 1)), tokens
        )
        buffers = sum(
            prod(shape) * torch._utils._element_size(dtype)
            for shape, dtype in specs.values()
        )
        temporary = batch * ratio * (width * 8 + 17) + tokens * width * 32
        temporary += batch * ((tokens + ratio - 1) // ratio) * 96
        return buffers + temporary

    def update_inputs(
        self,
        *,
        start_pos,
        query_lens,
        commit_lens,
        req_pool_indices,
        valid_requests,
        index_page_table,
    ):
        """Refresh all request inputs and final write addresses in place."""
        for name, value in (
            ("start_pos", start_pos),
            ("commit_lens", commit_lens),
            ("req_pool_indices", req_pool_indices),
            ("valid_requests", valid_requests),
        ):
            target = getattr(self, name)
            if value.shape != target.shape:
                raise RuntimeError(f"KeyPool {name} shape exceeds allocated workspace")
        if (
            query_lens.shape != self.start_pos.shape
            or index_page_table.shape[0] != self.batch_size
        ):
            raise RuntimeError(
                "KeyPool query/page-table batch shape differs from workspace"
            )
        for name, value in (
            ("start_pos", start_pos),
            ("commit_lens", commit_lens),
            ("req_pool_indices", req_pool_indices),
            ("valid_requests", valid_requests),
        ):
            getattr(self, name).copy_(value)
        self.query_start_loc[0].zero_()
        self.query_start_loc[1:-1].copy_(query_lens.cumsum(0))
        self.query_start_loc[-1] = self.num_tokens
        self.state_start_pos[:-1].copy_(start_pos % self.cmp_ratio)
        self.state_start_pos[-1].zero_()
        self._update_write_mappings(index_page_table)

    def _update_write_mappings(self, index_page_table):
        ratio, page_size = self.cmp_ratio, self.index_page_size
        batch = self.batch_size
        p = self.start_pos.long()
        q = (self.query_start_loc[1 : batch + 1] - self.query_start_loc[:batch]).long()
        active = self.valid_requests
        owner = torch.where(active, self.req_pool_indices.long(), 0)
        slots = self.pooled_slot_mapping
        self.tail_write_indices.fill_(-1)
        self.tail_state_indices.fill_(-1)
        self.tail_token_indices.fill_(-1)
        # Count every computed group, including invalid requests; acceptance
        # only masks destinations and never shifts the next request's source.
        count = (p + q) // ratio - p // ratio
        offsets = count.cumsum(0) - count
        j = torch.arange((self.max_query_len + ratio - 1) // ratio, device=p.device)
        source = offsets[:, None] + j
        computed = (j < count[:, None]) & (source < slots.numel())
        accepted = torch.minimum(self.commit_lens.long().clamp_min(0), q)
        committed = (p + accepted) // ratio - p // ratio
        group = p[:, None] // ratio + j
        column = group // page_size
        valid = (
            active[:, None]
            & computed
            & (j < committed[:, None])
            & (column >= 0)
            & (column < index_page_table.shape[1])
        )
        if index_page_table.shape[1]:
            page = index_page_table.long().gather(
                1, column.clamp(0, index_page_table.shape[1] - 1)
            )
            destination = page * page_size + group % page_size
            valid = valid & (page >= 0)
        else:
            destination = torch.full_like(group, -1)
        mapping = slots.new_full((slots.numel() + 1,), -1)
        mapping.scatter_(
            0,
            torch.where(computed, source, slots.numel()).flatten(),
            torch.where(valid, destination, -1).flatten(),
        )
        slots.copy_(mapping[:-1])

        col = torch.arange(self.tail_write_indices.shape[1], device=p.device)
        if self.mode == "mtp":
            position = p[:, None] + col
            valid = active[:, None] & (col < q[:, None])
            token = self.query_start_loc[:batch].long()[:, None] + col
            self.tail_token_indices[:batch].copy_(torch.where(valid, token, -1))
        else:
            end = p + q
            remain = end % ratio
            position = end[:, None] - remain[:, None] + col
            valid = active[:, None] & (col < remain[:, None])
            request = torch.arange(batch, device=p.device)
            block = 1 + request + torch.where(p % ratio + q >= ratio, batch + 1, 0)
            state_row = block[:, None] * ratio + col
            self.tail_state_indices[:batch].copy_(torch.where(valid, state_row, -1))
        destination = (
            owner[:, None] * self.tail_capacity + position % self.tail_capacity
        )
        self.tail_write_indices[:batch].copy_(torch.where(valid, destination, -1))

        col = torch.arange(ratio, device=p.device)
        history = p[:, None] - p[:, None] % ratio + col
        self._history_indices.copy_(
            owner[:, None] * self.tail_capacity + history % self.tail_capacity
        )
        self._history_valid.copy_(active[:, None] & (col < p[:, None] % ratio))

    def pack_history(
        self, tail_k: torch.Tensor, tail_gate: torch.Tensor
    ) -> torch.Tensor:
        """Gather BF16 rings into FP32 state; write mappings are already complete."""
        if tail_k.shape != tail_gate.shape or tail_k.shape[1:] != (
            self.tail_capacity,
            self.head_dim,
        ):
            raise ValueError("persistent tail shape differs from workspace geometry")
        if tail_k.dtype != torch.bfloat16 or tail_gate.dtype != torch.bfloat16:
            raise ValueError("persistent KeyPool tails must be BF16")
        key = tail_k.view(-1, self.head_dim)[self._history_indices].float()
        gate = tail_gate.view(-1, self.head_dim)[self._history_indices].float()
        self.state_cache.zero_()
        self.state_cache[1 : self.batch_size + 1, :, : self.head_dim].copy_(
            torch.where(self._history_valid[..., None], key, 0)
        )
        self.state_cache[1 : self.batch_size + 1, :, self.head_dim :].copy_(
            torch.where(self._history_valid[..., None], gate, 0)
        )
        return self.state_cache
