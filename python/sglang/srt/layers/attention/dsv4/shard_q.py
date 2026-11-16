"""Token-sharded prefill attention under attention TP.

Each attention-TP rank attends a zigzag share of the extend tokens with every
head: an all-to-all turns the head-sharded query [T, H / tp, D] into
[T_rank, H, D], and a second one returns the output to [T, H / tp, D].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import msgspec
import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

MIN_TOKENS = 1024


def zigzag_owner(extend_lens: List[int], world: int) -> np.ndarray:
    """The rank of every token: each request is cut into 2 * world chunks and rank r
    owns chunks r and 2 * world - 1 - r, which balances the causal work."""
    lens = np.asarray(extend_lens, dtype=np.int64)
    num_chunks = 2 * world
    pos = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    # chunk i spans [i * len // num_chunks, (i + 1) * len // num_chunks)
    chunk = ((pos + 1) * num_chunks - 1) // np.repeat(lens, lens)
    return np.minimum(chunk, num_chunks - 1 - chunk)


class ShardQ(msgspec.Struct, frozen=True, kw_only=True):
    group: GroupCoordinator  # attention TP
    rows_per_rank: int  # every rank's share padded to the largest one
    token_ids: torch.Tensor  # this rank's tokens, ascending
    positions: torch.Tensor  # their positions
    rows_per_request: List[int]  # this rank's tokens of each request
    send_ids: torch.Tensor  # [world * rows_per_rank] token sent to each slot
    recv_ids: torch.Tensor  # [T] each token's row among the gathered outputs

    @classmethod
    def build(
        cls, *, extend_lens: List[int], positions: torch.Tensor, group: GroupCoordinator
    ) -> Optional[ShardQ]:
        world, rank = group.world_size, group.rank_in_group
        owner = zigzag_owner(extend_lens, world)
        sizes = np.bincount(owner, minlength=world)
        if sizes.min() == 0:
            return None
        order = np.argsort(owner, kind="stable")
        rows = int(sizes.max())
        offsets = np.cumsum(sizes) - sizes
        slot = np.repeat(np.arange(world) * rows, sizes) + (
            np.arange(len(owner)) - np.repeat(offsets, sizes)
        )
        send_ids = np.zeros(world * rows, dtype=np.int64)
        send_ids[slot] = order
        recv_ids = np.empty(len(owner), dtype=np.int64)
        recv_ids[order] = slot
        local = order[offsets[rank] : offsets[rank] + sizes[rank]]
        request = np.repeat(np.arange(len(extend_lens)), extend_lens)
        device = positions.device
        token_ids = torch.from_numpy(local).to(device)
        return cls(
            group=group,
            rows_per_rank=rows,
            token_ids=token_ids,
            positions=positions.index_select(0, token_ids),
            rows_per_request=np.bincount(
                request[local], minlength=len(extend_lens)
            ).tolist(),
            send_ids=torch.from_numpy(send_ids).to(device),
            recv_ids=torch.from_numpy(recv_ids).to(device),
        )

    def scatter_q(self, q: torch.Tensor) -> torch.Tensor:
        """[T, H / tp, D] -> [T_rank, H, D]; head block r comes from rank r."""
        world, rows = self.group.world_size, self.rows_per_rank
        send = q.index_select(0, self.send_ids)
        recv = torch.empty_like(send)
        self.group.all_to_all_single(recv, send)
        num_local = self.token_ids.shape[0]
        recv = recv.view(world, rows, *q.shape[1:])[:, :num_local]
        return recv.transpose(0, 1).reshape(num_local, -1, q.shape[-1])

    def gather_o(self, o: torch.Tensor) -> torch.Tensor:
        """[T_rank, H, D] -> [T, H / tp, D]."""
        world, rows = self.group.world_size, self.rows_per_rank
        num_local, num_heads, dim = o.shape
        send = o.new_empty(world, rows, num_heads // world, dim)
        send[:, :num_local] = o.view(num_local, world, -1, dim).transpose(0, 1)
        recv = torch.empty_like(send)
        self.group.all_to_all_single(recv, send)
        return recv.view(world * rows, -1, dim).index_select(0, self.recv_ids)
