"""Token row layout for the two QSA GQA KV groups on Ascend."""

from typing import NamedTuple

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import cdiv


class QSAPrefillTPLayout(NamedTuple):
    indexer_slice: slice
    attention_slice: slice
    rows_per_rank: int
    kv_group: int
    peer_splits: list[int]
    attention_splits: list[int]

    @staticmethod
    def build(num_rows: int, tp_size: int, rank: int) -> "QSAPrefillTPLayout":
        # TP partitions Q heads contiguously. Each half of the ranks holds
        # one replicated GQA KV head, and both halves use the same row order.
        group_width = tp_size // 2
        kv_group, owner = divmod(rank, group_width)
        rows = cdiv(num_rows, tp_size)
        attention_start = 2 * owner * rows
        indexer_start = attention_start + kv_group * rows
        partner = (rank + group_width) % tp_size
        return QSAPrefillTPLayout(
            indexer_slice=slice(indexer_start, indexer_start + rows),
            attention_slice=slice(attention_start, attention_start + 2 * rows),
            rows_per_rank=rows,
            kv_group=kv_group,
            peer_splits=[rows if r == partner else 0 for r in range(tp_size)],
            attention_splits=[
                2 * rows if r // group_width == kv_group else 0 for r in range(tp_size)
            ],
        )


def qsa_all_to_all(tensor: torch.Tensor, splits: list[int]) -> torch.Tensor:
    output = torch.empty_like(tensor)
    dist.all_to_all_single(
        output,
        tensor,
        output_split_sizes=splits,
        input_split_sizes=splits,
        group=get_parallel().attn_tp_group.device_group,
    )
    return output
