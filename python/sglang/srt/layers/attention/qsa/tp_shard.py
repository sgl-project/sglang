"""Token row layout for QSA GQA KV groups on Ascend."""

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
    num_kv_groups: int
    peer_splits: list[int]
    attention_splits: list[int]

    @staticmethod
    def build(
        num_rows: int, tp_size: int, rank: int, num_kv_groups: int
    ) -> "QSAPrefillTPLayout":
        # Each contiguous group of TP ranks holds the same GQA KV head.
        group_width = tp_size // num_kv_groups
        kv_group, owner = divmod(rank, group_width)
        rows = cdiv(num_rows, tp_size)
        attention_rows = num_kv_groups * rows
        attention_start = owner * attention_rows
        indexer_start = attention_start + kv_group * rows
        return QSAPrefillTPLayout(
            indexer_slice=slice(indexer_start, indexer_start + rows),
            attention_slice=slice(attention_start, attention_start + attention_rows),
            rows_per_rank=rows,
            kv_group=kv_group,
            num_kv_groups=num_kv_groups,
            peer_splits=[
                rows if r % group_width == owner and r != rank else 0
                for r in range(tp_size)
            ],
            attention_splits=[
                attention_rows if r // group_width == kv_group else 0
                for r in range(tp_size)
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
