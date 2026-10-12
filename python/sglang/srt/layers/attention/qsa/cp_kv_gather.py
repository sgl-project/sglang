"""QSA prefill-CP K/V all-gather on a side stream.

A QSA layer's indexer compression, scoring and top-k do not read the layer's
K/V, so the K/V all-gather can run underneath them. It is issued on a dedicated
stream right after the indexer's key gather, which keeps every rank issuing the
CP group's collectives in one order, and the attention waits for it.
"""

from typing import Tuple

import msgspec
import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.runtime_context import get_parallel, get_stream


def build_cp_kv_gather_index(cp_metadata, device) -> torch.Tensor:
    """For every global new token, its row in the rank-major gather of all
    ranks' physical rows ([cp * physical_rows], padding included).

    Mirrors ZigzagCPStrategy.gather_kv_cache from host metadata: each rank's
    logical rows are concatenated in rank order, split by reverse_split_len and
    reassembled in cp_reverse_index order."""
    physical_rows = cp_metadata.per_rank_actual_token[0]
    logical_rows = list(
        cp_metadata.per_rank_logical_token or cp_metadata.per_rank_actual_token
    )
    unpadded = torch.arange(sum(logical_rows), dtype=torch.int64)
    parts = torch.split(unpadded, list(cp_metadata.reverse_split_len))
    global_to_unpadded = torch.cat([parts[i] for i in cp_metadata.cp_reverse_index])
    rank_starts = torch.tensor([0] + logical_rows[:-1], dtype=torch.int64).cumsum(0)
    rank = torch.searchsorted(rank_starts, global_to_unpadded, right=True) - 1
    index = rank * physical_rows + (global_to_unpadded - rank_starts[rank])
    return index.to(device)


class CPKVGatherHandle(msgspec.Struct):
    """A K/V gather in flight; wait() orders the current stream after it."""

    key: torch.Tensor
    value: torch.Tensor
    done: torch.cuda.Event
    # Read or written by the side stream; released only after the wait, so the
    # caching allocator cannot hand them out while the gather still runs.
    keepalive: Tuple[torch.Tensor, ...]

    def wait(self) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.current_stream().wait_event(self.done)
        return self.key, self.value


class CPKVGather:
    """Issues one layer's K/V all-gather on the QSA CP side stream."""

    def __init__(self) -> None:
        self._stream = get_stream("qsa_cp_kv")
        self._fork = torch.cuda.Event()

    def issue(
        self, kv_local: torch.Tensor, key_width: int, gather_index: torch.Tensor
    ) -> CPKVGatherHandle:
        """kv_local is [physical_rows, key_width + value_width]: this rank's K|V."""
        group = get_parallel().attn_cp_group
        rows, width = kv_local.shape
        num_tokens = gather_index.numel()
        # Allocated on the consumer stream; the side stream only writes into
        # them, so no record_stream is needed once the consumer has waited.
        with use_symmetric_memory(group, disabled=not is_allocation_symmetric()):
            gathered = kv_local.new_empty((group.world_size * rows, width))
        key = kv_local.new_empty((num_tokens, key_width))
        value = kv_local.new_empty((num_tokens, width - key_width))
        done = torch.cuda.Event()
        self._fork.record()
        self._stream.wait_event(self._fork)
        with torch.cuda.stream(self._stream):
            group.all_gather_into_tensor(gathered, kv_local)
            torch.index_select(gathered[:, :key_width], 0, gather_index, out=key)
            torch.index_select(gathered[:, key_width:], 0, gather_index, out=value)
            done.record()
        return CPKVGatherHandle(
            key=key, value=value, done=done, keepalive=(kv_local, gathered)
        )
