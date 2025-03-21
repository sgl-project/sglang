from __future__ import annotations

from collections import deque
from enum import Enum
from typing import List

import torch
import torch.distributed as dist


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


def poll_and_all_reduce(pollers, gloo_group):
    polls = [int(poller.poll()) for poller in pollers]
    tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")
    dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
    return tensor_to_reduce.tolist()


class ReqToMetadataIdxAllocator:
    """A memory pool that maps a request to its first output token location."""

    def __init__(
        self,
        size: int,
    ):
        self.size = size
        self.free_slots = deque(list(range(size)))

    def available_size(self):
        return len(self.free_slots)

    def alloc(self) -> List[int]:
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        self.free_slots.append(free_index)
