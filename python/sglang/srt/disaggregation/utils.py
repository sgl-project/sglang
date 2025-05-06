from __future__ import annotations

from collections import deque
from enum import Enum
from typing import List

import numpy as np
import torch
import torch.distributed as dist


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


FakeBootstrapHost = "2.2.2.2"


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


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    FAKE = "fake"


class KVClassType(Enum):
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


def get_kv_class(transfer_backend: TransferBackend, class_type: KVClassType):
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    if transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    if transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

        class_mapping = {
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # 1. The page is guaruanteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    return (num_kv_indices + page_size - 1) // page_size
