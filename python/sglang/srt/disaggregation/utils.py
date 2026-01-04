from __future__ import annotations

import os
import random
import logging
from collections import deque
from contextlib import nullcontext
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)
#########################
# Constants & Enums
#########################
FAKE_BOOTSTRAP_HOST = "2.2.2.2"
NVSHMEM_PWRITE_MODE = os.getenv("SGLANG_NVSHMEM_PWRITE_MODE", "0").lower() in (
    "1",
    "true",
)


class DisaggregationMode(Enum):
    NULL = "null"
    PREFILL = "prefill"
    DECODE = "decode"


#########################
# Synchronization
#########################

# env var for testing failure, convert to float explicitly
FAILURE_PROB = float(os.getenv("DISAGGREGATION_TEST_FAILURE_PROB", 0))


_POLL_LOCAL_FIRST = os.getenv("SGLANG_DISAGG_LOCAL_POLL", "0").lower() in (
    "1",
    "true",
)
_POLL_LOCAL_CYCLES = max(1, int(os.getenv("SGLANG_DISAGG_LOCAL_POLL_CYCLES", "1")))
_POLL_LOCAL_STATE = {"counter": 0}


def _should_sync_collective() -> bool:
    if not _POLL_LOCAL_FIRST:
        return True
    counter = (_POLL_LOCAL_STATE["counter"] + 1) % _POLL_LOCAL_CYCLES
    _POLL_LOCAL_STATE["counter"] = counter
    return counter == 0


def poll_and_all_reduce(pollers, gloo_group):
    # at a certain prob, the poll is failed to simulate failure
    if FAILURE_PROB > 0:
        from sglang.srt.disaggregation.base import KVPoll

        polls = [
            int(KVPoll.Failed) if random.random() < FAILURE_PROB else int(poller.poll())
            for poller in pollers
        ]
    else:
        polls = [int(poller.poll()) for poller in pollers]
    if _should_sync_collective():
        tensor_to_reduce = torch.tensor(polls, dtype=torch.uint8, device="cpu")

        #import time

        #start_time = time.perf_counter()
        dist.all_reduce(tensor_to_reduce, op=dist.ReduceOp.MIN, group=gloo_group)
        #duration = (time.perf_counter() - start_time) * 1000
        #if duration > 0.5:
        #    logger.warning(f"Slow poll_and_all_reduce: {duration:.2f} ms")
        return tensor_to_reduce.tolist()

    # Local-fast path: return per-rank polls without a collective to avoid blocking TTFT.
    return polls


#########################
# Metadata Buffers
#########################


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

    def alloc(self) -> Optional[int]:
        if len(self.free_slots) == 0:
            return None

        return self.free_slots.popleft()

    def free(self, free_index: int):
        self.free_slots.append(free_index)


class MetadataBuffers:
    def __init__(
        self,
        size: int,
        hidden_size: int,
        hidden_states_dtype: torch.dtype,
        max_top_logprobs_num: int = 128,
        custom_mem_pool: torch.cuda.MemPool = None,
        device: Optional[str] = None,
        nvshmem_tensor_factory: Optional[
            Callable[[Tuple[int, ...], torch.dtype], torch.Tensor]
        ] = None,
        require_nvshmem: bool = False,
        nvshmem_use_peer_view: bool = False,
    ):
        self.custom_mem_pool = custom_mem_pool
        self.nvshmem_tensor_factory = nvshmem_tensor_factory
        self.require_nvshmem = require_nvshmem
        self.uses_nvshmem = nvshmem_tensor_factory is not None
        self.nvshmem_use_peer_view = nvshmem_use_peer_view
        device = device or "cpu"
        if is_npu():
            # For ascend backend, output tokens are placed in the NPU and will be transferred by D2D channel.
            device = "npu"
        elif self.uses_nvshmem:
            device = "cuda"
        elif self.custom_mem_pool:
            # TODO(shangming): Fix me (use 'cuda') when nvlink_transport of Mooncake is bug-free
            device = "cpu"
        self.device = device

        def _alloc(shape: Tuple[int, ...], tensor_dtype: torch.dtype, peer_view: bool = False):
            if self.uses_nvshmem and self.nvshmem_tensor_factory is not None:
                try:
                    tensor = self.nvshmem_tensor_factory(shape, tensor_dtype)
                    if peer_view and self.nvshmem_use_peer_view:
                        return self._peer_view(tensor)
                    return tensor
                except Exception as exc:  # pragma: no cover
                    if self.require_nvshmem:
                        raise
                    logger.warning(
                        "NVSHMEM metadata allocation failed (%s); falling back to %s tensors.",
                        exc,
                        device,
                    )
                    self.uses_nvshmem = False
            return torch.zeros(shape, dtype=tensor_dtype, device=device)

        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # TODO: abort top_logprobs_num > 128 in PD

            # We transfer the metadata of first output token to decode
            # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
            self.output_ids = _alloc((size, 16), torch.int32, peer_view=True)

            self.cached_tokens = torch.zeros(
                (size, 16), dtype=torch.int32, device=device
            )
            self.output_token_logprobs_val = _alloc((size, 16), torch.float32, peer_view=True)
            self.output_token_logprobs_idx = _alloc((size, 16), torch.int32, peer_view=True)
            self.output_top_logprobs_val = _alloc(
                (size, max_top_logprobs_num), torch.float32, peer_view=True
            )
            self.output_top_logprobs_idx = _alloc(
                (size, max_top_logprobs_num), torch.int32, peer_view=True
            )
            # For PD + spec decode
            self.output_topk_p = torch.zeros(
                (size, 16), dtype=torch.float32, device=device
            )
            self.output_topk_index = torch.zeros(
                (size, 16), dtype=torch.int64, device=device
            )
            self.output_hidden_states = _alloc((size, hidden_size), hidden_states_dtype)
            self.transfer_status = _alloc((size,), torch.int32)

    def get_buf_infos(self):
        ptrs = [
            self.output_ids.data_ptr(),
            self.cached_tokens.data_ptr(),
            self.output_token_logprobs_val.data_ptr(),
            self.output_token_logprobs_idx.data_ptr(),
            self.output_top_logprobs_val.data_ptr(),
            self.output_top_logprobs_idx.data_ptr(),
            self.output_topk_p.data_ptr(),
            self.output_topk_index.data_ptr(),
            self.output_hidden_states.data_ptr(),
        ]
        data_lens = [
            self.output_ids.nbytes,
            self.cached_tokens.nbytes,
            self.output_token_logprobs_val.nbytes,
            self.output_token_logprobs_idx.nbytes,
            self.output_top_logprobs_val.nbytes,
            self.output_top_logprobs_idx.nbytes,
            self.output_topk_p.nbytes,
            self.output_topk_index.nbytes,
            self.output_hidden_states.nbytes,
        ]
        item_lens = [
            self.output_ids[0].nbytes,
            self.cached_tokens[0].nbytes,
            self.output_token_logprobs_val[0].nbytes,
            self.output_token_logprobs_idx[0].nbytes,
            self.output_top_logprobs_val[0].nbytes,
            self.output_top_logprobs_idx[0].nbytes,
            self.output_topk_p[0].nbytes,
            self.output_topk_index[0].nbytes,
            self.output_hidden_states[0].nbytes,
        ]
        return ptrs, data_lens, item_lens

    def get_buf(self, idx: int):
        return (
            self.output_ids[idx],
            self.cached_tokens[idx],
            self.output_token_logprobs_val[idx],
            self.output_token_logprobs_idx[idx],
            self.output_top_logprobs_val[idx],
            self.output_top_logprobs_idx[idx],
            self.output_topk_p[idx],
            self.output_topk_index[idx],
            self.output_hidden_states[idx],
        )

    def set_buf(self, req: Req):
        if self.uses_nvshmem:
            self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
            self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
            if req.return_logprob:
                if req.output_token_logprobs_val:  # not none or empty list
                    self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                        req.output_token_logprobs_val[0]
                    )
                if req.output_token_logprobs_idx:  # not none or empty list
                    self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                        req.output_token_logprobs_idx[0]
                    )

                if req.output_top_logprobs_val:  # not none or empty list
                    self.output_top_logprobs_val[req.metadata_buffer_index][
                        : len(req.output_top_logprobs_val[0])
                    ] = torch.tensor(
                        req.output_top_logprobs_val[0],
                        dtype=torch.float32,
                        device=self.output_top_logprobs_val.device,
                    )
                if req.output_top_logprobs_idx:  # not none or empty list
                    self.output_top_logprobs_idx[req.metadata_buffer_index][
                        : len(req.output_top_logprobs_idx[0])
                    ] = torch.tensor(
                        req.output_top_logprobs_idx[0],
                        dtype=torch.int32,
                        device=self.output_top_logprobs_idx.device,
                    )
        else:
            self.output_ids[req.metadata_buffer_index][0] = req.output_ids[0]
            self.cached_tokens[req.metadata_buffer_index][0] = req.cached_tokens
            if req.return_logprob:
                if req.output_token_logprobs_val:  # not none or empty list
                    self.output_token_logprobs_val[req.metadata_buffer_index][0] = (
                        req.output_token_logprobs_val[0]
                    )
                if req.output_token_logprobs_idx:  # not none or empty list
                    self.output_token_logprobs_idx[req.metadata_buffer_index][0] = (
                        req.output_token_logprobs_idx[0]
                    )

                if req.output_top_logprobs_val:  # not none or empty list
                    self.output_top_logprobs_val[req.metadata_buffer_index][
                        : len(req.output_top_logprobs_val[0])
                    ] = torch.tensor(
                        req.output_top_logprobs_val[0], dtype=torch.float32, device="cpu"
                    )
                if req.output_top_logprobs_idx:  # not none or empty list
                    self.output_top_logprobs_idx[req.metadata_buffer_index][
                        : len(req.output_top_logprobs_idx[0])
                    ] = torch.tensor(
                        req.output_top_logprobs_idx[0], dtype=torch.int32, device="cpu"
                    )
        # For PD + spec decode
        if req.hidden_states_tensor is not None:
            # speculative_eagle_topk should not be greater than 16 currently
            topk = req.output_topk_p.size(0)

            self.output_topk_p[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_p
            )
            self.output_topk_index[req.metadata_buffer_index, :topk].copy_(
                req.output_topk_index
            )
            self.output_hidden_states[req.metadata_buffer_index].copy_(
                req.hidden_states_tensor
            )

    def mark_transfer_done(self, idx: int):
        if hasattr(self, "transfer_status"):
            self.transfer_status[idx] = 1

    def reset_transfer_status(self, idx: int):
        if hasattr(self, "transfer_status"):
            self.transfer_status[idx] = 0

    def _peer_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return a view on the peer rank when NVSHMEM is active, else local tensor."""

        if not self.uses_nvshmem:
            return tensor
        try:
            from sglang.srt.disaggregation.nvshmem import nvshmem_utils

            peer_fn = nvshmem_utils.get_peer_tensor_fn()
            peer_rank = nvshmem_utils.get_peer_rank()
            if peer_fn is not None and peer_rank is not None:
                return peer_fn(tensor, peer_rank)
        except Exception:
            pass
        return tensor

    def get_all_tensors(self) -> List[torch.Tensor]:
        tensors = [
            self.output_ids,
            self.output_token_logprobs_val,
            self.output_token_logprobs_idx,
            self.output_top_logprobs_val,
            self.output_top_logprobs_idx,
            self.output_hidden_states,
        ]
        if hasattr(self, "transfer_status"):
            tensors.append(self.transfer_status)
        return tensors


#########################
# Transfer Backend
#########################


class TransferBackend(Enum):
    MOONCAKE = "mooncake"
    NIXL = "nixl"
    NVSHMEM = "nvshmem"
    ASCEND = "ascend"
    FAKE = "fake"


class KVClassType(Enum):
    KVARGS = "kvargs"
    MANAGER = "manager"
    SENDER = "sender"
    RECEIVER = "receiver"
    BOOTSTRAP_SERVER = "bootstrap_server"


def get_kv_class(
    transfer_backend: TransferBackend, class_type: KVClassType
) -> Optional[Type]:
    from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

    if transfer_backend == TransferBackend.MOONCAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.mooncake import (
            MooncakeKVBootstrapServer,
            MooncakeKVManager,
            MooncakeKVReceiver,
            MooncakeKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: MooncakeKVManager,
            KVClassType.SENDER: MooncakeKVSender,
            KVClassType.RECEIVER: (MooncakeKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: MooncakeKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.ASCEND:
        from sglang.srt.disaggregation.ascend import (
            AscendKVBootstrapServer,
            AscendKVManager,
            AscendKVReceiver,
            AscendKVSender,
        )
        from sglang.srt.disaggregation.base import KVArgs

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: AscendKVManager,
            KVClassType.SENDER: AscendKVSender,
            KVClassType.RECEIVER: (AscendKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: AscendKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NIXL:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.nixl import (
            NixlKVBootstrapServer,
            NixlKVManager,
            NixlKVReceiver,
            NixlKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.MANAGER: NixlKVManager,
            KVClassType.SENDER: NixlKVSender,
            KVClassType.RECEIVER: (NixlKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NixlKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.NVSHMEM:
        from sglang.srt.disaggregation.nvshmem.conn import (
            NVSHMEMKVArgs,
            NVSHMEMKVBootstrapServer,
            NVSHMEMKVManager,
            NVSHMEMKVReceiver,
            NVSHMEMKVSender,
        )

        class_mapping = {
            KVClassType.KVARGS: NVSHMEMKVArgs,
            KVClassType.MANAGER: NVSHMEMKVManager,
            KVClassType.SENDER: NVSHMEMKVSender,
            KVClassType.RECEIVER: (NVSHMEMKVReceiver),
            KVClassType.BOOTSTRAP_SERVER: NVSHMEMKVBootstrapServer,
        }
        return class_mapping.get(class_type)
    elif transfer_backend == TransferBackend.FAKE:
        from sglang.srt.disaggregation.base import KVArgs
        from sglang.srt.disaggregation.fake import FakeKVReceiver, FakeKVSender

        class_mapping = {
            KVClassType.KVARGS: KVArgs,
            KVClassType.SENDER: FakeKVSender,
            KVClassType.RECEIVER: (FakeKVReceiver),
        }
        return class_mapping.get(class_type)

    raise ValueError(f"Unsupported transfer backend: {transfer_backend}")


#########################
# KV Pages
#########################


def kv_to_page_indices(kv_indices: np.ndarray, page_size: int):
    # 1. The page is guaranteed to be full except the last page.
    # 2. page index = kv_index // page_size
    # The return vector is kv_indices[::page_size] // page_size
    if page_size == 1:  # shortcut
        return kv_indices

    return kv_indices[::page_size] // page_size


def kv_to_page_num(num_kv_indices: int, page_size: int):
    # ceil(num_kv_indices / page_size)
    return (num_kv_indices + page_size - 1) // page_size


#########################
# Misc
#########################


def is_mla_backend(target_kv_pool) -> bool:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return isinstance(target_kv_pool, MLATokenToKVPool)


def prepare_abort(req: Req, error_message: str, status_code=None):
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    # populate finish metadata and stream output
    req.finished_reason = FINISH_ABORT(error_message, status_code)

    if req.return_logprob:
        req.input_token_logprobs_val = []
        req.input_token_logprobs_idx = []
        req.input_top_logprobs_val = []
        req.input_top_logprobs_idx = []
        req.input_token_ids_logprobs_val = []
        req.input_token_ids_logprobs_idx = []
